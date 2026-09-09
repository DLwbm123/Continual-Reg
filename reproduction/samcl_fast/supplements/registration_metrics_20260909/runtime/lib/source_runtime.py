"""Exact-step real-data runner for the pinned Continual-Reg implementation."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from core.datasets.continual3d import Continual3D
from core.methods.mer import MER
from core.methods.mersam import MERSAM
from core.methods.sgd import Sgd

from .samcl_runtime import RunConfig, TASK_ORDER, budget_steps, capture_rng_state, restore_rng_state
from .resume_evidence import inherit_resume_evidence


class StatefulBatchStream:
    """Deterministic shuffled batches whose exact position can be checkpointed."""

    def __init__(self, dataset: Continual3D, batch_size: int, seed: int):
        if len(dataset) < batch_size:
            raise ValueError(f"Dataset length {len(dataset)} is smaller than batch size {batch_size}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(dataset)).astype(np.int64)
        self.position = 0
        self.epoch = 0

    def next_indices(self) -> List[int]:
        if self.position + self.batch_size > len(self.order):
            self.order = self.rng.permutation(len(self.dataset)).astype(np.int64)
            self.position = 0
            self.epoch += 1
        result = self.order[self.position:self.position + self.batch_size].tolist()
        self.position += self.batch_size
        return result

    def state_dict(self) -> Dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "order": self.order,
            "position": self.position,
            "epoch": self.epoch,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state["batch_size"]) != self.batch_size:
            raise ValueError("Batch size changed across resume")
        order = np.asarray(state["order"], dtype=np.int64)
        if len(order) != len(self.dataset):
            raise ValueError("Dataset length changed across resume")
        self.order = order
        self.position = int(state["position"])
        self.epoch = int(state["epoch"])
        self.rng.bit_generator.state = state["rng_state"]


def _source_paths() -> Tuple[Path, Path]:
    samcl_root = Path(os.environ.get("SAMCL_ROOT", "/remote-home/wangbomin/Continual-Reg-repro"))
    source_root = samcl_root / "src/Continual-Reg"
    deep_kit_root = samcl_root / "src/deep_kit"
    if not source_root.is_dir() or not deep_kit_root.is_dir():
        raise FileNotFoundError(f"Pinned source roots are missing under {samcl_root}")
    return source_root, deep_kit_root


def build_source_cfg(config: RunConfig, device: torch.device, run_dir: Path):
    source_root, deep_kit_root = _source_paths()
    method_file = {"multitask": "sgd.yml", "mer": "mer.yml", "samcl": "mersam.yml"}[config.method]
    cfg = OmegaConf.merge(
        OmegaConf.load(deep_kit_root / "src/deep_kit/cfgs/base.yml"),
        OmegaConf.load(source_root / "cfgs/default/experiment.yml"),
        OmegaConf.load(source_root / "cfgs/default/models/continual_reg.yml"),
        OmegaConf.load(source_root / "cfgs/default/datasets/continual3d.yml"),
        OmegaConf.load(source_root / f"cfgs/{method_file}"),
    )
    cfg.exp.name = f"samcl_fast_{config.method}_{config.budget}"
    cfg.exp.names_exp_delete = None
    cfg.exp.path_save = str(run_dir.parent)
    cfg.exp.idx_device = 0
    cfg.exp.rand_seed = config.seed
    cfg.exp.n_workers = 0
    cfg.exp.mode = "train"
    cfg.exp.train.batch_size = 4
    cfg.exp.val.batch_size = 4
    cfg.exp.test.batch_size = 4
    cfg.exp.train.optimizer.lr = config.learning_rate
    cfg.exp.train.optimizer.adam.weight_decay = 0.0
    cfg.dataset.root = str(Path(config.data_root).resolve())
    cfg.dataset.size_img = [112, 96, 112]
    cfg.dataset.intensity_aug = False
    cfg.model.task_sequential = config.method != "multitask"
    cfg.method.er.buffer_size = config.buffer_size
    cfg.method.mer.beta = config.beta
    cfg.method.sam.rho = config.rho
    cfg.method.sam.adaptive = config.adaptive
    cfg.method.sam.weight_decay = config.weight_decay
    cfg.var = OmegaConf.create({}, flags={"allow_objects": True})
    cfg.var.obj_operator = SimpleNamespace(device=device, path_exp=str(run_dir), task_idx=0)
    return cfg


def _resolved_cfg(cfg) -> Dict[str, object]:
    keys = [key for key in cfg.keys() if key != "var"]
    copy_cfg = OmegaConf.masked_copy(cfg, keys)
    return OmegaConf.to_container(copy_cfg, resolve=True)


def _model_for(method: str, cfg):
    if method == "multitask":
        return Sgd(cfg)
    if method == "mer":
        return MER(cfg)
    if method == "samcl":
        return MERSAM(cfg)
    raise ValueError(method)


def _optimizer(model, method: str):
    return model.opt_sam.base_optimizer if method == "samcl" else model.opt


def _buffer_state(model) -> Optional[Dict[str, object]]:
    if not hasattr(model, "buffer"):
        return None
    return {
        "num_seen_examples": int(model.buffer.num_seen_examples),
        "buffer_data": getattr(model.buffer, "buffer_data", None),
    }


def _restore_buffer(model, state: Optional[Mapping[str, object]]) -> None:
    if state is None:
        return
    model.buffer.num_seen_examples = int(state["num_seen_examples"])
    if state.get("buffer_data") is not None:
        # Buffer.add_data_dict stores volumes on buffer.device but keeps
        # keypoints on the model device; restore that same mixed-device layout.
        memory_device = getattr(model.buffer, "device", "cpu")
        model_device = next(model.parameters()).device
        model.buffer.buffer_data = [
            {key: value.to(model_device if key == "keypoints" else memory_device)
             if torch.is_tensor(value) else value for key, value in sample.items()}
            for sample in state["buffer_data"]
        ]


def _atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        partial.unlink()
    torch.save(dict(payload), partial)
    os.replace(partial, path)


def save_source_checkpoint(path: Path, model, config: RunConfig, cfg, streams: Mapping[str, StatefulBatchStream],
                           task_index: int, task_step: int, global_step: int,
                           next_task_index: int, next_task_step: int,
                           matrix: Sequence[Sequence[float]], evaluations: Sequence[Mapping[str, object]]) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": _optimizer(model, config.method).state_dict(),
        "scheduler": model.sch.state_dict() if getattr(model, "sch", None) is not None else None,
        "task_index": int(task_index), "task_step": int(task_step), "global_step": int(global_step),
        "next_task_index": int(next_task_index), "next_task_step": int(next_task_step),
        "replay_buffer": _buffer_state(model),
        "num_seen_examples": int(model.buffer.num_seen_examples) if hasattr(model, "buffer") else 0,
        "rng_states": capture_rng_state(),
        "stream_states": {task: stream.state_dict() for task, stream in streams.items()},
        "resolved_config": {"runner": asdict(config), "source": _resolved_cfg(cfg)},
        "matrix": [list(row) for row in matrix],
        "evaluations": list(evaluations),
    }
    _atomic_torch_save(payload, path)


def load_source_checkpoint(path: Path, model, config: RunConfig, streams: Mapping[str, StatefulBatchStream],
                           device: torch.device) -> Mapping[str, object]:
    # RNG ByteTensors and the CPU replay buffer must stay on CPU. Model and
    # optimizer load_state_dict transfer parameter states to their own device.
    payload = torch.load(str(path), map_location="cpu", mmap=True)
    runner = payload["resolved_config"]["runner"]
    for key in ("method", "budget", "data_root", "seed", "buffer_size"):
        if str(runner[key]) != str(getattr(config, key)):
            raise ValueError(f"Resume configuration mismatch for {key}: {runner[key]} != {getattr(config, key)}")
    if hasattr(model, 'cfg') and model.cfg.dataset.get('keypoint_grid') == 'native_v4':
        previous = payload['resolved_config'].get('source', {})
        old_grid = previous.get('dataset', {}).get('keypoint_grid')
        payload['_protocol_migration'] = old_grid != 'native_v4'
        if payload['_protocol_migration']:
            if (config.method not in ('mer', 'samcl') or payload['global_step'] != 20000
                    or payload['next_task_index'] != 2 or payload['next_task_step'] != 0
                    or payload['stream_states']['nlst']['position'] != 0):
                raise ValueError('Legacy geometry may only migrate from the pre-NLST 20k checkpoint')
            for sample in payload['replay_buffer']['buffer_data']:
                if 'NLST' in str(sample.get('names', '')):
                    raise ValueError('Pre-NLST replay buffer unexpectedly contains NLST')
        elif previous['model']['tre']['coordinate_scale'] != model.cfg.model.tre.coordinate_scale:
            raise ValueError('Landmark loss units changed across a corrected resume')
    model.load_state_dict(payload["model"])
    _optimizer(model, config.method).load_state_dict(payload["optimizer"])
    if config.method == "samcl" and hasattr(model.opt_sam, "param_groups"):
        model.opt_sam.param_groups = model.opt_sam.base_optimizer.param_groups
    if getattr(model, "sch", None) is not None and payload.get("scheduler") is not None:
        model.sch.load_state_dict(payload["scheduler"])
    _restore_buffer(model, payload.get("replay_buffer"))
    for task, state in payload["stream_states"].items():
        streams[task].load_state_dict(state)
    restore_rng_state(payload["rng_states"])
    return payload


def _batch(dataset: Continual3D, indices: Sequence[int], device: torch.device,
           pool: ThreadPoolExecutor) -> Dict[str, object]:
    samples = list(pool.map(dataset.__getitem__, indices))
    batch = dataset.get_batch(samples)
    return dataset.to_device(batch, device)


def _next_batch(stream: StatefulBatchStream, device: torch.device,
                pool: ThreadPoolExecutor) -> Dict[str, object]:
    return _batch(stream.dataset, stream.next_indices(), device, pool)


def _numeric(metrics: Mapping[str, object]) -> Dict[str, float]:
    result = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            value = value.detach().item()
        result[key] = float(value)
    return result


def evaluate_task(model, dataset: Continual3D, task: str, device: torch.device,
                  pool: ThreadPoolExecutor, batch_size: int = 4) -> Dict[str, object]:
    model.eval()
    model.before_epoch("val")
    sums: Dict[str, float] = {}
    total = 0
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            indices = list(range(start, min(start + batch_size, len(dataset))))
            batch = _batch(dataset, indices, device, pool)
            output = model(batch)
            metrics = _numeric(model.get_metrics(batch, output, mode="val"))
            n_samples = len(indices)
            total += n_samples
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + value * n_samples
            del batch, output
    means = {key: value / total for key, value in sums.items()}
    if task == "nlst":
        score = means["metric_final"]
        metric_name = "TRE_mm"
        metric_value = means["tre_mean_mm"]
    else:
        score = means["metric_final"]
        metric_name = "Dice"
        metric_value = means["dice_mean"]
    return {
        "task": task, "pairs": total, "score_higher_is_better": score,
        "metric": metric_name, "value": metric_value,
        "elapsed_seconds": time.time() - started,
        **{f"mean_{key}": value for key, value in means.items()},
    }


def evaluate_all(model, test_sets: Mapping[str, Continual3D], device: torch.device,
                 pool: ThreadPoolExecutor, after_task: str) -> Tuple[List[float], List[Dict[str, object]]]:
    records = []
    row = []
    for task in TASK_ORDER:
        record = evaluate_task(model, test_sets[task], task, device, pool)
        record["after_task"] = after_task
        records.append(record)
        row.append(float(record["score_higher_is_better"]))
    return row, records


def _write_matrix(path: Path, rows: Sequence[Sequence[float]], labels: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["after_task"] + list(TASK_ORDER))
        for label, row in zip(labels, rows):
            writer.writerow([label] + [f"{value:.8f}" for value in row])


def _write_evaluations(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    fields = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _prepare_run_dir(run_dir: Path, config: RunConfig, cfg) -> None:
    allowed = {"stdout.log", "stderr.log", "run.pid", "run.lock", "command.sh"}
    if run_dir.exists():
        unexpected = {path.name for path in run_dir.iterdir()} - allowed
        if unexpected:
            raise FileExistsError(f"Run directory is not empty: {run_dir} ({sorted(unexpected)})")
    else:
        run_dir.mkdir(parents=True)
    for name in ("checkpoints", "samplewise", "results"):
        (run_dir / name).mkdir(exist_ok=False)
    resolved = {"runner": asdict(config), "source": _resolved_cfg(cfg)}
    OmegaConf.save(OmegaConf.create(resolved), run_dir / "config_resolved.yaml")


def _update_latest_link(checkpoints: Path, target: Path) -> None:
    latest = checkpoints / "latest.pt"
    temporary = checkpoints / ".latest.pt.tmp"
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(target.name)
    os.replace(temporary, latest)


def run_source_training(config: RunConfig, run_dir: Path, device: torch.device,
                        resume: Optional[Path] = None,
                        stop_after_global_step: Optional[int] = None) -> Dict[str, object]:
    if config.synthetic:
        raise ValueError("source runtime is for real data only")
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    torch.cuda.set_device(device); torch.cuda.manual_seed_all(config.seed)
    torch.cuda.reset_peak_memory_stats(device)
    cfg = build_source_cfg(config, device, run_dir)
    _prepare_run_dir(run_dir, config, cfg)
    model = _model_for(config.method, cfg).to(device)

    train_sets = {task: Continual3D(cfg, task=task, mode="train") for task in TASK_ORDER}
    test_sets = {task: Continual3D(cfg, task=task, mode="test") for task in TASK_ORDER}
    validation_set = Continual3D(cfg, task="nlst", mode="val")
    streams = {task: StatefulBatchStream(train_sets[task], 4, config.seed + index * 1009)
               for index, task in enumerate(TASK_ORDER)}
    cfg.var.obj_operator.train_sets = list(train_sets.values())
    cfg.var.obj_operator.test_sets = list(test_sets.values())

    global_step = 0
    next_task_index = 0
    next_task_step = 0
    matrix: List[List[float]] = []
    evaluations: List[Dict[str, object]] = []
    resume_evidence = None
    if resume is not None:
        payload = load_source_checkpoint(resume, model, config, streams, device)
        global_step = int(payload["global_step"])
        next_task_index = int(payload.get("next_task_index", payload["task_index"]))
        next_task_step = int(payload.get("next_task_step", payload["task_step"]))
        matrix = [list(row) for row in payload.get("matrix", [])]
        evaluations = list(payload.get("evaluations", []))
        resume_evidence = inherit_resume_evidence(resume, run_dir, payload, protocol_migration=payload.get("_protocol_migration", False))

    steps, per_task = budget_steps(config.method, config.budget, synthetic=False)
    total_steps = steps * len(TASK_ORDER) if per_task else steps
    segment_end = total_steps if stop_after_global_step is None else stop_after_global_step
    if not global_step < segment_end <= total_steps:
        raise ValueError(f"Segment end must be in ({global_step}, {total_steps}], got {segment_end}")
    (run_dir / "execution_segment.json").write_text(json.dumps({
        "start_global_step": global_step, "stop_after_global_step": segment_end,
        "full_budget_global_steps": total_steps, "geometry_protocol": "native_v4",
    }, indent=2) + "\n")
    checkpoint_interval = 1000
    checkpoints = run_dir / "checkpoints"
    log_path = run_dir / "train.jsonl"
    started = time.time()

    def emit(record: Mapping[str, object]) -> None:
        with log_path.open("a") as handle:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")

    with ThreadPoolExecutor(max_workers=4) as pool:
        if resume is not None and payload.get('_protocol_migration', False):
            # Recompute inherited evaluations so the matrix never mixes geometry protocols.
            matrix, evaluations = [], []
            for past_task in TASK_ORDER[:next_task_index]:
                past_path = checkpoints / f'after_{past_task}.pt'
                past = torch.load(str(past_path), map_location='cpu', mmap=True)
                model.load_state_dict(past['model'])
                row, records = evaluate_all(model, test_sets, device, pool, past_task)
                matrix.append(row); evaluations.extend(records)
                del past
            model.load_state_dict(payload['model'])
            restore_rng_state(payload['rng_states'])
            _write_evaluations(run_dir / 'results/inherited_evaluation_native_v4.csv', evaluations)

        validation_start_step = global_step

        def validate_nlst():
            rng = capture_rng_state()
            record = evaluate_task(model, validation_set, 'nlst', device, pool)
            record.update(global_step=global_step, split='valid', geometry_protocol='native_v4')
            with (run_dir / 'validation.jsonl').open('a') as stream:
                stream.write(json.dumps(record, sort_keys=True) + '\n')
            print('VALIDATION ' + json.dumps(record, sort_keys=True), flush=True)
            restore_rng_state(rng)
            model.train(); model.before_epoch('train')

        validate_nlst()
        if config.method == "multitask":
            model.before_epoch("train")
            for zero_based in range(global_step, segment_end):
                task = TASK_ORDER[zero_based % len(TASK_ORDER)]
                step_started = time.time()
                batch = _next_batch(streams[task], device, pool)
                model.train()
                model.observe(batch)
                metrics = _numeric(model.metrics)
                global_step = zero_based + 1
                if not np.isfinite(metrics["loss_final"]):
                    raise FloatingPointError(f"Non-finite loss at step {global_step}: {metrics}")
                record = {"method": config.method, "budget": config.budget, "task": task,
                          "task_step": (global_step - 1) // 4 + 1, "global_step": global_step,
                          "step_seconds": time.time() - step_started, **metrics}
                emit(record)
                if global_step == 1 or global_step % 25 == 0:
                    print(json.dumps(record, sort_keys=True), flush=True)
                if global_step == validation_start_step + 50 or global_step % 1000 == 0:
                    validate_nlst()
                if global_step % checkpoint_interval == 0 and global_step < segment_end:
                    state = checkpoints / "state_latest.pt"
                    save_source_checkpoint(state, model, config, cfg, streams, 3, global_step, global_step,
                                           0, global_step, matrix, evaluations)
                    _update_latest_link(checkpoints, state)
                del batch
            pre_eval_state = checkpoints / "state_latest.pt"
            save_source_checkpoint(pre_eval_state, model, config, cfg, streams, 3, global_step, global_step,
                                   0, global_step, matrix, evaluations)
            _update_latest_link(checkpoints, pre_eval_state)
            labels = []
            if global_step == total_steps:
                row, records = evaluate_all(model, test_sets, device, pool, "multitask_final")
                matrix.append(row); evaluations.extend(records)
                state = checkpoints / "multitask_final.pt"
                save_source_checkpoint(state, model, config, cfg, streams, 3, steps, global_step,
                                       4, 0, matrix, evaluations)
                _update_latest_link(checkpoints, state)
                pre_eval_state.unlink()
                labels = ["multitask_final"]
        else:
            labels = list(TASK_ORDER[:len(matrix)])
            for task_index in range(next_task_index, len(TASK_ORDER)):
                task = TASK_ORDER[task_index]
                cfg.var.obj_operator.task_idx = task_index
                start_step = next_task_step if task_index == next_task_index else 0
                model.before_epoch("train")
                task_step = start_step
                task_end = min(steps, start_step + segment_end - global_step)
                for zero_based in range(start_step, task_end):
                    step_started = time.time()
                    batch = _next_batch(streams[task], device, pool)
                    model.train()
                    model.observe(batch)
                    metrics = _numeric(model.metrics)
                    task_step = zero_based + 1
                    global_step += 1
                    if not np.isfinite(metrics["loss_final"]):
                        raise FloatingPointError(f"Non-finite loss at {task}/{task_step}: {metrics}")
                    record = {"method": config.method, "budget": config.budget, "task": task,
                              "task_index": task_index, "task_step": task_step,
                              "global_step": global_step, "step_seconds": time.time() - step_started,
                              "buffer_seen": int(model.buffer.num_seen_examples), **metrics}
                    emit(record)
                    if task_step == 1 or task_step % 25 == 0:
                        print(json.dumps(record, sort_keys=True), flush=True)
                    if global_step == validation_start_step + 50 or task_step % 1000 == 0:
                        validate_nlst()
                    if task_step % checkpoint_interval == 0 and task_step < steps and global_step < segment_end:
                        state = checkpoints / "state_latest.pt"
                        save_source_checkpoint(state, model, config, cfg, streams, task_index, task_step,
                                               global_step, task_index, task_step, matrix, evaluations)
                        _update_latest_link(checkpoints, state)
                    del batch
                pre_eval_state = checkpoints / "state_latest.pt"
                save_source_checkpoint(pre_eval_state, model, config, cfg, streams, task_index, task_step,
                                       global_step, task_index, task_step, matrix, evaluations)
                _update_latest_link(checkpoints, pre_eval_state)
                if task_step < steps:
                    break
                row, records = evaluate_all(model, test_sets, device, pool, task)
                matrix.append(row); evaluations.extend(records); labels.append(task)
                state = checkpoints / f"after_{task}.pt"
                save_source_checkpoint(state, model, config, cfg, streams, task_index, steps, global_step,
                                       task_index + 1, 0, matrix, evaluations)
                _update_latest_link(checkpoints, state)
                pre_eval_state.unlink()
                next_task_step = 0
                if global_step == segment_end:
                    break

    elapsed = time.time() - started
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    _write_matrix(run_dir / "results" / f"R_{config.method}_{config.budget}.csv", matrix, labels)
    _write_evaluations(run_dir / "results" / "evaluation.csv", evaluations)
    status = "complete" if global_step == total_steps else "segment_complete"
    summary = {
        "method": config.method, "budget": config.budget, "synthetic": False,
        "model": "pinned Continual-Reg LKUNet", "input_size": [112, 96, 112],
        "batch_size": 4, "global_steps": global_step, "elapsed_seconds": elapsed,
        "peak_vram_mb": peak_vram_mb, "matrix_rows": len(matrix), "status": status,
        "full_budget_global_steps": total_steps, "stop_after_global_step": segment_end,
        "geometry_protocol": "native_v4",
        "latest_checkpoint": str((checkpoints / "latest.pt").resolve()),
    }
    if resume_evidence is not None:
        summary.update({
            "resumed_session_seconds": elapsed,
            "elapsed_seconds": resume_evidence["inherited_accounted_seconds"] + elapsed,
            "elapsed_seconds_basis": resume_evidence["runtime_basis"],
            "elapsed_seconds_is_lower_bound": True,
            "resume_global_step": resume_evidence["checkpoint_global_step"],
            "resume_provenance": str(run_dir / "resume_provenance.json"),
            "peak_vram_basis": "resumed_session_only",
            "rollback_steps": resume_evidence["rollback_steps"],
            "rollback_training_seconds": resume_evidence["cumulative_rollback_training_seconds"],
        })
    (run_dir / "results" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_dir / (".complete" if status == "complete" else ".segment_complete")).write_text(status + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def evaluate_source_payload(payload: Mapping[str, object], data_root: Path, output: Path,
                            device: torch.device) -> List[Dict[str, object]]:
    runner = dict(payload["resolved_config"]["runner"])
    runner["data_root"] = str(data_root.resolve())
    runner["output_root"] = str(output.parent.resolve())
    if isinstance(runner.get("input_size"), list):
        runner["input_size"] = tuple(runner["input_size"])
    config = RunConfig(**runner)
    if config.synthetic:
        raise ValueError("Expected a real-data source checkpoint")
    cfg = build_source_cfg(config, device, output.parent)
    model = _model_for(config.method, cfg).to(device)
    model.load_state_dict(payload["model"])
    test_sets = {task: Continual3D(cfg, task=task, mode="test") for task in TASK_ORDER}
    cfg.var.obj_operator.test_sets = list(test_sets.values())
    with ThreadPoolExecutor(max_workers=4) as pool:
        _, records = evaluate_all(model, test_sets, device, pool, "standalone_latest")
    _write_evaluations(output, records)
    return records
