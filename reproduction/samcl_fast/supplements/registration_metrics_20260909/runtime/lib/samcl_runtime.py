from __future__ import annotations

import copy
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


TASK_ORDER = ("oasis", "ctct", "nlst", "mrct")
TASK_DISPLAY = {
    "oasis": "OASIS",
    "ctct": "AbdomenCTCT",
    "nlst": "NLST",
    "mrct": "AbdomenMRCT",
}
SEGMENTATION_TASKS = frozenset({"oasis", "ctct", "mrct"})


@dataclass
class RunConfig:
    method: str
    budget: str
    data_root: str
    output_root: str
    seed: int = 42
    beta: float = 0.25
    rho: float = 0.05
    adaptive: bool = True
    weight_decay: float = 1.0e-5
    buffer_size: int = 200
    learning_rate: float = 1.0e-4
    input_size: Tuple[int, int, int] = (24, 24, 24)
    synthetic: bool = False


def budget_steps(method: str, budget: str, synthetic: bool = False) -> Tuple[int, bool]:
    if synthetic:
        return ({"multitask": 12, "mer": 3, "samcl": 3}[method], method != "multitask")
    if method == "multitask":
        return ({"smoke": 100, "quick": 4000, "full": 40000}[budget], False)
    return ({"smoke": 100, "quick": 1000, "full": 10000}[budget], True)


def task_loss_names(task: str) -> Tuple[str, ...]:
    if task == "nlst":
        return ("ncc", "tre", "membrane", "bending")
    if task in SEGMENTATION_TASKS:
        return ("ncc", "dice", "membrane", "bending")
    raise ValueError(f"Unknown task: {task}")


def round_robin_tasks(total_steps: int) -> List[str]:
    return [TASK_ORDER[i % len(TASK_ORDER)] for i in range(total_steps)]


def _provider_cfg(data_root: str):
    return OmegaConf.create({
        "dataset": {
            "root": str(data_root), "dim": 3, "size_img": [112, 96, 112],
            "one_sample_only": False, "normalization": "min-max", "intensity_aug": False,
        },
        "model": {"tre": {"label_center": False}},
        "exp": {"mode": "train", "test": {"save_result": {"enable": False, "idx_sample": -1}}},
    })


def load_provider_batch(data_root: str, task: str, mode: str = "train") -> Dict[str, object]:
    from core.datasets.continual3d import Continual3D

    dataset = Continual3D(_provider_cfg(data_root), task=task, mode=mode)
    sample = dataset[0]
    batch = dataset.get_batch([sample])
    batch["task"] = task
    return batch


def _resize_volume(value: torch.Tensor, size: Sequence[int], mode: str) -> torch.Tensor:
    original_dtype = value.dtype
    value = value.float()
    resized = F.interpolate(value, size=size, mode=mode, align_corners=False if mode != "nearest" else None)
    if mode == "nearest" and not original_dtype.is_floating_point:
        resized = resized.round().to(original_dtype)
    return resized


def prepare_smoke_batch(batch: Mapping[str, object], device: torch.device, size=(24, 24, 24)) -> Dict[str, object]:
    result: Dict[str, object] = {"task": batch["task"], "names": list(batch["names"])}
    imgs = batch["imgs"].float()
    result["imgs"] = _resize_volume(imgs, size, "trilinear").to(device)
    if "masks" in batch:
        result["masks"] = _resize_volume(batch["masks"], size, "nearest").to(device)
    if "segs" in batch:
        result["segs"] = _resize_volume(batch["segs"], size, "nearest").long().to(device)
    if "keypoints" in batch:
        original = torch.tensor(batch["imgs"].shape[-3:], dtype=torch.float32)
        target = torch.tensor(size, dtype=torch.float32)
        scale = (target - 1) / (original - 1)
        result["keypoints"] = [kp.float().to(device) * scale.to(device) for kp in batch["keypoints"]]
    return result


class TinyRegistrationNet(nn.Module):
    def __init__(self, channels: int = 8):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(2, channels, 3, padding=1), nn.InstanceNorm3d(channels), nn.LeakyReLU(0.1),
            nn.Conv3d(channels, channels, 3, padding=1), nn.InstanceNorm3d(channels), nn.LeakyReLU(0.1),
            nn.Conv3d(channels, 3, 3, padding=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return 0.10 * torch.tanh(self.body(imgs))


def _base_grid(shape: Sequence[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    z, y, x = [torch.linspace(-1, 1, int(n), device=device, dtype=dtype) for n in shape]
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1)


def warp(volume: torch.Tensor, flow: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    grid = _base_grid(volume.shape[-3:], volume.device, volume.dtype).unsqueeze(0)
    displacement = flow.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
    return F.grid_sample(volume, grid + displacement, mode=mode, padding_mode="border", align_corners=True)


def local_ncc_loss(fixed: torch.Tensor, moved: torch.Tensor) -> torch.Tensor:
    fixed = fixed - fixed.mean(dim=(-3, -2, -1), keepdim=True)
    moved = moved - moved.mean(dim=(-3, -2, -1), keepdim=True)
    numerator = (fixed * moved).mean(dim=(-3, -2, -1))
    denominator = torch.sqrt((fixed.square().mean(dim=(-3, -2, -1)) + 1e-6) *
                             (moved.square().mean(dim=(-3, -2, -1)) + 1e-6))
    return (1.0 - numerator / denominator).mean()


def soft_dice_loss(fixed: torch.Tensor, moved: torch.Tensor) -> torch.Tensor:
    classes = max(2, int(torch.max(torch.stack((fixed.max(), moved.max()))).item()) + 1)
    fixed_oh = F.one_hot(fixed[:, 0].long(), classes).permute(0, 4, 1, 2, 3).float()
    moved_oh = F.one_hot(moved[:, 0].long(), classes).permute(0, 4, 1, 2, 3).float()
    numerator = 2 * (fixed_oh[:, 1:] * moved_oh[:, 1:]).sum(dim=(-3, -2, -1))
    denominator = (fixed_oh[:, 1:] + moved_oh[:, 1:]).sum(dim=(-3, -2, -1)).clamp_min(1e-6)
    return 1.0 - (numerator / denominator).mean()


def registration_losses(model: nn.Module, batch: Mapping[str, object]) -> Dict[str, torch.Tensor]:
    task = str(batch["task"])
    imgs = batch["imgs"]
    flow = model(imgs)
    moved = warp(imgs[:, [1]], flow)
    losses: Dict[str, torch.Tensor] = {"ncc": local_ncc_loss(imgs[:, [0]], moved)}
    dz = flow[:, :, 1:] - flow[:, :, :-1]
    dy = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dx = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
    losses["membrane"] = (dz.square().mean() + dy.square().mean() + dx.square().mean()) / 3
    losses["bending"] = ((dz[:, :, 1:] - dz[:, :, :-1]).square().mean() +
                         (dy[:, :, :, 1:] - dy[:, :, :, :-1]).square().mean() +
                         (dx[:, :, :, :, 1:] - dx[:, :, :, :, :-1]).square().mean()) / 3
    if task in SEGMENTATION_TASKS:
        fixed_seg = batch["segs"][:, [0]]
        moved_seg = warp(batch["segs"][:, [1]].float(), flow, mode="nearest").round().long()
        losses["dice"] = soft_dice_loss(fixed_seg, moved_seg)
    elif task == "nlst":
        keypoints = batch["keypoints"]
        mean_flow = flow.mean(dim=(-3, -2, -1))
        spatial = torch.tensor(flow.shape[-3:], device=flow.device, dtype=flow.dtype)
        voxel_shift = mean_flow * ((spatial - 1) / 2)
        per_sample = []
        for index, points in enumerate(keypoints):
            predicted = points[0] + voxel_shift[index]
            per_sample.append(torch.linalg.vector_norm(predicted - points[1], dim=-1).mean() * 3.0)
        losses["tre"] = torch.stack(per_sample).mean()
    else:
        raise ValueError(task)
    expected = set(task_loss_names(task))
    if set(losses) != expected:
        raise RuntimeError(f"Loss routing mismatch for {task}: {sorted(losses)}")
    return losses


def combined_loss(losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
    weights = {"ncc": 1.0, "dice": 1.0, "tre": 0.02, "membrane": 1.0, "bending": 1.0}
    return sum(weights[name] * value for name, value in losses.items())


class ReservoirBuffer:
    def __init__(self, capacity: int, seed: int = 42):
        self.capacity = int(capacity)
        self.num_seen_examples = 0
        self.items: List[Dict[str, object]] = []
        self.rng = np.random.default_rng(seed)

    def reservoir_index(self) -> int:
        seen = self.num_seen_examples
        if seen < self.capacity:
            return seen
        index = int(self.rng.integers(0, seen + 1))
        return index if index < self.capacity else -1

    def add(self, batch: Mapping[str, object]) -> bool:
        index = self.reservoir_index()
        self.num_seen_examples += 1
        if index < 0:
            return False
        item = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                item[key] = value.detach().cpu().clone()
            elif key == "keypoints":
                item[key] = [point.detach().cpu().clone() for point in value]
            else:
                item[key] = copy.deepcopy(value)
        if index == len(self.items):
            self.items.append(item)
        else:
            self.items[index] = item
        return True

    def sample(self, device: torch.device) -> Dict[str, object]:
        if not self.items:
            raise RuntimeError("Cannot sample an empty buffer")
        item = copy.deepcopy(self.items[int(self.rng.integers(0, len(self.items)))])
        for key, value in list(item.items()):
            if torch.is_tensor(value):
                item[key] = value.to(device)
            elif key == "keypoints":
                item[key] = [point.to(device) for point in value]
        return item

    def state_dict(self) -> Dict[str, object]:
        return {"capacity": self.capacity, "num_seen_examples": self.num_seen_examples,
                "items": self.items, "rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.capacity = int(state["capacity"])
        self.num_seen_examples = int(state["num_seen_examples"])
        self.items = list(state["items"])
        self.rng.bit_generator.state = state["rng_state"]


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        if rho < 0:
            raise ValueError("rho must be non-negative")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        grads = [((p.abs() if group["adaptive"] else 1.0) * p.grad).norm(2)
                 for group in self.param_groups for p in group["params"] if p.grad is not None]
        if not grads:
            raise RuntimeError("SAM first_step called without gradients")
        norm = torch.norm(torch.stack(grads), 2)
        for group in self.param_groups:
            scale = group["rho"] / (norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.detach().clone()
                p.add_((p.square() if group["adaptive"] else 1.0) * p.grad * scale.to(p))
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.copy_(self.state[p]["old_p"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()


def capture_rng_state() -> Dict[str, object]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, buffer: Optional[ReservoirBuffer],
                    task_index: int, task_step: int, global_step: int, config: Mapping[str, object]) -> None:
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "task_index": task_index, "task_step": task_step, "global_step": global_step,
        "replay_buffer": buffer.state_dict() if buffer is not None else None,
        "num_seen_examples": buffer.num_seen_examples if buffer is not None else 0,
        "rng_states": capture_rng_state(), "resolved_config": dict(config),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer, scheduler=None,
                    buffer: Optional[ReservoirBuffer] = None, map_location="cpu") -> Dict[str, object]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload["scheduler"] is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if buffer is not None and payload["replay_buffer"] is not None:
        buffer.load_state_dict(payload["replay_buffer"])
    restore_rng_state(payload["rng_states"])
    return payload


def evaluate_task(model: nn.Module, batch: Mapping[str, object]) -> float:
    model.eval()
    with torch.no_grad():
        losses = registration_losses(model, batch)
        if batch["task"] == "nlst":
            return -float(losses["tre"].item())
        return float(1.0 - losses["dice"].item())


def evaluate_all_tasks(model: nn.Module, batches: Mapping[str, Mapping[str, object]]) -> Dict[str, float]:
    return {task: evaluate_task(model, batches[task]) for task in TASK_ORDER}


def write_matrix(path: Path, rows: Sequence[Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["after_task"] + list(TASK_ORDER))
        for index, row in enumerate(rows):
            writer.writerow([TASK_ORDER[index]] + [f"{value:.8f}" for value in row])


def _move_toward(model: nn.Module, initial: Sequence[torch.Tensor], beta: float) -> None:
    with torch.no_grad():
        for parameter, start in zip(model.parameters(), initial):
            parameter.copy_(start + beta * (parameter - start))


def _standard_step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    losses = registration_losses(model, batch)
    total = combined_loss(losses)
    total.backward()
    optimizer.step()
    return losses, total


def _sam_step(model, optimizer: SAM, batch):
    optimizer.zero_grad(set_to_none=True)
    losses = registration_losses(model, batch)
    total = combined_loss(losses)
    total.backward()
    optimizer.first_step(zero_grad=True)
    second_losses = registration_losses(model, batch)
    second_total = combined_loss(second_losses)
    second_total.backward()
    optimizer.second_step(zero_grad=True)
    return losses, total


def run_training(config: RunConfig, run_dir: Path, device: torch.device, resume: Optional[Path] = None) -> Dict[str, object]:
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.empty(1, device=device)
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(device)
    if run_dir.exists():
        allowed = {"stdout.log", "stderr.log", "run.pid", "run.lock", "command.sh"}
        unexpected = {path.name for path in run_dir.iterdir()} - allowed
        if unexpected:
            raise FileExistsError(f"Run directory is not empty: {run_dir} ({sorted(unexpected)})")
    else:
        run_dir.mkdir(parents=True)
    for subdir in ("checkpoints", "samplewise", "results"):
        (run_dir / subdir).mkdir()
    OmegaConf.save(OmegaConf.create(asdict(config)), run_dir / "config_resolved.yaml")
    provider_batches = {task: load_provider_batch(config.data_root, task, "train") for task in TASK_ORDER}
    batches = {task: prepare_smoke_batch(batch, device, config.input_size)
               for task, batch in provider_batches.items()}
    model = TinyRegistrationNet().to(device)
    buffer = ReservoirBuffer(config.buffer_size, config.seed) if config.method in {"mer", "samcl"} else None
    if config.method == "samcl":
        optimizer = SAM(model.parameters(), torch.optim.Adam, lr=config.learning_rate,
                        rho=config.rho, adaptive=config.adaptive, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                     weight_decay=config.weight_decay if config.method == "mer" else 0.0)
    scheduler = None
    global_step = 0
    start_task = 0
    if resume:
        payload = load_checkpoint(resume, model, optimizer, scheduler, buffer, device)
        global_step = int(payload["global_step"])
        start_task = int(payload["task_index"])
    steps, per_task = budget_steps(config.method, config.budget, config.synthetic)
    matrix: List[List[float]] = []
    log_path = run_dir / "train.jsonl"
    started = time.time()

    def emit(record):
        with log_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    if config.method == "multitask":
        for task_step, task in enumerate(round_robin_tasks(steps), 1):
            model.train(); losses, total = _standard_step(model, optimizer, batches[task]); global_step += 1
            emit({"method": config.method, "task": task, "task_step": task_step,
                  "global_step": global_step, "loss": float(total.item()),
                  **{f"loss_{k}": float(v.item()) for k, v in losses.items()}})
        scores = evaluate_all_tasks(model, batches)
        matrix.append([scores[task] for task in TASK_ORDER])
        save_checkpoint(run_dir / "checkpoints/latest.pt", model, optimizer, scheduler, buffer,
                        len(TASK_ORDER) - 1, steps, global_step, asdict(config))
    else:
        for task_index, task in enumerate(TASK_ORDER[start_task:], start_task):
            for task_step in range(1, steps + 1):
                model.train(); initial = [p.detach().clone() for p in model.parameters()]
                if config.method == "samcl":
                    losses, total = _sam_step(model, optimizer, batches[task])
                else:
                    losses, total = _standard_step(model, optimizer, batches[task])
                buffer.add(batches[task])
                if len(buffer.items) > 0:
                    replay = buffer.sample(device)
                    if config.method == "samcl":
                        _sam_step(model, optimizer, replay)
                    else:
                        _standard_step(model, optimizer, replay)
                _move_toward(model, initial, config.beta)
                global_step += 1
                emit({"method": config.method, "task": task, "task_index": task_index,
                      "task_step": task_step, "global_step": global_step,
                      "loss": float(total.item()), "buffer_seen": buffer.num_seen_examples,
                      **{f"loss_{k}": float(v.item()) for k, v in losses.items()}})
            scores = evaluate_all_tasks(model, batches)
            matrix.append([scores[name] for name in TASK_ORDER])
            save_checkpoint(run_dir / f"checkpoints/after_{task}.pt", model, optimizer, scheduler, buffer,
                            task_index, steps, global_step, asdict(config))
        save_checkpoint(run_dir / "checkpoints/latest.pt", model, optimizer, scheduler, buffer,
                        len(TASK_ORDER) - 1, steps, global_step, asdict(config))

    elapsed = time.time() - started
    peak_vram_mb = (torch.cuda.max_memory_allocated(device) / 1024 ** 2) if device.type == "cuda" else 0.0
    write_matrix(run_dir / "results" / f"R_{config.method}_{config.budget}.csv", matrix)
    summary = {"method": config.method, "budget": config.budget, "synthetic": config.synthetic,
               "global_steps": global_step, "elapsed_seconds": elapsed, "peak_vram_mb": peak_vram_mb,
               "matrix_rows": len(matrix), "status": "complete"}
    (run_dir / "results" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_dir / ".complete").write_text("complete\n")
    return summary
