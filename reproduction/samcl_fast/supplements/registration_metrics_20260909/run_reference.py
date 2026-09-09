"""Single-task references for RMA; reuse the existing corrected source runtime."""
import argparse
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from core.datasets.continual3d import Continual3D
from core.methods.sgd import Sgd
from lib.samcl_runtime import RunConfig, TASK_ORDER, capture_rng_state, restore_rng_state
from lib.source_runtime import (StatefulBatchStream, build_source_cfg, _next_batch, _numeric,
                                _resolved_cfg, _atomic_torch_save, evaluate_task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=TASK_ORDER[1:], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or not torch.cuda.is_available():
        raise ValueError('A positive step budget and CUDA are required.')
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = run / 'checkpoints/latest.pt'
    if (run / 'config_resolved.yaml').exists() and not args.resume:
        raise FileExistsError(f'Existing reference: {run}')
    (run / 'results').mkdir(exist_ok=True)
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    # The common reference uses the plain Adam update, with no past task, replay, or SAM.
    config = RunConfig(method='multitask', budget='full', data_root=str(args.data_root.resolve()),
                       output_root=str(run.parent), seed=seed, input_size=(112, 96, 112))
    cfg = build_source_cfg(config, device, run)
    cfg.exp.name = 'single_task_reference'; cfg.model.task_sequential = True
    model = Sgd(cfg).to(device)
    train = Continual3D(cfg, task=args.task, mode='train')
    test = Continual3D(cfg, task=args.task, mode='test')
    stream = StatefulBatchStream(train, 4, seed + TASK_ORDER.index(args.task) * 1009)
    cfg.var.obj_operator.train_sets = [train]; cfg.var.obj_operator.test_sets = [test]
    protocol = dict(task=args.task, steps=args.steps, seed=seed, geometry_protocol='native_v4',
                    reference='independent_plain_Adam', batch_size=4, learning_rate=1e-4,
                    train_pairs=len(train), test_pairs=len(test), initialization='fresh_seed_42',
                    history_access=False, replay=False, input_size=[112, 96, 112],
                    device=torch.cuda.get_device_name(device))
    start, inherited_seconds, peak = 0, 0., 0.
    if args.resume:
        payload = torch.load(str(checkpoint), map_location='cpu', mmap=True)
        if payload['protocol'] != protocol:
            raise ValueError('Reference protocol changed across resume.')
        model.load_state_dict(payload['model']); model.opt.load_state_dict(payload['optimizer'])
        stream.load_state_dict(payload['stream']); restore_rng_state(payload['rng_states'])
        start, inherited_seconds, peak = payload['steps'], payload['active_seconds'], payload['peak_vram_mib']
        del payload
    else:
        OmegaConf.save(OmegaConf.create(dict(protocol=protocol, runner=asdict(config), source=_resolved_cfg(cfg))),
                       run / 'config_resolved.yaml')
    print(json.dumps(dict(status='started', **protocol)), flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    def save(step):
        _atomic_torch_save(dict(model=model.state_dict(), optimizer=model.opt.state_dict(),
                               stream=stream.state_dict(), rng_states=capture_rng_state(), steps=step,
                               protocol=protocol, active_seconds=inherited_seconds+time.perf_counter()-started,
                               peak_vram_mib=max(peak, torch.cuda.max_memory_allocated(device)/2**20)), checkpoint)

    with ThreadPoolExecutor(max_workers=4) as pool, (run / 'train.jsonl').open('a', buffering=1) as log:
        model.train(); model.before_epoch('train')
        for step in range(start + 1, args.steps + 1):
            before = time.perf_counter()
            batch = _next_batch(stream, device, pool)
            model.observe(batch)
            metrics = _numeric(model.metrics)
            if not all(math.isfinite(v) for v in metrics.values()):
                raise FloatingPointError(f'Nonfinite training metric at step {step}: {metrics}')
            if step == start+1 or step % 25 == 0:
                record = dict(step=step, task=args.task, step_seconds=time.perf_counter()-before,
                              elapsed_seconds=inherited_seconds+time.perf_counter()-started, **metrics)
                log.write(json.dumps(record)+'\n'); print(json.dumps(record), flush=True)
            del batch
            if step % 1000 == 0 or step == args.steps:
                save(step)
        training_seconds = inherited_seconds + time.perf_counter()-started
        peak = max(peak, torch.cuda.max_memory_allocated(device)/2**20)
        # Evaluate a reloaded fixed-budget checkpoint; no selection on test performance.
        payload = torch.load(str(checkpoint), map_location='cpu', mmap=True)
        if payload['steps'] != args.steps:
            raise ValueError('Final checkpoint step does not match the budget.')
        model.load_state_dict(payload['model']); del payload
        evaluation = evaluate_task(model, test, args.task, device, pool)
    if not math.isfinite(evaluation['value']) or evaluation['value'] <= 0:
        raise ValueError('Independent reference cannot serve as a positive ratio denominator.')
    summary = dict(status='complete', **protocol, training_seconds=training_seconds,
                   elapsed_basis='active_process_time_including_checkpoint_writes',
                   peak_vram_mib=peak, parameter_count=sum(p.numel() for p in model.parameters()),
                   evaluation=evaluation, checkpoint_reload='passed', checkpoint='checkpoints/latest.pt')
    temporary = run / 'results/summary.json.partial'
    temporary.write_text(json.dumps(summary, indent=2)+'\n')
    temporary.replace(run / 'results/summary.json')
    (run / '.complete').write_text('complete\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
