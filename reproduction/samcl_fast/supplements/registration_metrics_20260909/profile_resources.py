"""Short resource measurements; never write back any trained checkpoint."""
import argparse
import gc
import json
import random
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from core.datasets.continual3d import Continual3D
from core.methods.sgd import Sgd
from lib.samcl_runtime import RunConfig, TASK_ORDER
from lib.source_runtime import (StatefulBatchStream, build_source_cfg, _model_for, _optimizer,
                                _restore_buffer, _batch, _next_batch, _numeric)


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--methods', nargs='+', choices=['plain_update', 'mer', 'samcl'], default=['plain_update', 'mer', 'samcl'])
    p.add_argument('--tasks', nargs='+', choices=TASK_ORDER, default=list(TASK_ORDER))
    args = p.parse_args()
    if args.warmup < 1 or args.repeats < 2 or args.output.exists():
        raise ValueError('Require warmup >= 1, repeats >= 2, and a new output file.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = args.experiment_root
    corrected = root / 'corrections/nlst_native_v4_20260831/core_runs'
    stage_runs = {
        'mer': root / 'core_runs/mer/full/j3_01_s42/checkpoints',
        'samcl': root / 'core_runs/samcl/full/j3_03_s42/checkpoints',
    }
    nlst_runs = {
        'mer': corrected / 'mer/full/parallel_segment_04_to_30000_seed42/checkpoints',
        'samcl': corrected / 'samcl/full/parallel_segment_03_to_30000_seed42/checkpoints',
    }
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    report = dict(status='running', gpu=torch.cuda.get_device_name(device), torch_version=torch.__version__,
                  geometry_protocol='native_v4', warmup=args.warmup, repeats=args.repeats,
                  training_batch_size=4, inference_batch_size=1, seed=42, records=[],
                  plain_update_scope='microbenchmark initialized from MER stage weights; no sequential performance claim',
                  training_scope='short real-data updates including batch loading, transfer, loss, backward, optimizer and replay',
                  inference_scope='device-resident image/mask pair, model forward including deformation/warping; excludes I/O, label scoring and checkpoint loading')
    for method in args.methods:
        source_method = 'mer' if method == 'plain_update' else method
        for task in args.tasks:
            index = TASK_ORDER.index(task)
            random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
            config = RunConfig(method='multitask' if method == 'plain_update' else method, budget='full',
                               data_root=str(args.data_root.resolve()), output_root=str(args.output.parent),
                               input_size=(112, 96, 112))
            cfg = build_source_cfg(config, device, args.output.parent)
            cfg.model.task_sequential = True; cfg.var.obj_operator.task_idx = index
            model = Sgd(cfg).to(device) if method == 'plain_update' else _model_for(method, cfg).to(device)
            checkpoint_label = 'fresh_seed_42'
            if index:
                folder = nlst_runs[source_method] if index == 3 else stage_runs[source_method]
                checkpoint = folder / f'after_{TASK_ORDER[index-1]}.pt'
                payload = torch.load(str(checkpoint), map_location='cpu', mmap=True)
                expected_step = index * 10000
                if payload['global_step'] != expected_step:
                    raise ValueError(f'Wrong checkpoint stage: {checkpoint}')
                model.load_state_dict(payload['model'])
                _optimizer(model, config.method).load_state_dict(payload['optimizer'])
                if method != 'plain_update':
                    _restore_buffer(model, payload['replay_buffer'])
                    if method == 'samcl':
                        model.opt_sam.param_groups = model.opt_sam.base_optimizer.param_groups
                checkpoint_label = f'{source_method}/after_{TASK_ORDER[index-1]}/step_{expected_step}'
                del payload
            train = Continual3D(cfg, task=task, mode='train')
            test = Continual3D(cfg, task=task, mode='test')
            stream = StatefulBatchStream(train, 4, 42 + index*1009)
            memory = getattr(getattr(model, 'buffer', None), 'buffer_data', [])
            counts = Counter(sample['names'].split('_')[0] for sample in memory if sample)
            record = dict(method=method, task=task, checkpoint=checkpoint_label, train_pairs=len(train), test_pairs=len(test),
                          parameter_count=sum(p.numel() for p in model.parameters()),
                          parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                          replay_capacity=200 if method != 'plain_update' else 0,
                          replay_occupied=len([s for s in memory if s]), replay_tensor_bytes=tensor_bytes(memory),
                          replay_source_counts=dict(counts))
            with ThreadPoolExecutor(max_workers=4) as pool:
                model.train(); model.before_epoch('train')
                durations = []
                for n in range(args.warmup + args.repeats):
                    if n == args.warmup:
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize(); begin = time.perf_counter()
                    batch = _next_batch(stream, device, pool)
                    model.observe(batch); metrics = _numeric(model.metrics)
                    if not all(np.isfinite(v) for v in metrics.values()):
                        raise FloatingPointError(f'Nonfinite profile update: {method}/{task}')
                    torch.cuda.synchronize()
                    if n >= args.warmup:
                        durations.append(time.perf_counter()-begin)
                    del batch
                record.update(train_step_seconds=durations, train_step_median_s=statistics.median(durations),
                              train_step_mean_s=statistics.mean(durations),
                              train_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                              train_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                model.eval(); model.before_epoch('val')
                durations = []
                with torch.no_grad():
                    for n in range(args.warmup + args.repeats):
                        batch = _batch(test, [n % len(test)], device, pool)
                        inference_input = {k: batch[k] for k in ('imgs', 'masks', 'names')}
                        torch.cuda.synchronize(); begin = time.perf_counter()
                        output = model(inference_input)
                        torch.cuda.synchronize()
                        if n >= args.warmup:
                            durations.append(time.perf_counter()-begin)
                        del output, batch, inference_input
                record.update(inference_pair_seconds=durations, inference_pair_median_ms=1000*statistics.median(durations))
            report['records'].append(record)
            tmp = args.output.with_suffix('.partial'); tmp.write_text(json.dumps(report, indent=2)+'\n'); tmp.replace(args.output)
            print(json.dumps(record), flush=True)
            del memory, model, train, test, stream, cfg
            gc.collect(); torch.cuda.empty_cache()
    for method in args.methods:
        counts = [r['parameter_count'] for r in report['records'] if r['method'] == method]
        if len(set(counts)) != 1:
            raise ValueError('Unexpected parameter growth in a fixed-network method.')
    report['status'] = 'complete'
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
