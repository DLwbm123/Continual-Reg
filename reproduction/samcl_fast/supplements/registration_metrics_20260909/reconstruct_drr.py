"""Recover unique historical pair use through the original CPU sampling path.

No images, model scores, or gradient computations are used. Results are released
only when every stage's ordered buffer, NumPy RNG and batch streams match the
existing checkpoint. Patient/pair names remain in memory and are never exported.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from core.datasets.continual3d import Continual3D
from core.methods.buffer_mer import Buffer
from core.methods.mer import MER
from core.methods.mersam import MERSAM
from lib.samcl_runtime import RunConfig, TASK_ORDER
from lib.source_runtime import StatefulBatchStream, build_source_cfg


def noop(*args, **kwargs):
    pass


class SamplingOnly:
    """Execute unmodified observe/draw/buffer code with inert numeric operations."""

    def __init__(self, method, cfg, origin):
        self.cfg, self.origin = cfg, origin
        self.buffer = Buffer(cfg)
        self.net = SimpleNamespace(get_params=lambda: torch.zeros(1), set_params=noop)
        self.opt = self.opt_sam = SimpleNamespace(
            zero_grad=noop, step=noop, first_step=noop, second_step=noop)
        cls = {'mer': MER, 'samcl': MERSAM}[method]
        self.draw_batches = MethodType(cls.draw_batches, self)
        self.observe = MethodType(cls.observe, self)
        self.historical = {task: set() for task in TASK_ORDER}
        self.stage_used = {task: set() for task in TASK_ORDER}
        self.task_index = 0

    def forward(self, inputs):
        # Count actual consumption, not get_data_dict: MERSAM discards its first
        # draw. Repeated SAM forwards and repeated visits count once per pair.
        for name in inputs['names']:
            task = self.origin[name]
            index = TASK_ORDER.index(task)
            if index > self.task_index:
                raise ValueError('Future-task data access in sampling replay')
            if index < self.task_index:
                self.historical[task].add(name)
                self.stage_used[task].add(name)

    def get_metrics(self, *args, **kwargs):
        return {'loss_final': SimpleNamespace(backward=noop)}


def same_state(left, right):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_state(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(same_state(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    return left == right


def verify_stage(payload, model, streams, global_step):
    checks = {
        'global_step': payload['global_step'] == global_step,
        'num_seen_examples': payload['replay_buffer']['num_seen_examples'] == model.buffer.num_seen_examples,
        'ordered_buffer_pair_names': (
            [s['names'] for s in payload['replay_buffer']['buffer_data']]
            == [s['names'] for s in model.buffer.buffer_data]),
        'numpy_rng': same_state(payload['rng_states']['numpy'], np.random.get_state()),
        'all_batch_streams': same_state(
            payload['stream_states'], {task: stream.state_dict() for task, stream in streams.items()}),
    }
    if not all(checks.values()):
        raise ValueError(f'Sampling reconstruction failed at {global_step}: {checks}')
    return checks


def stage_checkpoints(root, method):
    original = root / 'core_runs' / method / 'full' / {'mer': 'j3_01_s42', 'samcl': 'j3_03_s42'}[method]
    corrected = root / 'corrections/nlst_native_v4_20260831/core_runs' / method / 'full'
    nlst = corrected / {'mer': 'parallel_segment_04_to_30000_seed42',
                        'samcl': 'parallel_segment_03_to_30000_seed42'}[method]
    final = corrected / {'mer': 'parallel_segment_06_to_40000_seed42',
                         'samcl': 'parallel_gpu2_segment_05_to_40000_seed42_retry1'}[method]
    return [folder / 'checkpoints' / f'after_{task}.pt'
            for folder, task in zip((original, original, nlst, final), TASK_ORDER)]


def reconstruct(root, data_root, output):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', seed=42, steps_per_task=10000, batch_size=4,
                  buffer_capacity=200, sample_unit='ordered registration pair',
                  numerator='unique training pairs consumed in later tasks; union over later stages',
                  denominator='number of unique pairs in that task training dataset',
                  DRR_formula='mean(N_unique_later_replayed / N_train) over OASIS, CTCT, NLST',
                  replay_type='raw_images', notation='DRR*',
                  recovery='original observe/draw/Buffer code with numeric work disabled; checkpoint-gated',
                  path_scope='retained final training trajectory; excludes rolled-back attempts',
                  no_gpu_or_image_loading=True, methods=[], task_records=[], stage_records=[], checks=[])
    started = time.time()
    try:
        for method in ('mer', 'samcl'):
            random.seed(42); np.random.seed(42)
            config = RunConfig(method=method, budget='full', seed=42,
                               data_root=str(data_root), output_root=str(output.parent))
            cfg = build_source_cfg(config, torch.device('cpu'), output.parent)
            if cfg.dataset.intensity_aug or cfg.dataset.one_sample_only:
                raise ValueError('Metadata replay requires the original unaugmented full datasets')
            datasets = {task: Continual3D(cfg, task=task, mode='train') for task in TASK_ORDER}
            names, origin = {}, {}
            for task, dataset in datasets.items():
                names[task] = ['_'.join(Path(p).name[:-7] for p in dataset.dataset.get_image_name(i))
                               for i in range(len(dataset))]
                if len(set(names[task])) != len(dataset) or set(names[task]) & origin.keys():
                    raise ValueError('Training pair identities are not unique')
                origin.update({name: task for name in names[task]})
            streams = {task: StatefulBatchStream(datasets[task], 4, 42 + i * 1009)
                       for i, task in enumerate(TASK_ORDER)}
            model = SamplingOnly(method, cfg, origin)
            # Original model initialization and unaugmented preprocessing do not
            # consume NumPy RNG. Endpoint checks reject any divergence in that premise.
            for i, (task, checkpoint) in enumerate(zip(TASK_ORDER, stage_checkpoints(root, method))):
                model.task_index = i
                cfg.var.obj_operator.task_idx = i
                model.stage_used = {t: set() for t in TASK_ORDER}
                for _ in range(10000):
                    batch = {k: torch.zeros(4, 1) for k in ('imgs', 'masks', 'segs')}
                    batch.update(keypoints=[torch.zeros(1) for _ in range(4)],
                                 names=[names[task][j] for j in streams[task].next_indices()])
                    model.observe(batch)
                # mmap reads metadata without materializing the multi-GB image tensors.
                payload = torch.load(str(checkpoint), map_location='cpu', mmap=True)
                runner = payload['resolved_config']['runner']
                if any(runner[k] != v for k, v in dict(method=method, seed=42, budget='full', buffer_size=200).items()):
                    raise ValueError('Unexpected historical runner configuration')
                checks = verify_stage(payload, model, streams, (i + 1) * 10000)
                report['checks'].append(dict(method=method, after_task=task,
                                             checkpoint=str(checkpoint.relative_to(root)), **checks))
                for source in TASK_ORDER[:i]:
                    report['stage_records'].append(dict(method=method, current_task=task,
                        source_task=source, unique_pairs_used_in_stage=len(model.stage_used[source]),
                        cumulative_unique_pairs_used_later=len(model.historical[source])))
                print(json.dumps(dict(method=method, after_task=task, checks=checks)), flush=True)
                del payload
            values = []
            for task in TASK_ORDER[:-1]:
                numerator, denominator = len(model.historical[task]), len(names[task])
                ratio = numerator / denominator
                if not 0 <= ratio <= 1:
                    raise ValueError('Invalid unique-pair replay ratio')
                values.append(ratio)
                report['task_records'].append(dict(method=method, source_task=task,
                    unique_later_replayed_pairs=numerator, training_pairs=denominator, DRR_task=ratio))
            report['methods'].append(dict(method=method, DRR=sum(values)/len(values), tasks=len(values)))
        report.update(status='complete', elapsed_seconds=time.time()-started,
                      completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    except Exception as exc:
        report.update(status='failed', failure=str(exc), elapsed_seconds=time.time()-started)
        output.with_suffix('.failed.json').write_text(json.dumps(report, indent=2)+'\n')
        raise
    output.write_text(json.dumps(report, indent=2)+'\n')
    for key, filename in [('task_records', 'drr_task_metrics.csv'), ('stage_records', 'drr_stage_counts.csv')]:
        with (output.parent / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report[key][0]))
            writer.writeheader(); writer.writerows(report[key])
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--experiment-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = reconstruct(a.experiment_root, a.data_root, a.output)
    print(json.dumps(result['methods']), flush=True)
