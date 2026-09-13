"""Fixed-budget native-v4 baselines with task-boundary state and resumable checkpoints."""
import argparse
import csv
import json
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from baseline_methods import CLASSES, method_state, restore_method
from lib.samcl_runtime import RunConfig, TASK_ORDER, capture_rng_state, restore_rng_state
from lib.source_runtime import (build_source_cfg, StatefulBatchStream, _batch, _next_batch, _numeric,
                                _resolved_cfg, _atomic_torch_save, evaluate_all, evaluate_task)
from core.datasets.continual3d import Continual3D


class FirstBatch:
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return min(4, len(self.dataset))
    def __getitem__(self, i): return self.dataset[i]
    def __getattr__(self, key): return getattr(self.dataset, key)


class BoundaryLoader:
    def __init__(self, dataset, indices, device, pool):
        self.dataset, self.indices, self.device, self.pool = dataset, indices, device, pool
    def __len__(self): return (len(self.indices)+3)//4
    def __iter__(self):
        for i in range(0, len(self.indices), 4):
            yield _batch(self.dataset, self.indices[i:i+4], self.device, self.pool)


def bytes_of(value):
    if torch.is_tensor(value): return value.numel()*value.element_size()
    if isinstance(value, np.ndarray): return value.nbytes
    if isinstance(value, dict): return sum(bytes_of(v) for v in value.values())
    if isinstance(value, (list, tuple)): return sum(bytes_of(v) for v in value)
    return 0


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def configure(method, data_root, run, device):
    runner = RunConfig(method='multitask', budget='full', data_root=str(data_root), output_root=str(run.parent),
                       seed=42, input_size=(112, 96, 112))
    cfg = build_source_cfg(runner, device, run)
    cfg.model.task_sequential = True
    cfg.method.name = {'sequential': 'sgd', 'ewc': 'ewc_on', 'si': 'si', 'gpm': 'gpm'}[method]
    cfg.method.ewc.e_lambda = 1e7; cfg.method.ewc.gamma = 1
    cfg.method.gpm.threshold = .97; cfg.method.gpm.step_size = .005; cfg.method.gpm.num_samples = 200
    # SI has no preset in the pinned repository: explicit untuned reproduction choice.
    cfg.method.si = OmegaConf.create(dict(c=1.0, xi=.1))
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=CLASSES, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    run = a.run_dir.resolve(); run.mkdir(parents=True, exist_ok=True)
    (run/'checkpoints').mkdir(exist_ok=True); (run/'results').mkdir(exist_ok=True)
    if (run/'config_resolved.yaml').exists() and not a.resume:
        raise FileExistsError('Refusing to overwrite configured run')
    if (run/'.complete').exists(): raise FileExistsError('Run already complete')
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    cfg = configure(a.method, a.data_root.resolve(), run, device)
    model = CLASSES[a.method](cfg).to(device)
    train = {t: Continual3D(cfg, task=t, mode='train') for t in TASK_ORDER}
    tests = {t: Continual3D(cfg, task=t, mode='test') for t in TASK_ORDER}
    if a.smoke: tests = {t: FirstBatch(ds) for t, ds in tests.items()}
    cfg.var.obj_operator.train_sets = list(train.values()); cfg.var.obj_operator.test_sets = list(tests.values())
    streams = {t: StatefulBatchStream(train[t], 4, 42+i*1009) for i,t in enumerate(TASK_ORDER)}
    steps = 2 if a.smoke else 10000
    protocol = dict(method=a.method, seed=42, steps_per_task=steps, batch_size=4, geometry_protocol='native_v4',
                    source_commit='7cf685a7c429635a64380dd1d286accb68117b2c', smoke=a.smoke,
                    task_order=list(TASK_ORDER), train_pairs={t:len(ds) for t,ds in train.items()},
                    test_pairs={t:len(ds) for t,ds in tests.items()}, parameter_count=sum(x.numel() for x in model.parameters()),
                    SI_parameters=dict(c=1.,xi=.1,provenance='explicit untuned supplement configuration; no upstream SI preset'),
                    resource_scope='observational timings on shared GPU; not directly matched to prior isolated profiling')
    start_task, start_step, records, resources, boundaries, profiles = 0, 0, [], [], [], {}
    elapsed_before = 0.
    if a.resume:
        payload = torch.load(str(run/'checkpoints/latest.pt'), map_location='cpu', mmap=True)
        if payload['protocol'] != protocol: raise ValueError('Resume protocol mismatch')
        model.load_state_dict(payload['model']); model.opt.load_state_dict(payload['optimizer'])
        restore_method(model, a.method, payload['method_state'], device)
        for t,s in payload['stream_states'].items(): streams[t].load_state_dict(s)
        restore_rng_state(payload['rng_states'])
        start_task, start_step = payload['next_task'], payload['next_step']
        records, resources, boundaries, profiles = (payload[k] for k in ('evaluations','resources','boundaries','profiles'))
        elapsed_before = payload['active_seconds']; del payload
    else:
        OmegaConf.save(OmegaConf.create(dict(protocol=protocol, source=_resolved_cfg(cfg))), run/'config_resolved.yaml')
    started = time.perf_counter()
    def save(next_task, next_step, name='latest.pt'):
        _atomic_torch_save(dict(protocol=protocol, model=model.state_dict(), optimizer=model.opt.state_dict(),
            method_state=method_state(model,a.method), stream_states={t:s.state_dict() for t,s in streams.items()},
            rng_states=capture_rng_state(), next_task=next_task, next_step=next_step,
            evaluations=records, resources=resources, boundaries=boundaries, profiles=profiles,
            active_seconds=elapsed_before+time.perf_counter()-started), run/'checkpoints'/name)
    print(json.dumps(dict(status='started', **protocol)), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool, (run/'train.jsonl').open('a',buffering=1) as log:
        for i in range(start_task,4):
            task=TASK_ORDER[i]; cfg.var.obj_operator.task_idx=i
            offset=start_step if i==start_task else 0
            if a.method=='gpm': model.begin_task()
            if offset==0:
                model.eval(); model.before_epoch('val')
                batch=_batch(tests[task],[0],device,pool)
                inputs={k:batch[k] for k in ('imgs','masks','names')}
                times=[]
                with torch.no_grad():
                    for n in range(2 if a.smoke else 23):
                        torch.cuda.synchronize(); before=time.perf_counter(); model(inputs); torch.cuda.synchronize()
                        if n >= (1 if a.smoke else 3): times.append(time.perf_counter()-before)
                profiles[task]=dict(inference_pair_median_ms=statistics.median(times)*1000, train_times=[],
                    method_state_bytes=bytes_of(method_state(model,a.method)), peak_allocated_mib=0.)
                del batch,inputs
            model.train(); model.before_epoch('train')
            for step in range(offset+1,steps+1):
                measured=(1 <= step <= 2) if a.smoke else (4 <= step <= 23)
                if step==(1 if a.smoke else 4): torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize(); before=time.perf_counter()
                batch=_next_batch(streams[task],device,pool); model.observe(batch)
                metrics=_numeric(model.metrics); torch.cuda.synchronize()
                duration=time.perf_counter()-before
                if not all(math.isfinite(v) for v in metrics.values()): raise FloatingPointError(f'Nonfinite metric at {task}/{step}')
                if measured:
                    profiles[task]['train_times'].append(duration)
                    profiles[task]['peak_allocated_mib']=max(profiles[task]['peak_allocated_mib'],torch.cuda.max_memory_allocated()/2**20)
                if step==offset+1 or step%25==0 or step==steps:
                    rec=dict(task=task,task_step=step,global_step=i*steps+step,step_seconds=duration,**metrics)
                    log.write(json.dumps(rec)+'\n'); print(json.dumps(rec),flush=True)
                    write_json(run/'status.json',dict(status='training',**rec))
                del batch
                if step%1000==0 or step==steps: save(i,step)
            # Every stage evaluation is performed after loading the saved fixed-budget state.
            payload=torch.load(str(run/'checkpoints/latest.pt'),map_location='cpu',mmap=True)
            model.load_state_dict(payload['model']); del payload
            rng=capture_rng_state()
            _, stage_records=evaluate_all(model,tests,device,pool,task)
            if not all(math.isfinite(r['value']) and r['value']>0 for r in stage_records):
                raise ValueError('Invalid stage evaluation')
            records.extend(stage_records); restore_rng_state(rng)
            row=dict(method=a.method,task=task,parameter_count=protocol['parameter_count'],MPE=0,
                     train_step_median_s=statistics.median(profiles[task]['train_times']),**profiles[task])
            resources.append(row)
            if i<3 and a.method!='sequential':
                model.train(); model.before_epoch('train')
                target=(min(200,len(train[task])) if a.method=='gpm' else len(train[task]))
                if a.smoke: target=min(4,target)
                indices=np.random.default_rng(42000+i).permutation(len(train[task]))[:target].tolist()
                loader=BoundaryLoader(train[task],indices,device,pool)
                write_json(run/'status.json',dict(status='boundary_update',task=task,samples=target))
                before=time.perf_counter(); model.end_task(loader)
                # SI uses only its accumulated online state, not the boundary loader.
                used=0 if a.method=='si' else target
                boundaries.append(dict(task=task,source_train_pairs=len(train[task]),samples_revisited=used,
                    representation_pairs=target if a.method=='gpm' else 0,
                    seconds=time.perf_counter()-before))
                for value in method_state(model,a.method).values():
                    if torch.is_tensor(value) and not torch.isfinite(value).all(): raise ValueError('Nonfinite boundary state')
            save(i+1,0,'after_'+task+'.pt'); save(i+1,0)
            write_json(run/'results/evaluations.json',records)
            write_json(run/'results/resources.json',resources)
            write_json(run/'results/boundaries.json',boundaries)
            write_json(run/'status.json',dict(status='stage_complete',task=task,global_step=(i+1)*steps))
    matrix={(r['after_task'],r['task']):r['value'] for r in records}
    rows=[]
    for j,t in enumerate(TASK_ORDER):
        initial,final=matrix[t,t],matrix['mrct',t]
        ref=None
        if j and not a.smoke:
            refdata=json.loads((a.reference_root/t/'results/summary.json').read_text())
            if (refdata['status']!='complete' or refdata['seed']!=42 or refdata['steps']!=10000
                    or refdata['geometry_protocol']!='native_v4' or refdata['evaluation']['pairs']!=len(tests[t])):
                raise ValueError('Mismatched independent reference')
            ref=refdata['evaluation']['value']
        rows.append(dict(task=t,metric='TRE_mm' if t=='nlst' else 'Dice',initial=initial,final=final,
            BWTR=(initial/final-1 if t=='nlst' else final/initial-1) if j<3 else None,
            RMA=(ref/initial if t=='nlst' else initial/ref) if ref is not None else None))
    drr=sum(x['representation_pairs']/x['source_train_pairs'] for x in boundaries)/3 if a.method=='gpm' else 0.
    summary=dict(status='complete',**protocol,active_seconds=elapsed_before+time.perf_counter()-started,
        metrics=rows, MPE=0, DRR=drr, DRR_type='gradient_subspace_construction' if a.method=='gpm' else 'no_sample_replay',
        DRR_note='EWC Fisher revisits are separately counted in boundaries; regularization statistics are not replay memory',
        checkpoint_reload='all_four_stage_evaluations_loaded_saved_model',boundaries=boundaries)
    write_json(run/'results/summary.json',summary)
    write_json(run/'status.json',dict(status='complete',global_step=steps*4))
    (run/'.complete').write_text('complete\n'); print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
