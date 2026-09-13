"""Validate aggregate results and regenerate the compact metric table (stdlib only)."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = ('oasis', 'ctct', 'nlst', 'mrct')


def read(path):
    return json.loads(path.read_text())


def main():
    rows = []
    for method in ('sequential', 'ewc', 'si', 'gpm'):
        run = ROOT / 'runs' / method
        s = read(run / 'results/summary.json')
        c = read(run / 'completion.json')
        assert c['exitcode'] == 0 and c['stderr_bytes'] == 0
        assert c['status'] == {'status': 'complete', 'global_step': 40000}
        assert s['status'] == 'complete' and not s['smoke']
        assert s['seed'] == 42 and s['steps_per_task'] == 10000
        evaluations = read(run / 'results/evaluations.json')
        matrix = {(v['after_task'], v['task']): v['value'] for v in evaluations}
        assert len(evaluations) == len(matrix) == 16
        assert set(matrix) == {(a, t) for a in TASKS for t in TASKS}
        assert all(math.isfinite(v) and v > 0 for v in matrix.values())
        assert all(v['pairs'] == s['test_pairs'][v['task']] for v in evaluations)
        for row in s['metrics']:
            t = row['task']; initial, final = matrix[t, t], matrix['mrct', t]
            assert row['initial'] == initial and row['final'] == final
            bwtr = (initial / final - 1 if t == 'nlst' else final / initial - 1) if t != 'mrct' else None
            assert row['BWTR'] is None if bwtr is None else math.isclose(row['BWTR'], bwtr, abs_tol=1e-12)
            if t != 'oasis':
                ref = read(ROOT.parent / 'registration_metrics_20260909/references' / t / 'results/summary.json')
                assert ref['status'] == 'complete' and ref['seed'] == 42 and ref['steps'] == 10000
                assert ref['geometry_protocol'] == s['geometry_protocol'] == 'native_v4'
                assert ref['evaluation']['pairs'] == s['test_pairs'][t]
                value = ref['evaluation']['value']
                assert math.isclose(row['RMA'], value / initial if t == 'nlst' else initial / value, abs_tol=1e-12)
            else:
                assert row['RMA'] is None
        resources = read(run / 'results/resources.json')
        assert len(resources) == 4 and {v['task'] for v in resources} == set(TASKS)
        assert all(math.isfinite(v[k]) and v[k] > 0 for v in resources for k in ('train_step_median_s', 'inference_pair_median_ms', 'peak_allocated_mib'))
        assert s['MPE'] == 0
        drr = sum(v['representation_pairs'] / v['source_train_pairs'] for v in s['boundaries']) / 3 if method == 'gpm' else 0
        assert math.isclose(s['DRR'], drr, abs_tol=1e-12)
        m = {v['task']: v for v in s['metrics']}
        rows.append(dict(method=method, Dice_RMA=(m['ctct']['RMA']+m['mrct']['RMA'])/2,
                         TRE_RMA=m['nlst']['RMA'], Dice_BWTR=(m['oasis']['BWTR']+m['ctct']['BWTR'])/2,
                         TRE_BWTR=m['nlst']['BWTR'], MPE=s['MPE'], DRR=s['DRR'], active_hours=s['active_seconds']/3600))
    with (ROOT / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print('PASS: 4 completed runs, 64 stage/task evaluations, 16 resource records, RMA/BWTR/DRR formulas')


if __name__ == '__main__':
    main()
