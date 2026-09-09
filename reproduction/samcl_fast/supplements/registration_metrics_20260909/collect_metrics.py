"""Recompute dimensionless metrics from the current native-v4 runs only."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

TASKS = ('oasis', 'ctct', 'nlst', 'mrct')


def relative_metrics(initial, final, reference=None, *, lower=False):
    for value in (initial, final, reference):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError('Ratio metrics require finite, strictly positive scores.')
    bwtr = initial / final - 1 if lower else final / initial - 1
    rma = None if reference is None else (reference / initial if lower else initial / reference)
    return bwtr, rma


def validated_drr(report):
    gates = ('global_step', 'num_seen_examples', 'ordered_buffer_pair_names', 'numpy_rng', 'all_batch_streams')
    expected = {(m, t) for m in ('mer', 'samcl') for t in TASKS}
    checks = report['checks']
    if (report['status'] != 'complete' or report['seed'] != 42 or report['steps_per_task'] != 10000
            or len(checks) != 8 or {(r['method'], r['after_task']) for r in checks} != expected
            or not all(r.get(k) is True for r in checks for k in gates)):
        raise ValueError('DRR reconstruction has not passed every checkpoint gate')
    rows = report['task_records']
    expected = {(m, t) for m in ('mer', 'samcl') for t in TASKS[:-1]}
    if len(rows) != 6 or {(r['method'], r['source_task']) for r in rows} != expected:
        raise ValueError('Incomplete DRR task coverage')
    values = {}
    for method in ('mer', 'samcl'):
        ratios = []
        for row in (r for r in rows if r['method'] == method):
            n, total = row['unique_later_replayed_pairs'], row['training_pairs']
            if not (isinstance(n, int) and isinstance(total, int) and 0 <= n <= total and total > 0):
                raise ValueError('Invalid DRR counts')
            if not math.isclose(row['DRR_task'], n / total):
                raise ValueError('DRR task ratio does not match counts')
            ratios.append(n / total)
        values[method] = mean(ratios)
    summaries = report['methods']
    if (len(summaries) != 2 or {r['method'] for r in summaries} != set(values)
            or any(r['tasks'] != 3 or not math.isclose(r['DRR'], values[r['method']]) for r in summaries)):
        raise ValueError('DRR aggregate does not match task counts')
    return values


def collect(root):
    drr_path = root / 'results/drr_reconstruction.json'
    drr_report = json.loads(drr_path.read_text()) if drr_path.exists() else None
    drr = validated_drr(drr_report) if drr_report else None
    matrix = {}
    for row in csv.DictReader((root / 'inputs/stage_metrics.csv').open()):
        key = (row['method'], row['after_task'], row['task'])
        if key in matrix:
            raise ValueError(f'Duplicate stage entry: {key}')
        matrix[key] = row
    references = {}
    for task in TASKS[1:]:
        path = root / 'references' / task / 'results/summary.json'
        if path.exists():
            summary = json.loads(path.read_text())
            if (summary['status'] != 'complete' or summary['steps'] != 10000
                    or summary['task'] != task or summary['seed'] != 42
                    or summary['geometry_protocol'] != 'native_v4'
                    or summary['reference'] != 'independent_plain_Adam'
                    or summary['history_access'] or summary['replay']):
                raise ValueError(f'Incomplete reference: {path}')
            if summary['evaluation']['pairs'] != int(matrix['mer', task, task]['pairs']):
                raise ValueError(f'Reference test coverage mismatch: {task}')
            references[task] = summary['evaluation']['value']
    records, summaries = [], []
    for method in sorted({key[0] for key in matrix}):
        method_rows = []
        for i, task in enumerate(TASKS):
            diagonal, last = matrix[method, task, task], matrix[method, TASKS[-1], task]
            expected = 'TRE_mm' if task == 'nlst' else 'Dice'
            if diagonal['metric'] != expected or last['metric'] != expected or diagonal['pairs'] != last['pairs']:
                raise ValueError(f'Incompatible evaluation entries for {method}/{task}')
            initial, final = float(diagonal['value']), float(last['value'])
            bwtr, rma = relative_metrics(initial, final, references.get(task), lower=task == 'nlst')
            row = dict(method=method, task=task, metric=expected, test_pairs=int(last['pairs']),
                       initial=initial, final=final, independent_reference=references.get(task),
                       BWT_raw=final-initial, BWTR=bwtr if i < 3 else None,
                       RMA=rma if i > 0 else None,
                       BWTR_status='computed' if i < 3 else 'not_applicable_last_task',
                       RMA_status=('not_applicable_first_task' if i == 0 else
                                   'computed' if task in references else 'pending_independent_reference'))
            records.append(row); method_rows.append(row)
        # Never average Dice and reciprocal TRE into a cross-metric ranking.
        for metric in ('Dice', 'TRE_mm'):
            rows = [r for r in method_rows if r['metric'] == metric]
            old = [r['BWTR'] for r in rows if r['BWTR'] is not None]
            current = [r for r in rows if r['task'] != TASKS[0]]
            rma_complete = all(r['RMA'] is not None for r in current)
            summaries.append(dict(method=method, metric=metric, BWTR=mean(old),
                                  BWTR_tasks=len(old), RMA=mean(r['RMA'] for r in current) if rma_complete else None,
                                  RMA_tasks=len(current), RMA_complete=rma_complete))
    out = root / 'results'; out.mkdir(exist_ok=True)
    for name, rows in [('task_metrics', records), ('group_metrics', summaries)]:
        with (out / f'{name}.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out / 'metric_status.json').write_text(json.dumps(dict(
        geometry_protocol='native_v4', seed=42, independent_reference_seed=42,
        reference_steps=10000, missing_references=[t for t in TASKS[1:] if t not in references],
        DRR_status='reconstructed_checkpoint_verified' if drr else 'missing_historical_counts',
        DRR=drr,
        TRE_score_transform='q = 1 / physical RMS TRE; BWTR = initial_TRE/final_TRE - 1; RMA = independent_TRE/initial_TRE',
        summaries=summaries), indent=2) + '\n')
    lines = ['# 当前配准复现：补充指标', '',
             '只使用 native-v4 实验记录。seed 42，原持续序列每任务 10,000 步；不使用论文表格数值。', '',
             '| 方法 | 指标组 | BWTR | RMA |', '|---|---|---:|---:|']
    for row in summaries:
        rma = '独立参照训练中' if row['RMA'] is None else f"{row['RMA']:.6f}"
        lines.append(f"| {row['method']} | {row['metric']} | {row['BWTR']:.6f} | {rma} |")
    lines += ['', 'BWTR：Dice 使用 final/initial - 1；TRE 使用 initial/final - 1，均越大越好。',
              'RMA：Dice 使用 initial/independent；TRE 使用 independent/initial。参照均为同配置的独立普通 Adam 训练。',
              'Dice-BWTR 汇总 OASIS、CTCT；Dice-RMA 汇总 CTCT、MRCT。NLST 单列，不跨 Dice/TRE 求总分。',
              '第一个任务不纳入 RMA；最后一个任务不纳入 BWTR。缺失参照不以零代替。',
              '该 RMA 比较固定外层步数下相对普通单任务 Adam 的建模水平，不单独分离 SAM、回放、额外内层更新与历史约束的作用。', '']
    profile = out / 'resource_profile.json'
    if profile.exists():
        profile_data = json.loads(profile.read_text())
        resource_rows = []
        for row in profile_data['records']:
            item = {k: row[k] for k in ('method', 'task', 'parameter_count', 'parameter_bytes',
                                 'replay_capacity', 'replay_occupied', 'replay_tensor_bytes',
                                 'train_step_median_s', 'train_peak_allocated_mib', 'inference_pair_median_ms')}
            stages = [r for r in profile_data['records'] if r['method'] == row['method']]
            item['MPE'] = 0.0 if len(stages) == 4 and len({r['parameter_count'] for r in stages}) == 1 else None
            resource_rows.append(item)
        if resource_rows:
            with (out / 'resource_metrics.csv').open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(resource_rows[0])); writer.writeheader(); writer.writerows(resource_rows)
        lines += [f"资源测量状态：{profile_data['status']}，已完成 {len(resource_rows)}/12 个方法—任务组合。", '',
                  '| 方法 | 当前任务 | 训练外层步中位数 s | 峰值显存 MiB | 单对推理中位数 ms |',
                  '|---|---|---:|---:|---:|']
        for row in resource_rows:
            lines.append(f"| {row['method']} | {row['task']} | {row['train_step_median_s']:.4f} | {row['train_peak_allocated_mib']:.1f} | {row['inference_pair_median_ms']:.2f} |")
        lines += ['', 'plain_update 是普通 Adam 更新的计算基线，使用 MER 对应阶段权重；不是新完成的顺序训练性能实验。',
                  '训练测量包括当前批次加载、传输、损失、反向传播和实际回放/元更新；推理为 GPU 上 batch=1 图像与掩膜的模型前向，不含磁盘读取与标签评分。',
                  '使用 3 次预热、20 次实测；短段耗时不外推为完整训练实测时间。回放字节数为张量逻辑载荷，不等于 RSS 或 checkpoint 文件大小。',
                  '固定网络无任务头增长；所有阶段参数量一致时 MPE=0。', '']
    else:
        lines += ['资源测量尚未完成。', '']
    if drr:
        lines += ['DRR 已通过原采样流程重放恢复；MER、SAMCL 各四个阶段的有序缓冲区、NumPy 随机状态、批次流及累计样本数均与已有 checkpoint 一致。', '',
                  '| 方法 | OASIS：回放独立对数 / 训练对数 | CTCT | NLST | DRR* |',
                  '|---|---:|---:|---:|---:|']
        for method, value in drr.items():
            counts = {r['source_task']: f"{r['unique_later_replayed_pairs']} / {r['training_pairs']}"
                      for r in drr_report['task_records'] if r['method'] == method}
            lines.append(f"| {method} | {counts['oasis']} | {counts['ctct']} | {counts['nlst']} | {value:.6f} |")
        lines += ['', 'DRR 按历史来源任务统计：后续阶段实际使用过的独立配准对取并集，再除以该任务训练配准对数，最后对前三个任务取均值；MRCT 没有后续任务，不纳入。',
                  '同一对重复回放或 SAM 重复前向只计一次；当前任务的缓冲区样本与抽到但未使用的样本不计入历史回放。星号表示回放原始图像；配准对是本实验的样本单位，不是独立患者数。',
                  '恢复范围是最终保留的训练轨迹，排除已回滚尝试；仅复现采样，不读取图像或重新训练。此指标不是回放更新次数或计算开销比。',
                  '逐来源任务与阶段计数见 drr_task_metrics.csv、drr_stage_counts.csv；核对证据见 drr_reconstruction.json。', '']
    else:
        lines += ['历史 DRR 所需的累计独立回放样本计数没有保留，不能由容量或当前缓冲区反推；采样恢复结果尚不可用。', '']
    lines += ['逐任务明细见 task_metrics.csv；指标状态见 metric_status.json。',
              '独立参照完成后，后台流程自动重新生成本报告。所有测试使用固定预算的最终模型，无测试集选优。',
              '此处报告当前复现实验口径，保留现有数据划分、前景 Dice、物理 RMS-TRE；不声称与其他实验协议相同。']
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    print(json.dumps(collect(parser.parse_args().root), indent=2))
