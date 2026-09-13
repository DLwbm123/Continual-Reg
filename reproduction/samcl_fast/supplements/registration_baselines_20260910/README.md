# Registration baseline supplement (2026-09-10)

Authorized scope: complete the Sequential, EWC, SI and GPM rows missing from the
registration metric supplement. Existing MER/SAMCL sequences and independent
references remain the source of their own results and are not retrained.

All four runs use seed 42, the existing native-v4 registration evaluator,
OASIS → CTCT → NLST → MRCT, 10,000 outer steps per task, batch 4, LR 1e-4,
LK-UNet with 8,187,014 parameters, and the existing prepared data splits.
The same existing Python/PyTorch environment is reused.

## Method configurations and adapters

- Sequential: existing `Sgd` implementation (ordinary Adam optimizer).
- EWC: upstream online EWC, lambda 1e7 and gamma 1 from `cfgs/ewc.yml`.
  Task-end empirical Fisher visits the full current training set, retaining
  names, masks, segmentations and landmarks so the registration loss is valid.
- SI: upstream update rule, with a device property and compatible observe
  signature. No SI preset exists in the pinned source or the supplied paper.
  This supplement explicitly fixes **c=1.0, xi=0.1**, without tuning; it does
  not claim these are the original paper's hyperparameters.
- GPM: upstream gradient-subspace algorithm, num_samples=200, threshold=.97,
  step_size=.005. At task boundaries, use up to 200 unique current-task pairs,
  with deterministic seed 42000 + task index. Handle a final partial batch.
  One first-order autograd call produces the layer gradients. Apply the same
  projection as `(G @ U) @ U.T`, avoiding the dense `U @ U.T` allocation;
  the upstream SVD/subspace update is unchanged.

The source checkout is not edited. Adapters are in `baseline_methods.py`.
Method-specific Fisher/importance/subspace state, optimizer, RNG, batch streams,
and stage results are saved for resume. Save every 1,000 steps and at every
boundary; an interrupted boundary update can be repeated from its pre-boundary
checkpoint. All four stage evaluations reload the saved model.

## Metrics and resource scope

RMA uses the completed independent plain-Adam CTCT/NLST/MRCT references from
`registration_metrics_20260909/references`. Dice and reciprocal physical RMS-TRE
are reported separately. The first task has no RMA; the final task has no BWTR.

MPE counts only prediction-network parameter growth. Sequential, EWC and SI have
DRR=0 under the no-sample-replay convention. EWC's additional Fisher data passes
are **not free**: `boundaries.json` records their sample count and duration.
GPM's representation DRR counts unique pairs used to construct retained gradient
subspaces divided by each source task's training pair count, averaged over the
first three tasks. It is representation-based DRR, without a raw-image star.
Auxiliary parameter statistics/subspaces are counted as `method_state_bytes`.

Training profiling records steps 4–23, after three warmup steps, within the
actual fixed training budget. Inference uses batch=1, 3 warmups/20 measurements,
image/mask model forward including warping. These runs use **shared GPUs**;
observational timings are labeled accordingly and must not be presented as a
matched uncontended comparison to the previous resource microbenchmark.

## Execution

`run_queues.sh` runs two queues (launch its controller in a detached session): GPU2 Sequential → SI, GPU3 EWC → GPM.
Each method writes `runs/<method>/train.jsonl`, `status.json`, `checkpoints/`,
and `results/`; logs and PID ledger live at the supplement root. A failed method
records its exit code and allows the next independent baseline to run;
insufficient memory at a subsequent launch marks that queue blocked.
No external GPU processes are interrupted. No scheduled monitoring is created.
The controller uses the neutral command alias `ctl`; GPU workers use `w02` and
`w03`. Runner paths and method arguments are passed through environment variables,
not the OS command line. Both `ps` (parents and workers) and `nvidia-smi` must be
checked after launch. Storage paths remain unchanged. Existing checkpoints are
resumed automatically; the PID ledger must be archived explicitly after stopping
the previous controller and its workers before a restart.

On 2026-09-10 the initial controller and workers were stopped to correct their
visible command names. Sequential and EWC resumed from OASIS step 1000. Original
control logs and training logs were preserved under `restarts/20260910T143610Z/`;
active training logs retain records through the checkpoint and append resumed
updates. Steps after the last checkpoint are recomputed. `LAUNCH.json` records
the replacement controller and verification evidence.

`--smoke` runs two updates per task, at most four boundary pairs and four test
pairs per task. Smoke outputs are excluded from formal metrics.

```
python -m unittest test_adapters
python run_baseline.py --method sequential --smoke --data-root "$LEARN2REG_ROOT" \
  --run-dir <new-smoke-directory> --reference-root <existing-reference-root>
```

Formal completion requires four full task stages, valid evaluations, ratio
metrics and method state, `.complete` and process exit. Public delivery follows
completion; patient images, pair identities and checkpoints containing private
state are excluded from the public artifact set.

## Completed release

All four formal runs completed; the final run finished at 2026-09-12 03:58 Asia/Shanghai. See `REPORT.md`, `metrics.csv`, and `runs/*/results/` for aggregate evidence.

The portable queue requires `RUN_ROOT` in its environment and the existing deployment layout (`corrections/nlst_native_v4_20260831/env.sh`). Use the pinned source/environment and native-v4 runtime published in the sibling `registration_metrics_20260909/runtime/` supplement; keep existing data and checkpoint storage. The queue script is the deployed script with its machine-specific root replaced by this environment variable.

Run `python validate_results.py` to independently verify matrix coverage and metric formulas against the published independent references. No patient-level data, resolved private paths, checkpoints or raw logs are included.
