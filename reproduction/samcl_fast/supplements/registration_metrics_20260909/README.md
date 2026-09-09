# Registration metrics supplement

This supplement fills missing metrics for the **existing seed-42 native-v4
reproduction**, without importing scores from a paper or rerunning the MER/SAMCL
continual sequences. It retains their input size (112 × 96 × 112), task order
(OASIS → CTCT → NLST → MRCT), fixed test splits and 10,000 outer steps per task.

Reference training and resource measurements completed on 2026-09-09 at 12:31 CST.
DRR reconstruction is now complete, with all eight checkpoint endpoints verified;
see `results/COMPLETION.json` and `results/REPORT.md` for final status and values.

## Outputs and scope

- `results/task_metrics.csv`: full-precision initial/final values, per-task BWTR
  and RMA, with explicit missing/not-applicable states.
- `results/group_metrics.csv`: Dice and TRE groups reported separately.
- `results/REPORT.md`: generated report; pending references remain pending.
- `results/resource_profile.json` and `resource_metrics.csv`: short, real-data
  resource measurements, once available. Per-step samples are retained.
- `references/{ctct,nlst,mrct}/results/summary.json`: fixed-budget independent
  references. Their absence does not become a zero score.

For Dice, BWTR = final / initial − 1 and RMA = initial / independent.
For positive physical RMS-TRE, the declared score adapter is q = 1 / TRE:
BWTR = initial_TRE / final_TRE − 1; RMA = independent_TRE / initial_TRE.
The reference is a shared **plain Adam single-task** model, freshly initialized
with seed 42, learning rate 1e−4, batch size 4, and 10,000 outer steps. No previous
task, replay, SAM, checkpoint selection or hyperparameter tuning is used.
RMA therefore measures performance relative to this ordinary single-task
training procedure; it does not isolate historical constraints from additional
inner updates or other optimizer differences.

OASIS is excluded from RMA. MRCT is excluded from BWTR. Dice-BWTR averages
OASIS/CTCT; Dice-RMA averages CTCT/MRCT. NLST is reported separately. Do not
average Dice and TRE into a total score. This is one seed, not a multi-seed
uncertainty estimate. Metrics use the native-v4 stage evaluation CSVs, including
the reevaluated inherited OASIS/CTCT stages; the later final-checkpoint reloads
verified those models but are not silently substituted for the matrix's last row.

The resource experiment compares plain updates, MER and SAMCL at each task's
entry state. The plain-update timing baseline loads MER's entry weights and
uses no buffer; it is not a newly trained sequential-performance baseline.
Three warmup updates and 20 measured updates are used. The data-loading-inclusive
training measurement includes the method's real replay and inner update path.
Inference uses a device-resident image/mask pair with batch size 1 and calls
the model's image-only forward, including deformation/warping; no reference
labels, loss/score computation, I/O or checkpoint loading are timed.
Short timings are not extrapolated into measured full-training duration.
Replay bytes mean logical tensor payload, not serialized size or process RSS.
The fixed network has no growing task head; MPE is zero when stage parameter
counts agree. The original runs did not log cumulative unique replay identities.
DRR is recovered separately by checkpoint-verified sampling reconstruction,
never inferred from buffer capacity. See the recovery protocol below.

## DRR recovery without retraining

`reconstruct_drr.py` runs the original MER/MERSAM `observe`, `draw_batches` and
`Buffer` methods on CPU with scalar placeholders. Numeric model/optimizer work
is disabled; no image volumes are loaded and no new performance scores are
generated. It reuses the training providers' ordered pair lists and the exact
`StatefulBatchStream` implementation. Training augmentation was disabled, and
reservoir insertion and replay selection use NumPy RNG independently of weights.

This is not an independently seeded approximation: at all four task endpoints
for each method, the ordered buffer pair identities, number of seen examples,
NumPy RNG state, global step, and all four batch-stream states must match the
existing checkpoint. No stage resets or corrections are made to force a match.
A mismatch writes a failure report and prevents release of the DRR result.
Checkpoint tensors are memory-mapped; only metadata are inspected.

For each historical source task, the numerator is the union of **ordered
registration pairs actually consumed in later tasks**. The denominator is that
source task's training pair count. DRR is the mean of those ratios over OASIS,
CTCT and NLST; MRCT has no later task and is excluded. The sample unit is a
registration pair, not an individual image or patient. A repeated pair or a
second SAM forward counts once. Buffer draws that the method discards do not
count; MERSAM's otherwise-unused first draw must still consume its original RNG.
Same-task buffer use does not count as historical replay. `DRR*` denotes raw
image replay; it does not measure optimizer updates or elapsed computation.

The reconstruction follows the retained final training trajectory. Rolled-back
attempts are excluded, consistent with the checkpoints used for RMA/BWTR.
Pair names remain in memory; the public output contains only aggregate counts
and checkpoint gate outcomes. No patient identities or replay tensors are exported.

In the existing experiment environment, with the runtime paths below configured:

```sh
python -m unittest test_drr
python reconstruct_drr.py --experiment-root "$EXPERIMENT_ROOT" \
  --data-root "$LEARN2REG_ROOT" --output results/drr_reconstruction.json
python collect_metrics.py --root .
```

The output must not already exist. Outputs are `drr_reconstruction.json`,
`drr_task_metrics.csv`, and `drr_stage_counts.csv` under `results/`. The metric
collector accepts the aggregate only after all eight checkpoint gates and the
six source-task counts pass validation.

## Recompute the available metrics without a GPU

```sh
python collect_metrics.py --root .
python -m unittest test_metrics.py
```

`inputs/stage_metrics.csv` contains aggregate task-level scores only. Provenance
points to existing experiment artifacts; no patient images, labels, individual
predictions, checkpoint weights or replay contents are distributed here.

## GPU runtime

The running experiment reuses its existing Python 3.10 / PyTorch 2.1.2 environment.
`runtime/environment.samcl.yml` documents the environment for other machines;
no installation or environment replacement is needed on the experiment host.
The helper modules in `runtime/lib` are copied from the existing runner, not a
new implementation of MER/SAMCL. The source correction is supplied as a patch
so the repository's main implementation is not overwritten by this supplement.

To reconstruct the source layout on another machine, set `SAMCL_ROOT` to an
empty experiment directory and `SUPPLEMENT_ROOT` to this directory, then:

```sh
mkdir -p "$SAMCL_ROOT/src"
git clone https://github.com/xzluo97/Continual-Reg "$SAMCL_ROOT/src/Continual-Reg"
git -C "$SAMCL_ROOT/src/Continual-Reg" checkout 0b63d5bb06ff2db1bbada50fdbbc2a100bfef3db
git -C "$SAMCL_ROOT/src/Continual-Reg" apply "$SUPPLEMENT_ROOT/runtime/native_v4.patch"
git clone https://github.com/xzluo97/deep_kit "$SAMCL_ROOT/src/deep_kit"
git -C "$SAMCL_ROOT/src/deep_kit" checkout 96b738debb1166be85381157bcb634f4bbce0329
export PYTHONPATH="$SAMCL_ROOT/src/Continual-Reg:$SAMCL_ROOT/src/deep_kit/src:$SUPPLEMENT_ROOT/runtime"
```

The patch represents the already used corrected source state
`7cf685a7c429635a64380dd1d286accb68117b2c`. It includes data-root configuration,
reservoir rejection handling, the MER beta spelling correction, native-grid
landmark sampling and physical-mm TRE. It does not change that run's training
or evaluation protocol for this supplement.

Use the same prepared Learn2Reg split/preprocessing as the existing experiment;
these scripts do not download data or recreate private/prepared test splits.
Set `LEARN2REG_ROOT` to that data directory. A reference command is:

```sh
CUDA_VISIBLE_DEVICES=2 python run_reference.py --task ctct \
  --data-root "$LEARN2REG_ROOT" --run-dir "$SUPPLEMENT_ROOT/references/ctct"
```

Repeat for `nlst` and `mrct`. Each run saves a resumable latest checkpoint every
1,000 steps and evaluates the reloaded final checkpoint. `--resume` continues
that exact reference; a new invocation refuses to overwrite a configured run.
After all references finish, run `collect_metrics.py` again to fill RMA.

`profile_resources.py --help` describes the resource measurement arguments.
It requires the existing stage checkpoints laid out in the experiment root
(specified in the script) and mutates only temporary in-memory model copies.
The site launcher `run_supplement.sh` uses GPU2 for CTCT/MRCT references and
GPU3 for uncontended resource measurements followed by the NLST reference.
It is a finite background pipeline with a lock and job ledger, not a recurring
monitor. It automatically collects final metrics after its workers exit.

## Validation

The metric checks cover Dice ratios, the lower-is-better TRE adapter and unit
invariance, missing references, and invalid denominators. A real CTCT two-step
training/checkpoint/reload/evaluation check and MER/SAMCL CTCT resource smoke
checks passed in the existing environment. Their tiny smoke scores are not
included in the reported performance metrics.
