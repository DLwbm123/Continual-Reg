# Registration baseline supplement: completed

All four seed-42 native-v4 runs completed 40,000 updates (10,000 per task), with exit code 0 and empty stderr. Final completion: 2026-09-12 03:58 Asia/Shanghai. The two queues are complete.

The aggregate validation passes: 64 stage/task evaluations, 16 resource records, full matrix coverage, positive finite scores and independent recalculation of RMA, BWTR and DRR. Stage evaluations in the training runner load their saved checkpoint; this closeout validates their stored aggregate evidence without another inference run.

| Method | Dice RMA | TRE RMA | Dice BWTR | TRE BWTR | MPE | DRR | Recorded active hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| sequential | 1.1140 | 0.9855 | -0.4184 | -0.5595 | 0.0000 | 0.0000 | 10.1483 |
| ewc | 0.9985 | 0.8047 | -0.0807 | -0.0012 | 0.0000 | 0.0000 | 12.2786 |
| si | 1.0578 | 0.9871 | -0.4117 | -0.5427 | 0.0000 | 0.0000 | 11.0334 |
| gpm | 1.0698 | 1.0276 | -0.2676 | -0.4606 | 0.0000 | 0.4987 | 17.4075 |

Dice-RMA averages CTCT/MRCT; Dice-BWTR averages OASIS/CTCT. NLST uses physical RMS-TRE and the reciprocal score adapter, reported separately. OASIS RMA and MRCT BWTR are not applicable. These are newly run reproduction results, not the original thesis/paper scores; no thesis table was overwritten.

MPE is parameter expansion relative to the shared network, not regularizer/subspace storage; see per-task `method_state_bytes` in `resources.json`. GPM DRR (0.498730...) counts the fraction of pairs used for gradient-subspace construction averaged over the first three tasks; it is not ordinary sample-buffer replay. EWC Fisher estimation revisited 10,000/420/147 pairs, recorded separately in `boundaries.json`, despite sample-replay DRR=0. Timing was measured on shared GPUs; do not claim a matched comparison to earlier isolated microbenchmarks. Recorded active time includes boundary work/evaluation but, for the two resumed runs, excludes discarded work after the restart checkpoint.

This is one seed. SI uses the explicitly untuned c=1.0, xi=0.1 setting. Results do not establish exact paper-protocol reproduction. Data, pair identities, checkpoints, resolved private configurations and raw logs remain excluded from this public release. Included: training/adaptation code, portable queue, verification script, validation receipt, completion receipts, aggregate stage evaluations, resource measurements and summary metrics. The existing pinned runtime and independent references are in the sibling registration-metrics supplement.

Validation: `python validate_results.py` (Python standard library only).
