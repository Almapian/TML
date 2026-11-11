# Legacy dense models: validation RMSE (comparison baseline)

Date: 2026-11-09. Branch `paper/nov-fixes`. Models: the existing dense stage 2 and dense stage 3 tide-only checkpoints (`models/checkpoints/[richmond/]dense/`), loaded and scored only, with no training. This is the baseline that new dense variants are compared against during development.

Command: `python models/run_gauge.py --pipeline dense --gauge richmond --gauge southend` (validation only, the default).

Sources:
- Validation: `outputs/[richmond/]dense/dense_val_metrics.csv` (new).
- Test: `outputs/[richmond/]dense/dense_metrics.csv` (published 2026-10-29, unchanged). It is shown for reference only.

## RMSE (m) at selected leads

Validation 2018 to 2020; test 2021 to 2024. All anchors in each period, flagged targets masked.

### Richmond Lock

| Model | Split | 10min | 1h | 6h | 24h | 48h | 72h |
|---|---|---|---|---|---|---|---|
| Harmonic baseline | val | 0.4980 | 0.4980 | 0.4982 | 0.4990 | 0.5002 | 0.5008 |
| | test | 0.5596 | 0.5597 | 0.5597 | 0.5599 | 0.5598 | 0.5597 |
| Dense stage 2 | val | 0.0880 | 0.1454 | 0.2226 | 0.2459 | 0.2802 | 0.3111 |
| | test | 0.0921 | 0.1468 | 0.2295 | 0.2561 | 0.2980 | 0.3316 |
| Dense stage 3 tide-only | val | 0.0854 | 0.1456 | 0.2237 | 0.2469 | 0.2802 | 0.3086 |
| | test | 0.0898 | 0.1517 | 0.2402 | 0.2635 | 0.3009 | 0.3360 |

### Southend Pier

| Model | Split | 10min | 1h | 6h | 24h | 48h | 72h |
|---|---|---|---|---|---|---|---|
| Harmonic baseline | val | 0.2365 | 0.2365 | 0.2365 | 0.2365 | 0.2362 | 0.2360 |
| | test | 0.2446 | 0.2446 | 0.2443 | 0.2436 | 0.2439 | 0.2453 |
| Dense stage 2 | val | 0.0458 | 0.0638 | 0.1754 | 0.2182 | 0.2210 | 0.2279 |
| | test | 0.0491 | 0.0645 | 0.1823 | 0.2176 | 0.2233 | 0.2378 |
| Dense stage 3 tide-only | val | 0.0556 | 0.0733 | 0.1768 | 0.2193 | 0.2208 | 0.2261 |
| | test | 0.0604 | 0.0768 | 0.1835 | 0.2209 | 0.2220 | 0.2321 |

## Checks

- Inference only: the log shows "loaded weights" and "left as-is" for every checkpoint, and no training epochs.
- 124 pre-existing files (legacy dense outputs, checkpoints, resume files, run state, `report_images/`) had identical SHA-256 hashes before and after the run.
- The mean squared validation RMSE over the 73 leads, in scaled units, equals each model's best validation loss in `dense_history.json`. That is 0.03367 and 0.03389 at Richmond, and 0.01807 and 0.01808 at Southend, matching to 5 decimal places. So the scoring uses the same validation windows and the same checkpoints that training selected.

## Discussion

- Validation RMSE is slightly lower than test at almost every lead, for both gauges and both models. The harmonic baseline shows the same gap (Richmond 0.50 vs 0.56 m), so the validation years are somewhat easier than the test years, not overfitted. Validation numbers should therefore be compared only with other validation numbers.
- The short-lead problem is visible on validation as well. Dense stage 2 is at 0.088 m (Richmond) and 0.046 m (Southend) at 10min. That is the target for the skip-connection variant.
