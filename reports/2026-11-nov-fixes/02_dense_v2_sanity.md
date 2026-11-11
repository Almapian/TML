# 02 Dense v2: short-lead investigation and sanity runs (Southend, validation)

Date: 2026-11-10. Branch `paper/nov-fixes`, commits `961a545` (skip), `639d5a4` (pad), `191303f` (weighted loss).
Validation only, apart from the last-12-lead table below, which also lists already-published test numbers. The sanity models were trained for 2 epochs and are not results of record.

## 1. How the 10-min lead is produced, and bug checks

- **How it is produced.** The dense stage 2 model (`TidalSeq2Seq`) encodes 16h of gauge history with an LSTM. A bidirectional LSTM decoder then steps once per lead (10 min, then 1h to 72h), fed UTide at each target time. Its initial state is projected from the encoder's final state, and attention looks back over the encoder outputs. A single linear readout, shared by all 73 steps, gives each lead's level.
  - The last observation reaches the 10-min output only through that projected state and the attention context.
  - To be accurate the network must reproduce close to an identity map (the level now, plus the tidal change over 10 min), to about 1 cm on a scale whose standard deviation is 1.57 m.
- **Bug checks**, on real Southend validation windows. All pass, and no bug was found:
  - the last encoder value is the observation at the anchor;
  - every lead's target time is the anchor plus its lead, for all 73 leads;
  - the targets are the observations at those times;
  - the lead steps are 1, then 6, 12, ..., 432 (10 min, then hourly);
  - UTide sits at the target times;
  - one scaler (train mean and standard deviation) is used for every lead.
- **Loss imbalance (confirmed).** On validation, the 10-min lead contributes 0.064% of the summed per-lead MSE at Southend and 0.154% at Richmond. The 1h lead contributes 0.125% and 0.419%. In the sparse model each lead was about 1/7 of the loss.

## 2. The Southend 72h jump (report only)

Dense stage 2 RMSE (m) over the last 12 leads. Validation is from `outputs/dense/dense_val_metrics.csv`; test was already published in `outputs/dense/dense_metrics.csv`.

| | 61h | 62h | 63h | 64h | 65h | 66h | 67h | 68h | 69h | 70h | 71h | 72h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val | 0.2220 | 0.2223 | 0.2228 | 0.2232 | 0.2233 | 0.2233 | 0.2235 | 0.2236 | 0.2233 | 0.2231 | 0.2241 | 0.2279 |
| test | 0.2263 | 0.2268 | 0.2277 | 0.2285 | 0.2287 | 0.2289 | 0.2295 | 0.2298 | 0.2296 | 0.2300 | 0.2327 | 0.2378 |

- RMSE is flat from 61h to 70h and jumps over the last two leads.
- The decoder has no positional encoding and no padding. The 72h step is the last step of the BiLSTM, so its backward direction has seen only one input, UTide(72h), on top of its initial state. Every earlier step has later context.
- Padded dense stage 3 (3 hourly points after 72h) rises much less from 71h to 72h on validation: 0.0018 m against 0.0038 m.
- The diagnostics also showed the bias jumping at the last lead.
- This is what `dense_v2_pad` addresses.

## 3. Sanity runs: Southend, 2 full epochs, validation RMSE (m), same 157,297 anchors

| Model | 10min | 1h | 6h | 24h | 48h | 72h | mean over 73 |
|---|---|---|---|---|---|---|---|
| Old dense (fully trained, 45 epochs) | 0.0458 | 0.0638 | 0.1754 | 0.2182 | 0.2210 | 0.2279 | 0.2088 |
| dense_v2_skip (2 epochs) | 0.0488 | 0.1032 | 0.2167 | 0.2378 | 0.2435 | 0.2537 | 0.2372 |
| dense_v2_pad (2 epochs) | 0.1517 | 0.1816 | 0.2351 | 0.2339 | 0.2340 | 0.2455 | 0.2321 |
| dense_v2_skip_wloss (2 epochs) | 0.0229 | 0.0909 | 0.2322 | 0.2790 | 0.2894 | 0.3113 | 0.2767 |
| Residual persistence | 0.0197 | 0.0989 | 0.2512 | 0.3128 | 0.3243 | 0.3360 | 0.3071 |
| Harmonic baseline alone | 0.2365 | 0.2365 | 0.2365 | 0.2365 | 0.2362 | 0.2360 | 0.2363 |

Sources: `outputs/<variant>_sanity2/dense_val_metrics.csv` and `outputs/dense/dense_val_metrics.csv`. The table is also in `outputs/diagnostics_2026-11/sanity_dense_v2_southend_val.csv`.

Validation loss (scaled MSE averaged over the 73 leads) after epochs 1 and 2:

| Model | Epoch 1 | Epoch 2 |
|---|---|---|
| Old dense | 0.02247 | 0.02211 |
| dense_v2_skip | 0.03548 | 0.02312 |
| dense_v2_pad | 0.02222 | 0.02183 |

The weighted-loss variant's validation loss is on a different scale, so it isn't comparable.

What each variant does at 10 min, measured against residual persistence:

| Model | Correction rms | Error bias | Error std |
|---|---|---|---|
| dense_v2_skip | 0.043 m | +0.013 m | 0.047 m |
| dense_v2_skip_wloss | 0.020 m | +0.000 m | 0.023 m |

## Discussion
- **dense_v2_skip fails the success criterion.** Its 10-min RMSE (0.0488 m) is not below the old model's (0.0458 m), and is far from residual persistence (0.0197 m). The skip gives a perfect starting point, but the network adds a 4.3 cm rms, biased correction on top of it. Under equal per-lead weighting nothing pushes the 10-min correction towards zero. **The skip changes what the network starts from, not what the loss asks of it.** It is also slightly behind the old model at epoch 2 overall (0.0231 vs 0.0221).
- **The imbalance is the real cause.** Adding the persistence-normalised loss (`dense_v2_skip_wloss`) brings 10 min to 0.0229 m with zero bias after 2 epochs. But with weights of about 0.3 at 24 to 72h, the long leads are badly behind (72h 0.311 m). Whether full training recovers them is unknown.
- Residual persistence is a strong base only for the first hour or so: it is worse than the old model from 1h onwards (0.0989 vs 0.0638 m at 1h).
- **dense_v2_pad** is on track overall: validation loss 0.02183 at epoch 2 against 0.02211 for the old model. Its short leads are, as expected, untouched by padding and still undertrained at 2 epochs. Its 71h to 72h step is 0.0024 m, inconclusive at 2 epochs.

## Timings and memory (Southend, measured)

| Variant | Seconds per epoch (full training set + validation pass) | Peak RAM |
|---|---|---|
| dense_v2_skip | 91 to 92 | 6.7 GB |
| dense_v2_pad | 101 to 113 | 6.7 GB |
| dense_v2_skip_wloss | 115 to 122 | 6.7 GB |

- Peak RAM is the process working set, dominated by the UTide fit, and is far under the 24 GB limit.
- Data preparation plus scoring adds about 1.5 to 2 min per run.
- Richmond has the same number of windows (235,728 training, 156,079 validation, against Southend's 234,088 and 157,297), so the same per-epoch cost and similar RAM are expected. Richmond was not measured.

Projected full runs, at most 60 epochs with patience 10. The legacy dense stage 2 stopped at epoch 45 (Southend) and 33 (Richmond).

| Run | If it stops like the legacy model | Worst case (60 epochs) |
|---|---|---|
| dense_v2_skip, Southend | about 71 min | about 94 min |
| dense_v2_skip, Richmond | about 53 min | about 94 min |
| dense_v2_pad, Southend | about 83 min | about 110 min |
| dense_v2_pad, Richmond | about 62 min | about 110 min |
| Queue items 1 to 4 in total | about 4.5h | about 7.3h |
