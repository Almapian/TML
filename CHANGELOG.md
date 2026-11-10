# Changelog and run record

Every change to the pipeline, and every step taken to set up or carry out a run, is
recorded here, newest first. Entries say what changed, why, and how it was checked.

---

## 2026-11-10 (about 00:45): overnight queue built, dry-run passed, launched

### Code (commits `2352fae`, `fb967e4`, `a2f2c86`)
- **Post-processing** (`models/postprocess_hw.py`): the dense correction to UTide is interpolated (linear and cubic) onto all 433 ten-minute leads; high-water and low-water height and timing errors are reported for UTide, native leads, both interpolations and the AR residual baseline at 10-min resolution.
  - Checked on the legacy dense **validation** predictions, both gauges (`outputs/[richmond/]dense/postprocess/`). Southend high-water height bias goes from -6.7 cm (native leads only) to -3.8 cm (linear) and -3.3 cm (cubic); timing MAE goes from 15.2 to about 5 min. Cubic overshoots between the 10-min and 1h knots (0.074 m vs 0.050 m linear at 30 min).
  - Richmond low waters are **not scored**: they pile up against a hard floor near -1.0 m ODN (almost none below -1.06 m; summer medians sit on it) and stay within 5 cm for a median 60 min, against 30 min at high water.
- **dense_v2_skip_longmem**: skip plus 30 daily-mean residuals before the anchor's day (with missing flags), fed into the decoder initial state through a zero-initialised MLP. Smoke check: trains, 6.7 GB.
- **Queue** (`scripts/overnight_queue.py`): as agreed, in the order recorded in the PR, with selection by the amended rule.
  - Baselines are re-evaluated against the winner from their saved test forecasts (AR residual reproduction spot check, parity check), plus joint block-bootstrap tests over all 73 leads and over 24 to 72h.
- Tests: `tests/test_dense_v2.py` 12/12, `test_stage3_windowing.py` 5/5, `test_baselines.py` 12/12.

### Dry run (00:17 to 00:40, `--dry-run`: smoke training, 50 bootstrap reps, 2,000-anchor post-processing)
- All 19 steps ok, plus the report, commit (`830a88a`), push and PR comment, all from the detached launch. That proves non-interactive git/gh from outside Claude Code.
- The throwaway report and figures were removed in the next commit. All `*_smoke` models and outputs were deleted.

### Disclosures
- **Test viewed:**
  - The joint-test values of the *legacy* dense model against the baselines (Southend, 50 reps) were printed while testing `eval_baselines_vs_dense.py`. Both models' test results were already published.
  - The dry-run summary printed test RMSE of 8-batch smoke models, which are meaningless and have been deleted.
  - No configuration choice used either.
- **Deleted by mistake** during the dry-run cleanup: `outputs/richmond/baselines_smoke/`. This was a gitignored scratch folder from the 31 Oct baseline smoke check, not a result of record.

### Launch
- `powershell -ExecutionPolicy Bypass -File scripts\launch_queue.ps1`, which starts pythonw through Win32_Process.Create, outside the terminal's process tree.
- Logs: `logs/overnight_2026-11-10/`. Status: `python scripts/queue_status.py`.
- Deadline 17:00 Europe/London; hard stop for steps 16:35.
- ntfy: not configured (no topic given).
- Power settings checked with powercfg: sleep and hibernate after idle are "never" (AC and DC); **lid close = Sleep (AC and DC)**, which keep-awake does not override, so the lid must stay open.

## 2026-11-10: new variant dense_v2_skip_scaled (lead-scaled skip correction)

- **What.** It is `dense_v2_skip` plus one change: the network's correction at lead k is multiplied by a fixed scale, s_k = (training residual-persistence RMSE at k) / (its mean over the 73 leads). The scale is computed once from the training windows and stored as a buffer in the checkpoint. The loss is unweighted and unchanged.
- **Why.** In the skip sanity run, the network added a 4.3 cm rms correction at 10 min where persistence is already within 2 cm. Scaling keeps the correction structurally small at short leads (s = 0.089 at 10 min, 0.336 at 1h, 0.987 at 24h, 1.078 at 72h at Southend) without reweighting the loss.
- **Checks.** `tests/test_dense_v2.py` 8/8, including that a zeroed network still gives residual persistence, the correction is scaled per lead, and the scale round-trips through the state dict.
- **Run-check.** Southend, 1 epoch (about 460 batches): trains, finite losses and RMSE, 103 s per epoch, peak RAM 6.7 GB. Log: `logs/sanity/southend_dense_v2_skip_scaled_check1.log`. The run-check checkpoint was deleted. No test scoring.

## 2026-11-10: AMENDED pre-registered selection rule (supersedes the rule in the "decisions" entry below)

Written before any full training run of any dense v2 variant. Disclosure: the 2-epoch (skip, pad, skip_wloss) and 1-epoch (skip_scaled run-check) Southend validation sanity numbers had been seen when this was written. They come from undertrained models and are not used as evidence for or against any candidate.

- **Candidates, per gauge (Southend Pier, Richmond Lock):**
  - the old dense stage 2 (`dense`, existing checkpoint, never retrained);
  - `dense_v2_skip`;
  - `dense_v2_skip_wloss`;
  - `dense_v2_skip_scaled`;
  - `dense_v2_pad`.

  All except the old model are trained from scratch, with full training and early stopping on each variant's own validation loss. For `dense_v2_skip_wloss` that is its weighted training objective.
- **Score.** For each of the 73 dense leads h, on validation (2018 to 2020): RMSE_h(candidate) / min(RMSE_h(residual persistence), RMSE_h(UTide alone)). The score is the arithmetic mean of that ratio over all 73 leads. **Lowest score wins**, separately per gauge.
  - Every RMSE is per lead, over every validation anchor, flagged targets masked, by `training.masked_rmse_mae`.
  - Candidate RMSE: the `Dense stage 2` rows of `outputs/[richmond/]<variant>/dense_val_metrics.csv`.
  - Reference RMSEs: the `Residual persistence` and `UTide standalone` rows of the same files. All candidates share the same validation anchors, so these rows are identical across files; the selection script asserts that. The old model's file has no persistence row, so its references come from a v2 file.
- **Test scoring.** Only the winner per gauge is scored on test (`--score-test`). If the old dense model wins, no new test scoring happens; its test numbers are already published.
- **Reporting.** Full per-lead validation curves for every candidate go in the report, whatever wins.
- **If a candidate does not finish** (dropped, crashed or timed out), it is reported as such and the rule is applied to the candidates that did finish.

## 2026-11-10: dense v2 variants (skip, pad, weighted loss): investigation, code, sanity runs

### Why
The dense model is worse than residual persistence at 10 min. Southend 10-min test RMSE (published) is 0.049 m dense vs 0.017 m sparse, and residual persistence gets 0.020 m. There is also a jump at the last lead (72h).

### Investigation (no bug found)
- Anchor/lead alignment, target times, the 10-min step, UTide placement and per-lead scaling were all checked on real Southend validation windows; all correct.
- Loss imbalance confirmed: the 10-min lead is 0.064% (Southend) and 0.154% (Richmond) of the summed validation MSE.
- 72h: the last BiLSTM step has no later context and no padding. Southend RMSE is flat from 61h to 70h, then jumps (validation 0.2231 / 0.2241 / 0.2279 at 70 / 71 / 72h).

### Code (commits `961a545`, `639d5a4`, `191303f`)
All variants are registered in `config.DENSE_VARIANTS`, every flag defaults off, and legacy checkpoints load unchanged (no new parameters).
- `dense_v2_skip`: output = UTide(t+k) + obs(t) - UTide(t) + network correction.
- `dense_v2_pad`: 3 hourly UTide steps past 72h, from a gap-free grid so anchors are unchanged, excluded from loss and scoring.
- `dense_v2_skip_wloss`: skip plus the per-lead loss divided by the training residual-persistence MSE (normalised to mean 1).
- v2 variants do not retrain dense stage 3.
- `TIDAL_SANITY_EPOCHS=N` gives short runs into `*_sanityN` dirs.
- Residual persistence is now a scored reference row.
- Tests: `tests/test_dense_v2.py` 7/7, including "network output zeroed = residual persistence exactly". `test_stage3_windowing.py` 5/5 and `test_baselines.py` 12/12 still pass.

### Sanity runs (Southend, 2 full epochs, validation only)
`TIDAL_GAUGE=southend TIDAL_DENSE_VARIANT=<v> TIDAL_SANITY_EPOCHS=2 python models/train_dense.py`, logs in `logs/sanity/`, outputs in `outputs/<v>_sanity2/`. The sanity checkpoints were deleted afterwards.

Validation RMSE at 10min / 1h / 24h / 72h (m):

| Model | 10min | 1h | 24h | 72h |
|---|---|---|---|---|
| Old dense (fully trained) | 0.0458 | 0.0638 | 0.2182 | 0.2279 |
| skip | 0.0488 | 0.1032 | 0.2378 | 0.2537 |
| pad | 0.1517 | 0.1816 | 0.2339 | 0.2455 |
| skip_wloss | 0.0229 | 0.0909 | 0.2790 | 0.3113 |
| Residual persistence | 0.0197 | 0.0989 | 0.3128 | 0.3360 |

- **dense_v2_skip fails the success criterion.** Its network adds a 4.3 cm rms correction with +1.3 cm bias at 10 min, because equal per-lead weighting still gives that lead almost no gradient.
- The weighted loss fixes 10 min but lags at long leads after 2 epochs.
- Report: `reports/2026-11-nov-fixes/02_dense_v2_sanity.md`.

### Timing and RAM (measured, Southend)
- Seconds per epoch: 91 to 92 (skip), 101 to 113 (pad), 115 to 122 (skip_wloss).
- Peak RAM 6.7 GB for all three.
- Projected full runs: about 53 to 110 min each; queue items 1 to 4 about 4.5h to 7.3h.

### Test
None of the dense v2 models was scored on test. The 72h table in the report lists the old model's already-published test numbers next to validation.

## 2026-11-10: decisions after the diagnostics; overnight queue plan and pre-registered selection rule

### Disclosure: future leak in the stage 3 discharge and rainfall inputs (dissertation results affected)
- **What.** In every stage 3 meteo-fusion model trained before 2026-11-09, an anchor on calendar day D used the NRFA daily values dated D. NRFA gauged daily flow is the mean over the 09:00 to 09:00 GMT water day, dated by its start, so the value dated D includes flow up to 09:00 on D+1. The model therefore saw up to 33h of future Kingston discharge. Catchment rainfall used the same mapping; its day convention is unconfirmed but likely the same.
- **Which dissertation results.** Only the stage 3 **meteo-fusion** model (`TidalSeq2SeqMulti`, with the river branch), at **both Southend Pier and Richmond Lock**:
  - its RMSE tables and figures;
  - its SHAP branch attributions (`plot_stage3.py`);
  - the Southend learning-rate search that chose its hyperparameters, which Richmond then reused.
- **Not affected.** Every model without river inputs: stage 1, stage 2, stage 3 tide-only, the stage 3 reduced-training ablation (tide-only), the dense models and the classical baselines.
- **Direction.** The leak gave the model future information, so it **flattered** the affected results rather than hurting them.
  - Evidence (Richmond, validation): feeding the existing model leak-free river inputs raises its RMSE by 0.003 m at 10 min and 0.011 m at 72h. Source: `outputs/diagnostics_2026-11/B_ablation_val_rmse.csv`.
  - That is an indicative size only: a model trained with the leak and scored without it is also seeing a shift in its inputs.
  - Southend was not measured.
  - The leak does **not** explain the poor Richmond stage 3 short-lead accuracy, which is an epoch-1 checkpoint (see the diagnostics entry below).
- **Fix.** Commit `46b998b`: river values are used only after their water day has ended (`config.RIVER_AVAILABILITY_LAG_HOURS = 33`, the default). `config.LEGACY_RIVER_LAG_HOURS = 0` is kept **only** to reproduce dissertation numbers. Nothing has been re-scored or retrained with it.

### Decisions (Al, 2026-11-10)
1. **Richmond stage 3 retrain: deferred, not tonight.** A follow-up needs (a) the validation-scoring change in `train_stage3.py` first and (b) a Richmond-specific hyperparameter search instead of Southend's winners.
2. **New variant `dense_v2_pad`.** Dense stage 2 with the decoder extended past 72h by **3 hourly steps (73h, 74h, 75h)**, fed UTide at those times. The extra steps are excluded from the loss and from scoring. No skip connection. Trained from scratch, flag-controlled, default off.
   - Why 3 steps:
     - It is the convention dense stage 3 already uses (`DENSE_PAD_STEPS = 3`, hourly spacing), so the stage 2 and stage 3 decoders end the same way.
     - The evidence suggests it is enough to remove most of the edge effect. From 71h to 72h on validation, padded dense stage 3 rises 0.0018 m (Southend) and 0.0005 m (Richmond), against 0.0038 m and 0.0020 m for unpadded dense stage 2 (`outputs/[richmond/]dense/dense_val_metrics.csv`).
     - 3h is a quarter of an M2 cycle of known-future tide for the backward LSTM pass at 72h.
     - The decoder grows from 73 to 76 steps, about 4% more compute.
   - The padding UTide must come from the extended reconstruction, not from gauge rows, so the anchor set stays identical to the legacy model's.
3. **NRFA alignment.** The legacy mapping stays as an option for reproducing dissertation numbers only; no re-scoring now (recorded above).
4. **Long-memory variant: low priority**, because dense and AR are level over 24 to 71h. It is the last optional queue item.

### Overnight queue order (Prompt 3)
1. dense_v2_skip, Southend
2. dense_v2_skip, Richmond
3. dense_v2_pad, Southend
4. dense_v2_pad, Richmond

Steps 1 to 4: all trained from scratch, validation scoring only.

5. Select the winner per gauge by the rule below; test-score only the winner.
6. Baseline re-evaluation against the winner (with the joint tests).
7. Interpolation and high-water post-processing for the winner.
8. Optional if time: dense_v2_skip_wloss Southend, then dense_v2_skip_longmem Southend (validation only).
9. Always: report, commit, push, PR comment.

Combining skip and pad is not queued until both have been seen individually.

### Pre-registered selection rule (written 2026-11-10, before any dense_v2 variant has been trained or scored)
**Superseded** by the amended rule at the top of this file (same day, before any full run).
- **Candidates:** `dense_v2_skip` and `dense_v2_pad`, separately per gauge (Southend Pier, Richmond Lock).
- **Metric:** the arithmetic mean, over all 73 dense leads (10min and 1h to 72h), of the validation RMSE (2018 to 2020) in `outputs/[richmond/]<variant>/dense_val_metrics.csv`, for model `Dense stage 2`. This is per-lead RMSE over every validation anchor, with flagged targets masked, by `training.masked_rmse_mae`.
- **Winner:** the candidate with the lower metric. Only the winner is scored on test (`--score-test`); the loser's test numbers are never computed.

## 2026-11-10: diagnostics A (bias decomposition) and B (stage 3 at Richmond); river leak fix

### Why
PR item (a): test whether the AR residual baseline's 72h advantage at Southend comes from a slowly varying mean offset. PR item (b): find out why Richmond stage 3 (meteo fusion) is about 3.5x worse than its tide-only ablation at 10 min.

### What ran (no training)
- `models/diag_bias_decomposition.py`: harmonic baseline bias/std/RMSE per split and calendar year; dense stage 2 vs AR residual per lead.
  - Inputs: saved dense predictions and AR test predictions.
  - AR validation predictions were regenerated from the frozen `best_configs.json` through `eval_baselines`' code path (deterministic Yule-Walker on train, no search). They reproduce the saved test predictions to 2.4e-7 m.
- `models/diag_stage3_richmond.py`: the existing Richmond stage 3 checkpoints were loaded and scored on validation, with each input replaced by its training mean, plus an input audit and a sanity plot. The checkpoints' validation MSE reproduces the training history (0.08981 meteo fusion, 0.07639 tide-only).
- Outputs: `outputs/diagnostics_2026-11/`. Report and figures: `reports/2026-11-nov-fixes/01_diagnostics.md`.

### Findings
- Southend 72h gap, dense stage 2 minus AR: 0.0053 m (validation) and 0.0125 m (test).
  - A constant bias explains 15% / 22% of it; a calendar-month offset explains 55% / 33%.
  - Most of the rest is a dense stage 2 last-lead artefact: test RMSE 0.2300 / 0.2327 / 0.2378 m at 70 / 71 / 72h, with the bias jumping at 72h. Over 24 to 71h the two models differ by 0.0004 m (validation) and 0.0027 m (test).
- The harmonic residual at Southend drifts: annual mean +0.045 to +0.070 m in 2022 to 2024, against ±0.025 m in training years. At Richmond it is dominated by river-driven swings of ±0.3 to 0.45 m with no trend.
- Richmond stage 3: the most likely cause is overfitting plus early stopping at epoch 1 (validation loss best at epoch 1 of 11 while training loss halves). No covariate is broken: removing any input makes the model worse. Scalers are Richmond's own, the timezone lag is 0 min in every month, and validation has no missing covariates.

### Code change: river leakage fix (commit `46b998b`)
- NRFA daily values are 09:00 to 09:00 GMT water-day means dated by their start, but every anchor on day D used the value dated D. That is up to 33h of future river data.
- New `config.RIVER_AVAILABILITY_LAG_HOURS = 33` (the default) and `LEGACY_RIVER_LAG_HOURS = 0`, which reproduces the old mapping for existing checkpoints. Test: `tests/test_stage3_windowing.py` (5/5); `tests/test_baselines.py` still passes 12/12.
- Nothing retrained. Fed leak-free inputs, the existing Richmond meteo-fusion model loses 0.003 m (10 min) and 0.011 m (72h) on validation.
- Frozen-script note: `train_stage3.py` / `plot_stage3.py` now default to the leak-free mapping. Re-scoring the existing stage 3 checkpoints exactly needs the legacy lag.
- Dense stage 3 tide-only uses no river values. Only its valid-anchor mask moves, by about 1.4 days at the start of the record (training windows only).

### Test disclosure
Diagnostic A computed and displayed **test** RMSE, bias and std for the harmonic baseline, dense stage 2 and the AR residual baseline, at the user's request. All three models' test results were already published. Diagnostic B used validation only; its timezone check uses the whole record but scores no model. No configuration was chosen using these test numbers.

## 2026-11-09: legacy dense models scored on validation (comparison baseline)

### Why
New dense variants are developed on validation, so the existing dense models need validation numbers to compare against.

### Run
- `python models/run_gauge.py --pipeline dense --gauge richmond --gauge southend` (validation only), 2.7 min (Richmond) and 2.8 min (Southend). Console log: `logs/dense_legacy_val_scoring_console.log`.
- Inference only: every checkpoint was loaded and left as-is, and no training ran.
- New files only: `outputs/[richmond/]dense/dense_val_{metrics.csv,vs_sparse_matched.csv,predictions_stage2.npz,predictions_stage3_tideonly.npz}`, plus `logs/<gauge>/dense/run_state_val.json`; `train_dense.log` was appended to. 124 pre-existing legacy files had identical SHA-256 hashes before and after.

### Check
The mean squared validation RMSE over the 73 leads (scaled units) equals each model's best validation loss in `dense_history.json`, to 5 decimal places: Richmond 0.03367 (stage 2) and 0.03389 (stage 3 tide-only); Southend 0.01807 and 0.01808.

### Results
Table and discussion: `reports/2026-11-09-dense-legacy-validation/README.md`. Dense stage 2 validation RMSE at 10min: Richmond 0.0880 m, Southend 0.0458 m.

### Test disclosure
The report shows the **published** test numbers (`dense_metrics.csv`, 2026-10-29) next to the new validation ones, at the user's request. No new test scoring was done.

## 2026-11-09: validation-only scoring, dense variants, explicit CPU threads (branch `paper/nov-fixes`)

### Why
Paper work needs every development decision made on validation. Until now `models/train_dense.py` scored only the test set, at the end of every run, so any new variant would have shown test numbers during development. New variants also need to train from scratch without touching the original dense model, which is the comparison baseline.

### Code
- `models/train_dense.py` scores **validation (2018 to 2020) by default** and writes `dense_val_metrics.csv`, `dense_val_vs_sparse_matched.csv` and `dense_val_predictions_*.npz`. Validation windows are rebuilt raw from the same split that `prepare_stage2` / `prepare_stage3` use. The matched sparse comparison is also on validation; the sparse models are only loaded for scoring, never as initial weights.
- Test is scored only with `--score-test` (env `TIDAL_SCORE_TEST=1`). That writes the original test files (`dense_metrics.csv`, `dense_vs_sparse_matched.csv`, `dense_test_predictions_*.npz`, the Feb 2022 example), in the same format `plot_dense.py` and `eval_baselines.py` read. It is reserved for the final scoring of a finished variant (queue step 3, `dense_v2_skip`).
- Dense variants: `run_gauge.py --variant <name>` (env `TIDAL_DENSE_VARIANT`) writes to `outputs/[<gauge>/]<name>/`, `models/checkpoints/[<gauge>/]<name>/` and `logs/<gauge>/<name>/`, and resumes only from its own `outputs/_resume/` path. The legacy `dense` variant is the default and is score-only: `train_dense.py` refuses to train it if a checkpoint is missing, and refuses `--score-test` for it, so its published test tables are never rewritten.
- `run_gauge.py --pipeline dense`: `plot_dense` runs only with `--score-test` (its figures are test results). Validation and test runs keep separate state files (`run_state_val.json`, `run_state_test.json`), so the legacy `run_state.json` is left alone. `--score-test` is rejected for other pipelines.
- `plot_dense.py` reads the variant's directory and adds `_<variant>` to its figure names.
- `config.NUM_THREADS` (physical cores via psutil, 8 here; override `TIDAL_NUM_THREADS`) is applied by `describe_device()`, so every training script sets `torch.set_num_threads` explicitly.
- Not changed, by decision: the dissertation-era `train_stage1/2/3.py` score test by default and are frozen. If any of them is retrained later (for example stage 3 at Richmond), the validation-scoring change is applied to that script first.

### Verification
- Guards: `TIDAL_SCORE_TEST=1` on the legacy variant exits before loading data; `--score-test` with `--pipeline baselines` is rejected; an invalid variant name (`../dense`) is rejected.
- Smoke run, Richmond: `TIDAL_SMOKE=1 python models/run_gauge.py --pipeline dense --variant dense_smoketest --score-test --gauge richmond`, exit 0 in about 5 min. Both models trained 1 epoch from scratch into the variant's own checkpoint dir. Validation and test files were all written, and the figures went to the variant dir with the variant suffix. Validation anchors: 156,079 dense; matched with sparse 154,351 (stage 2) and 144,415 (stage 3). `CPU threads: 8` was printed.
- Legacy files (`outputs/[richmond/]dense/`, `models/checkpoints/[richmond/]dense/`, `report_images/`) had unchanged modification times afterwards. The smoke directories were then deleted.
- **Test disclosure:** the smoke run computed test metrics for the 1-epoch smoke models, to exercise the `--score-test` path. They were not printed to the session or read, and the files were deleted. No decision used them.

## 2026-10-30: classical baselines (ARIMA, GP, dynamic harmonic, ridge)

### Why
Reviewers asked whether simpler statistical methods were tried before the neural networks. This adds tuned classical forecasters on the dense pipeline's splits, anchors, lead times, masks and metrics, for both gauges, forecasting both the full level and the UTide residual.

### Environment
- Installed into the conda `geospatial` env: `optuna` 5.0.0, which brought `alembic`, `colorlog`, `Mako` and `sqlalchemy`. No existing package was changed.

### Code
- New: `utils/models/baselines.py`, `models/identify_baselines.py`, `models/search_baselines.py`, `models/eval_baselines.py`, `models/plot_baselines.py`, `tests/test_baselines.py`.
- Changed: `config.py` (`BASELINE_*`), `paths.py`, `run_gauge.py` (`baselines` pipeline), `README.md`.

### Decisions
- Five forecasters fill the four planned slots: long AR and SARIMA both report under "ARIMA", then GP, dynamic harmonic and direct ridge.
- All are linear predictors. The gap to the neural models therefore measures what nonlinearity plus UTide-as-input adds.
- AR, GP and ridge are also searched at the neural models' 16h history ("parity"), because the pilot showed history length dominates.
- `load_series(upto=...)` cuts the arrays, so identification sees only train and the search cannot reach test.
- Gaps in history are filled causally by each method's own forecast.

### Findings from identification (train only)
- ACF peaks are at lags 149 and 4174 at both gauges (Richmond residual 4173). That is 2 x M2 and 56 x M2, not the 57 x M2 first assumed, so the search uses the measured lags.
- Southend level: AIC and BIC both pick AR order 8425 (~58 days). Richmond level: BIC 448, AIC 19618.
- Bug caught by the evidence: the first GP fit lost its memory after ~6 days and, on the residual at Q=32, collapsed to a constant kernel. Fixed by fixing line centres, modelling the Hann window's width separately from the kernel's, and bounding the OU rate. Southend level ACF RMSE went from ~0.50 to 0.067.
- Limitation: the Richmond residual kernel stays at ACF RMSE ~0.22 for every Q and either choice of centres. A stationary kernel cannot represent river-modulated tides there.
- GP ACF fit is not monotonic in Q (Richmond level: 0.06 to 0.49), so Whittle fits have local optima. Validation error makes the choice.

### Verification
- `tests/test_baselines.py` covers agreement with statsmodels (ACF, PACF, SARIMAX, Kalman filter), a dense Cholesky GP and sklearn ridge, plus synthetic recovery, gap filling, anchor rule and no-future-leakage checks.
- Smoke run on Richmond: re-scoring reproduced `dense_metrics.csv` to 3e-08 m (dense stage 2) and 7e-08 m (UTide).
- Disclosure: the smoke run used 2 random trials per study and printed test RMSEs. Those were looked at to confirm the code works; no search range or method choice was changed after seeing them, and the code has not been changed for performance since.

### Search rounds
- Round 1 left 31 winners per gauge on a range edge. A pipeline design error then ran test evaluation before those were reviewed. Those test tables were not opened and have been overwritten. Test evaluation is now the separate `baselines_eval` pipeline.
- Round 2 (`*_x` studies, seeded with the round 1 winner): history to 1 year, GP Q and dynamic harmonic K to 128, wider SARIMA orders and continuous ranges. Optuna cannot widen categorical choices inside a study, so these are new studies; round 1 CSVs are kept.
- Round 3 (`*_y`, Richmond ridge only): 2-year history, because Richmond level still gained at 1 year. It was worse (0.332 m vs 0.300 m validation), so 1 year stands.
- Left on edges: binary d/D terms, the 16h parity cap, ridge spacing and lambda (no numerical effect), and some noisy Richmond GP and dynamic harmonic settings. SARIMA hit its 2h cap at 16-21 trials in each round 2 study, target 80.

### Results (test RMSE, m)
- Southend: long AR residual beats Dense stage 2 at 71 of 73 leads, significantly at 57, none significantly worse. 72h: 0.225 vs 0.238.
- Richmond: Dense stage 2 wins beyond about 1h, significantly at 71 leads. 72h: 0.332 vs 0.352.
- Held to 16h history, no classical model beats Dense stage 2 beyond 1h (Southend) or 10min (Richmond).
- No classical model compounds like stage 1. Closeness rule, set before reading test scores: within 10% of the best NN at 72h and on average. 7 of 10 pass at Southend, 2 of 10 at Richmond.
- Full write-up: Claude Doc "Classical baselines vs neural networks", https://claude.ai/artifact/Rw37PWinNtXkBrLxdDsCMN

### To run
```powershell
& C:\Users\alfat\anaconda3\envs\geospatial\python.exe models/run_gauge.py --pipeline baselines --gauge southend --gauge richmond
# review outputs[/<gauge>]/baselines/search/edge_winners.csv and search_shortfall.csv, then:
& C:\Users\alfat\anaconda3\envs\geospatial\python.exe models/run_gauge.py --pipeline baselines_eval --gauge southend --gauge richmond
```
Logs: `logs/<gauge>/baselines/` and `logs/<gauge>/baselines_eval/`. Add `--redo` to re-run finished steps. Round 3: `TIDAL_GAUGE=richmond TIDAL_BASELINE_METHODS=ridge TIDAL_BASELINE_ROUND=_y python models/search_baselines.py`.

---

## 2026-10-28 — Dense (continuous 0–72h) pipeline: code, verification, overnight launch

**Why.** The supervisor asked for continuous 72h forecasts. The design is in the Claude Doc "Dense 0–72h forecasting: current vs proposed decoder", section 9. The stage 2/3 models stay direct (non-autoregressive), and only the list of lead times changes: from 7 or 30 sparse lead times to 10min plus every hour to 72h (73). The output head is unchanged: it is already applied once per decoder step. Built as a separate pipeline, so no existing result or checkpoint is touched.

**Code.**
- `utils/models/config.py`:
  - `DENSE_HORIZONS` (73 lead times).
  - `DENSE_TAIL_DAYS`: an optional daily tail for stage 3, empty by default. Days 4–30 can be added, with `DENSE_TAIL_LOSS_SHARE` = 0.5 of the loss.
  - `dense_stage3_decoder()`: lead point + targets + 3 padding points, giving 77 decoder points with no tail.
  - `dense_horizon_weights()`.
- `utils/paths.py`: `dense_output_dir` (`outputs/[<gauge>/]dense/`) and `dense_checkpoint_dir` (`models/checkpoints/[<gauge>/]dense/`).
- `models/train_dense.py` (new): trains dense `TidalSeq2Seq` (stage 2 settings: lr 1e-3, batch 512) and dense `TidalSeq2SeqTideOnly`, whose lr / weight decay / batch come from that gauge's sparse tide-only checkpoint.
  - Metrics at all 73 lead times plus UTide.
  - A matched-anchor comparison with the existing sparse models: the anchors both test sets share, at 10min / 1h / 6h / 24h / 48h / 72h.
  - All test predictions, and one 72h forecast issued 2022-02-17 12:00 (Storm Eunice).
  - `TIDAL_SMOKE=1` runs a 1-epoch mechanics check.
- `models/plot_dense.py` (new): two PNGs per gauge.
  - `stages_rmse_at_stage1_leads`: every model from stage 1 to dense, at stage 1's 5 lead times. Colour = stage family, from a palette that passed the dataviz colour checks. Filled = direct or dense.
  - `dense_72h_forecast`: one continuous 72h forecast against the observations.
- `models/run_gauge.py`: `--pipeline {main,dense}`, repeatable `--gauge`, and `--redo`. With no flags, the main pipeline runs exactly as before. Dense logs and state go to `logs/<gauge>/dense/`.
- `utils/plot_config.py`: `save_fig` now writes PNG (`FIG_FORMAT = 'png'`, 300 dpi) instead of PDF, at the user's request.

**Verification (smoke runs, both gauges, then deleted).**
- Tensor shapes: (N, 73) targets; stage 2 decoder input (N, 73); stage 3 decoder input (N, 77).
- Parameter counts are identical to the sparse models (80,705 each).
- The rebuilt sparse windows reproduce the published sparse RMSEs exactly: Richmond stage 2 at 10min 0.0329 m, stage 3 tide-only 0.0882 m; Southend stage 2 0.01726 m.
- Feb-2022 issue time is valid at both gauges.
- Tail option: 100 outputs, 104 decoder points, and the tail carries exactly 50% of the loss.
- Timing benchmark on CPU, one batch of 512: dense 245 ms vs sparse 93 ms (stage 2) and 112 ms (stage 3). That is about 2 min per training epoch per dense model, so up to about 10h if all four models run the full 60 epochs.

**Run.** `python models/run_gauge.py --pipeline dense --gauge richmond --gauge southend`, launched 2026-10-28 18:53. Console log: `logs/dense_run_console.log`.

**Run outcome (2026-10-29).** All 4 steps passed on the first attempt, in 5.5 h total.

| Gauge | Model | Stopped at epoch | Time |
|---|---|---|---|
| Richmond | stage 2 | 33 | 80 min |
| Richmond | stage 3 tide-only | 27 | 60 min |
| Southend | stage 2 | 45 | 89 min |
| Southend | stage 3 tide-only | 48 | 100 min |

**Outputs.**
- Weights: `models/checkpoints/[richmond/]dense/`.
- Tables: `outputs/[richmond/]dense/`.
- Figures: `report_images/stages_rmse_at_stage1_leads[_richmond].png` and `report_images/dense_72h_forecast[_richmond].png`.

**Headline results** (test RMSE, m, matched anchors):
- Dense stage 2 is within ±2% of sparse stage 2 from 6h to 72h at both gauges. At 72h: Richmond 0.332 vs 0.336, Southend 0.236 vs 0.228.
- It is worse at short leads. At 10min: Richmond 0.092 vs 0.033, Southend 0.049 vs 0.017.
- Dense stage 3 tide-only beats sparse stage 3 tide-only at Richmond by 6–20% from 1h to 72h.
- The 10-min error tracks the 10-min loss share (1/7, 1/30, 1/73), so the short-lead penalty comes from equal per-lead loss weighting.

Write-ups (Claude Docs):
- results: https://claude.ai/artifact/A44jSF5fPhpicfQQHuk8gG
- methodology shortcomings: https://claude.ai/artifact/MRsuBTKSkPFVstKioFcLxm
- newcomer explainer: https://claude.ai/artifact/QZWuuNgiCccFhaPNrjEFtW

## 2026-10-18 — UTide option search (constituent set × fitting method), both gauges

**Why.** The supervisor reported a better UTide fit for Richmond than our ~0.56 m test RMSE. All earlier work used `utide.solve` defaults: `constit='auto'` (68 constituents) and `method='ols'`. This step tests the alternatives on their own, before any model is retrained, so their effect on the harmonic baseline is known first.

**Code.**
- `utils/models/datasets.py`: `fit_utide` gains `constit`, `method` and `robust_kw`, passed straight to `utide.solve`. The defaults reproduce the old fit exactly, so every existing checkpoint and result is unchanged.
- `src/utide_option_search.py` (new, standalone):
  - Grid: `constit` ∈ {auto, auto + 22 shallow-water constituents} × `method` ∈ {ols, robust-cauchy, robust-fair, robust-bisquare}, fitted at stride 6.
  - The extra 22 are UTide-table constituents outside the auto set: N4 3MS4 KN4 SL4, MNO5 2MO5 3MP5 MNK5 2MP5 MSK5 3KM5, 2NM6 MSN6 MKN6 NSK6 S6, M7, 3MN8 3MS8 3MK8, 4MK9, M10.
  - Fits on train (2004–17), selects on validation RMSE (2018–20), and reports test (2021–24). Flagged points are excluded throughout.
  - The winner and the previous configuration are refit at stride 1.
  - Flags: `--smoke` (2-year fit) and `--plot-only` (redraw figures from the CSVs).
- `robust-huber` was dropped: `huber()` in utide 0.3.1 calls `max()` on an array and always raises. `fair` is the closest gentle alternative.

**How it was run.**
- `python src/utide_option_search.py`: about 15 min for both gauges, log in `logs/utide_search/`.
- Smoke test first (`--smoke`). In the 2-year fit, the trend term came out at −29 cm/yr, and extrapolating it gave a 2.15 m test RMSE at Richmond. For the real 14-year fit the trend is 0.7 cm/yr, and turning it off changes per-year errors by only a few cm, so `trend` was not added to the grid.

**Results** (test RMSE, m, stride 1; full table in `outputs/<gauge>/utide_search/utide_option_search.csv`):

| | previous (auto / ols) | validation winner | change |
|---|---|---|---|
| Richmond | 0.5587 | 0.5519 (auto+SW / ols) | −1.2% |
| Southend | 0.2468 | 0.2436 (auto+SW / robust-fair) | −1.3% |

- The stride-1 baseline reproduces the earlier UTide numbers: Richmond 0.559 here, against 0.548–0.561 in the stage 2/3 tables; Southend 0.247, against ~0.245. The small gaps come from those tables scoring windowed horizon targets rather than every row.
- **Robust fitting makes Richmond worse** (test 0.583–0.601). River-raised low waters are frequent and systematic, not outliers. Down-weighting them fits the low-flow tide, which then misses every high-flow period. At Southend, robust helps very slightly (about 2 mm).
- **Shallow-water constituents help both gauges**, a little. Richmond's largest new one is 3MS4 at 9.7 cm. M4 (43 cm) and MS4 (24 cm) were already in the auto set, so the main shallow-water signal was already being fitted.
- **Conclusion.** Changing `constit` and `method` gains about 1%. It does not explain a much better supervisor fit. That more likely comes from how the fit is scored: in-sample (our train RMSE is 0.52 m at Richmond), a shorter fit window, or a subset such as high waters or low-flow periods. Per-year RMSE at Richmond, from the 14-year auto/ols fit, ranges from 0.38 to 0.73 m across 2004–2024 (test years 0.44–0.72 m) with river conditions.
- Winners are in `outputs/<gauge>/utide_search/utide_choice.json`. Nothing in the modelling pipeline uses them.
- **Decision (user, 2026-10-18): UTide stays unchanged (auto / ols) for both gauges.** The gain is about 1%, and keeping the fit means existing results and the stage 3 checkpoints stay valid. The search is kept as a reported sensitivity check.

**Figures.** `report_images/utide_option_search_{richmond,southend}.png`: 300-dpi PNG dot plots of validation and test RMSE per option set. From now on, figures are PNG rather than PDF, at the user's request.

## 2026-10-13 — Richmond overnight run: results

**The run.** `models/run_gauge.py` (Richmond, 72h cap) ran from 20:31 to 23:38 on 12 Oct (3 h 7 min). All 6 steps passed on the first attempt, with no retries.

| Step | Time |
|---|---|
| train_stage2 | 87 min |
| train_stage3 | 41 min |
| plot_stage3 (incl. SHAP) | 11 min |
| train_stage1 | 47 min |

**Outputs.**
- 17 figures: `report_images/*_richmond*.pdf`.
- Tables and training histories: `outputs/richmond/`.
- Weights: `models/checkpoints/richmond/` (8 files, 1.6 MB). Tracked in git, like Southend's, at the user's request on 2026-10-13.

**Findings** (test RMSE, m; full write-up in the results report):

| | Richmond | Southend |
|---|---|---|
| Stage 2, 10min | 0.033 | 0.017 |
| Stage 2, 24h | 0.256 | 0.223 |
| Stage 2, 72h | 0.336 | 0.228 |
| UTide standalone | 0.561 | 0.245 |
| Stage 2 improvement over UTide at 72h | 40% | 6% |

- Stage 2 is the best model at Richmond at every horizon.
- Stage 3 is worse than stage 2 at Richmond.
  - The meteo-fusion model overfits: best validation loss at epoch 1, early stopped at 11. Its 10-min RMSE is 0.312.
  - The tide-only model is likely held back by its 30-horizon loss. Horizons up to 72h carry only 7% of its summed MSE at Richmond, against 14% at Southend. Its 10-min RMSE is 0.088.
- SHAP: rainfall and discharge carry 14–17% of attribution at Richmond, against 2–5% at Southend.
- Stage 1: the MLP's recursive rollout diverges at Richmond (34 m RMSE at 48h).
- Every stage 1 and stage 2 Richmond model used the full 60 epochs. Gains near the end were small, under 1% over the last 10 epochs.

**Suggested follow-ups, not done:**
- A stage-3 retrain capped at 72h, to test the loss-share explanation.
- Richmond-specific stage-3 hyperparameters.
- A log y-axis for the stage-1 rollout figure.
- Fixing the overlapping date labels on the stage-1 prediction figure.

---

## 2026-10-12 — Second gauge (Richmond Lock), 72h report figures, laptop-safe runs

### Why

- The revised report covers horizons up to 72h only.
- The same pipeline is to be run on a second gauge, Richmond Lock.
- Training is to run on a CPU-only laptop (Intel Core Ultra 200V, 8 cores, 32 GB) instead of Colab.

### Decisions (agreed before implementation)

| Question | Decision | Reason |
|---|---|---|
| Change the architecture for 72h? | No — same models, plot horizons up to 72h only | Stages 2/3 are direct multi-horizon predictors: each horizon is its own output, so dropping the longer ones at plot time *is* an evaluation to 72h. Keeps Richmond comparable with Southend. |
| Richmond meteo (stage 3) | Reuse Southend's ERA5 pressure/wind | Only that point was downloaded; pressure is synoptic-scale and Thames surge is driven mostly from the North Sea. Kingston rainfall/discharge are reused as-is (and matter more at Richmond). |
| Richmond stage-3 hyperparameters | Reuse Southend's search winners (multi: lr 1.78e-3, wd 1e-5, bs 512; tide-only: lr 2.02e-3, wd 0, bs 512) | Saves ~4.5 h CPU; the gauge is then the only thing that changes. |
| Richmond scope | Headline models only | The tide-input ablation and the reduced-training retrains stay Southend-only. |
| Hourly-resolution experiment | Not run — explanation only | Helps stage 1's recursive rollout (fewer compounding steps); much less relevant to the direct multi-horizon models (stages 2/3). |
| Git | Track code and figures; never large data | The Richmond CSV is gitignored and regenerated by its cleaning script. |

### Environment

- Runs use the conda env `geospatial` (`C:\Users\alfat\anaconda3\envs\geospatial\python.exe`, Python 3.12.4).
- The env had everything except PyTorch. Installed `torch==2.13.0+cpu` from the PyTorch CPU index; the install was purely additive.
- `xarray` (already broken against NumPy 2), `openpyxl` and `netCDF4` were left alone: only the `src/` download/ingestion scripts need them.

### Measurements that shaped the set-up

| Measurement | Result |
|---|---|
| Peak RAM, data preparation at full UTide resolution | stage 1: 8.7 GB; stages 2 and 3: 6.7 GB |
| UTide fit memory | ~130 bytes per row per constituent; 68 constituents |
| Hourly UTide fit vs 10-minute fit | Same constituents, ~1 mm RMS reconstruction difference, unchanged test RMSE. Kept as an option (`utide_fit_stride`); not used, since full resolution fits in RAM. |
| CPU time per batch of 512 (fwd+bwd) | flat LSTM 60 ms, stage-2 seq2seq 90 ms, stage-3 multi 274 ms, stage-3 tide-only 121 ms |
| Estimated time, headline Richmond models | ~6–7 h in total |

The crash seen earlier when running on the laptop could not be reproduced. With the current scripts, every data-preparation step stays below 9 GB.

### Richmond data (`src/clean_richmond.py` → `Data/richmond_lock_data.csv`)

**Input.** `3_Cleaned/01_RICHMOND.csv`, already in ODN: ODN = chart-datum reading − 0.61 m, per the PLA gauge table. The offset was verified on all 1,076,814 rows. The output keeps both columns with that exact offset.

**Raw state.** 2.52% of steps are missing, spread over 1,265 gaps, 1,153 of them an hour or shorter.

**Faults found and patched.** Patched values are flagged, so they are excluded from every loss and metric.

| Fault | Readings | Detection |
|---|---|---|
| Spikes / chatter | 696 | UTide residual > 0.3 m from its 70-min rolling median, 2 passes |
| Stuck sensor | 1,995 | ≥ 6 identical readings while the tide moves > 0.1 m; mostly at −0.61 m ODN, i.e. 0 on the gauge's chart datum |
| Datum shift (19–23 Aug 2009, ~+3 m) | 626 | UTide residual's 25 h rolling minimum > 1.0 m. A genuine flood never triggers this: the Feb 2014 flood was checked. |
| Out of range | 138 | outside −2.5 to 6.0 m ODN |

**Patching.**
- Short runs (≤ 6 h) of faulty or missing data are filled with UTide plus the residual interpolated across the run. Pure UTide is not used because river flow moves Richmond's low waters by up to ~3 m, so UTide's RMSE there is ~0.53 m against ~0.24 m at Southend. A pure-UTide patch would put ~1 m steps into the record.
- Anything longer than 6 h is left as a real gap.

**Result.**
- 1,078,071 rows; 0.44% flagged; 37 gaps remain.
- Share of possible windows usable (before → after): stage 1 91.8% → 97.3%; stage 2 67.3% → 94.1%; stage 3 39.0% → 84.3%.
- Before/after plots of the 40 largest episodes are in `outputs/richmond/cleaning/`.

### Code changes

**Gauge selection**
- `utils/models/config.py`:
  - New `GAUGES` registry: label, CSV, latitude, UTide fit stride, meteo site.
  - New `gauge_info()`.
  - New `from_env()`, so a script's CONFIG value can be overridden by `TIDAL_GAUGE` / `TIDAL_MAX_HORIZON_H`.
  - New `horizons_up_to()` for the horizon cap.
- `utils/paths.py`:
  - New `RICHMOND_CSV`.
  - New `stage_output_dir(stage, gauge)` and `checkpoint_dir(gauge)`. Southend keeps its original paths; other gauges use `outputs/<gauge>/…` and `models/checkpoints/<gauge>/`.

**Data preparation (`utils/models/datasets.py`)**
- `prepare_stage1/2/3` take `gauge=` and pass that gauge's latitude to UTide. It was previously hard-coded to Southend.
- `fit_utide(stride=)` can fit on every n-th row.
- `reconstruct_utide` reconstructs in 200k-step chunks. The result is identical; this was verified.
- `load_meteo(site=)` reads the chosen ERA5 point.

**Training (`utils/models/training.py`)**
- `train_model(resume_path=)` saves the full training state after every epoch to `outputs/_resume/…` and resumes from it. A resumed run was verified bit-identical to an uninterrupted one.
- `partial_path()` gives the resume-file location for a checkpoint.
- `save_checkpoint()` no longer rewrites a checkpoint the run only loaded. Before this, re-running a train script rewrote the tracked Southend checkpoints. The weights were unchanged, but the bytes and keys were not; those three files were restored from git.

**Train scripts (`models/train_stage{1,2,3}.py`)**
- All three have `GAUGE` in CONFIG and resumable training.
- Stage 3 has `REUSE_HPARAMS_FROM`.
- The robustness retrains run only for Southend.

**Plot scripts (`models/plot_stage{1,2,3}.py`)**
- `GAUGE` and `MAX_HORIZON_H` (default 72) in CONFIG.
- Figure filenames get `_<gauge>` (non-Southend) and `_72h` suffixes, so the original figures are never overwritten.
- Stage 3 names the horizons on the axis when capped. SHAP horizons become 10min/1h/6h/24h/48h/72h under the cap, and its tables are suffixed the same way.
- Stage 1 gained the `initial_rollout_rmse_vs_horizon` figure and no longer crashes when a table is missing.

**"Better" arrows (`utils/plot_config.py`)**
- `mark_better()` draws a small grey arrow with "better" just outside the right spine, pointing down for every RMSE, MAE and loss axis.
- Horizon, years, epoch and SHAP axes get none: neither direction is better on those.
- `set_name_suffix()` and `figure_suffix()` handle the filename suffixes.

**New scripts**
- `src/clean_richmond.py`: cleans the Richmond record, as described above.
- `models/run_gauge.py`: runs a whole gauge unattended. In order:
  1. cleaning, only if the cleaned CSV is missing;
  2. train and plot stage 2;
  3. train and plot stage 3;
  4. train and plot stage 1.

  How it behaves:
  - each script runs as its own process, with the gauge and horizon cap passed through the environment;
  - a failed step is retried up to 3 times, resuming from its last epoch;
  - each step's output is logged to `logs/<gauge>/<step>.log`;
  - completed steps are recorded in `logs/<gauge>/run_state.json` and skipped on re-launch;
  - Windows is kept from idle-sleeping while it runs.

  Tested with a deliberately failing step (it retried and passed) and on a real plot script (gauge and horizon cap reached it; re-launching skipped completed steps).

**Docs and ignores**
- README: data and running sections updated.
- `.gitignore`: `logs/` and `Data/richmond_lock_data.csv`.

### Runs carried out

**Southend, redrawn to 72h. No training; existing checkpoints and tables were used.**
- Stage 2: every figure, written as `report_images/stage2_*_72h.pdf`.
- Stage 3: RMSE figures, plus SHAP recomputed for the six horizons up to 72h (~12 min on CPU), written as `stage3_*_72h.pdf`. The tables are `outputs/stage3_outputs/stage3_shap_*_72h.csv`.
- Stage 1: `train_stage1.py` re-run to rebuild the missing `outputs/initial_outputs/` tables (models loaded, not retrained), then all stage-1 figures, including the new rollout figure.
- Loss curves cannot be redrawn for Southend: the `stage*_history.json` files from the Colab runs are not on this machine.
- Caveat: the local stage-3 reduced-training table, from the original notebook run, has the **meteo-fusion** model in its "14yr (full)" row, while the script version writes the tide-only model there.

**Richmond: prepared, not yet trained.**
- `Data/richmond_lock_data.csv` was built and checked:
  - window counts match the predicted yields;
  - every tensor is finite;
  - a 1-epoch smoke run on 2% of the data went end to end.
- UTide alone scores 0.56 m RMSE at Richmond on the test set.

**Richmond: rehearsal before the overnight run.**
- Each of the six steps was run once against scratch folders: 1 epoch, every 30th training window, a 20-sample SHAP.
- All six exited 0 and drew all 17 Richmond figures. Stage 3 picked up Southend's hyperparameters.
- The only skips were the Southend-only ablation figures, which is expected.
- The real `models/checkpoints/richmond/`, `outputs/_resume/` and `logs/richmond/` were confirmed empty afterwards, so the real run starts from scratch.

### To run Richmond

1. Plug the laptop in and leave the lid open.
2. From a terminal in the repo root:
   ```powershell
   & C:\Users\alfat\anaconda3\envs\geospatial\python.exe models/run_gauge.py
   ```
   It takes ~6–7 h; progress is in `logs/richmond/`.
3. If the machine restarts, run the same command again. It skips finished steps and resumes the interrupted one.
4. Outputs:
   - figures: `report_images/*_richmond_72h.pdf` (stage 1: `*_richmond.pdf`, since it has no horizon cap);
   - tables: `outputs/richmond/`;
   - weights: `models/checkpoints/richmond/`.
