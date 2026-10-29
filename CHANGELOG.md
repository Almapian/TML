# Changelog and run record

Every change to the pipeline, and every step taken to set up or carry out a run, is
recorded here, newest first. Entries say what changed, why, and how it was checked.

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
