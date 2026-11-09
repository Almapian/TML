# CLAUDE.md: Tidal Thames forecasting (TML)

Read this before doing anything in this repo. If something here conflicts with what you find in the code, trust the code, then flag the discrepancy to Al and suggest an update to this file.

## What this project is

An MSc dissertation (UCL CEGE, submitted September 2026), now being turned into a paper with supervisor Dr Cassandra Nanlal. It forecasts 10-minute total water level on the tidal Thames and compares neural sequence models against harmonic analysis (UTide).

- **Gauges in scope:** Southend Pier (estuary mouth, near-linear tide) and Richmond Lock (tidal limit, strong river-flow influence, nonlinear tide). Other Thames gauges exist in the data but are not used yet.
- **Paper framing (working):** a small neural model on top of harmonic analysis, as a cheaper alternative to large models. It is compared against harmonic analysis alone and against well-tuned classical linear forecasters (AR, ridge, GP, SARIMA, dynamic harmonic regression).
- **Operational target:** the Port of London cares about roughly a 2-day forecast. The current horizon is 0 to 72h.
- **Examiner/supervisor background:** hydrography and geomatics, with limited ML. Written explanations should assume tidal knowledge and explain ML choices.

## Model stages

| Stage | What it is | Notes |
|---|---|---|
| 1 | Single-step MLP/RNN/LSTM, recursive rollout | Historical only. Do not extend. `models/train_stage1.py` |
| 2 (sparse) | LSTM encoder, attention, BiLSTM decoder, one output per decoder step (`TidalSeq2Seq`) | 7 leads: 10min, 1h, 6h, 24h, 48h, 72h, 168h (`STAGE2_HORIZONS`). Inputs: 16h gauge history (`LOOKBACK = 96`) + UTide reconstruction at target times. `models/train_stage2.py` |
| 3 (sparse) | Stage 2 + meteorological/hydrological branches (`TidalSeq2SeqMulti`), plus a tide-only variant (`TidalSeq2SeqTideOnly`) | 30 leads: the 7 above plus daily 8 to 30 days (`STAGE3_HORIZONS`). ERA5 pressure, u10, v10 (the Southend Pier ERA5 point is used for both gauges); Kingston catchment rainfall and discharge; availability flags; SHAP in `models/plot_stage3.py`. `models/train_stage3.py` |
| Dense | Stage 2 and stage 3 **tide-only** retargeted to 73 leads (10 min, then hourly to 72h; `DENSE_HORIZONS`). No meteo branches. | `models/train_dense.py`. Branch `dense-predictor`. Known issue: poor short-lead accuracy (CHANGELOG 2026-10-28 attributes it to equal per-lead loss weighting) |
| Baselines | Classical linear forecasters on the dense pipeline's anchors, leads, masks and RMSE code | `utils/models/baselines.py`, `models/{identify,search,eval,plot}_baselines.py` |

The models are **direct** multi-horizon predictors, not recursive. Model classes are in `utils/models/architectures.py`; `utils/models/layers.py` holds only the shared `Encoder` and `BatchedAttention`.

## Data and evaluation rules (non-negotiable)

- Splits are chronological: **train before 2018, validation 2018 to 2020, test 2021 onwards** (`TRAIN_END = '2018-01-01'`, `VAL_END = '2021-01-01'` in config; `split_chronological` in `utils/models/datasets.py`). Data runs 2004 to 2024.
- UTide is fitted on **train only**. Never refit on validation or test.
- **Make all development decisions on validation.** Look at test only when producing final numbers for a finished configuration. If you see test numbers during development, record it in the CHANGELOG. `models/train_dense.py` scores **validation only** by default (`dense_val_*.csv`). Test is scored only with `--score-test` (env `TIDAL_SCORE_TEST=1`), which is reserved for the final scoring of a finished variant. `baselines_eval` is the baselines' test step. The dissertation-era stage 1 to 3 training scripts (`models/train_stage{1,2,3}.py`) score test by default and are **frozen**. If any of them is retrained later (for example stage 3 at Richmond), apply the validation-scoring change to that script first.
- Patched or flagged values are masked out of scoring. `is_flagged` = chatter | stuck | imputed (`load_gauge_df`). Use `masked_rmse_mae` (scoring) and `masked_mse` (loss) in `utils/models/training.py` rather than writing new scoring.
- "RMSE at a given lead" means the average over every anchor in the period, not a single forecast.
- No leakage: every input must be computable from data at or before the anchor time. Future UTide at target times is allowed because it is a deterministic astronomical prediction. Future weather is not allowed.

## Branches and git

- `master` (the default branch): the code as of dissertation submission, tagged `dissertation-submitted` (4428b54). Do not commit to it directly.
- `dense-predictor`: dense model plus the classical baselines (commit `ba70745`, tagged `baselines-audit-2026-11`, holds the baselines audit).
- Work branches come off `dense-predictor` and are named `paper/<topic>`, for example `paper/nov-fixes`. Open a **draft PR into `dense-predictor`**, not into master.
- The remote is SSH (`git@github.com:Almapian/TML.git`) via Windows OpenSSH (`core.sshCommand`), with the key held by the Windows `ssh-agent` service, so pushes need no prompt.
- Use one logical change per commit, and write messages that say what changed and why.
- Never force-push and never rewrite published history.
- `outputs/` is gitignored. Results meant for humans go in `reports/<date>-<topic>/` (markdown and PNGs only; no large CSVs over ~1 MB), and that folder **is** committed. (`reports/` is new; dissertation-era figures are in the tracked `report_images/`.)
- Model weights under `models/checkpoints/` **are** tracked in git.

## Experiment discipline

- **Make one architectural change per experiment**, so results can be attributed. If two changes are needed, train them as separate variants.
- **Never overwrite existing outputs or checkpoints.** Each new variant gets its own directory, following the existing per-gauge layout (Southend has no gauge folder, other gauges do): for example `outputs/[richmond/]dense_v2_skip/` and `models/checkpoints/[richmond/]dense_v2_skip/`, next to the current `dense/`. Pick it with `run_gauge.py --variant <name>` (env `TIDAL_DENSE_VARIANT`); logs go to `logs/<gauge>/<variant>/`. A variant trains from scratch and only ever loads or resumes from its own directories. The legacy `dense` variant is the comparison baseline: it is score-only, `train_dense.py` refuses to retrain it, and `--score-test` is refused for it so its published test tables stay as they are. Do not delete, move or retrain the old models.
- Every experiment gets a `CHANGELOG.md` entry (date, what, why, where the outputs are, and whether test was looked at).
- Keep configs in `utils/models/config.py`; do not hard-code them in scripts.

## Compute constraints (important)

Al's machine is an ASUS Zenbook 14 with an Intel Core Ultra 7 258V (Lunar Lake, 8 physical cores, no hyperthreading), integrated graphics and 32 GB of shared RAM (31.5 GB visible). There is no CUDA.

- Python environment: conda `geospatial` (`C:\Users\alfat\anaconda3\envs\geospatial\python.exe`, Python 3.12.4, torch 2.13.0+cpu, psutil 7.0.0).
- Train on CPU. Do not try to set up the Intel GPU (XPU) unless Al asks.
- **Never materialise all training windows in memory.** Index windows lazily in the `Dataset`. Before any long run, estimate peak RAM from a short run (use `psutil`) and stop if the projection exceeds about 24 GB. **Current state:** there is no `Dataset` class. `utils/models/windowing.py` and `datasets.py` build every window of every split up front as NumPy arrays, then copy them to tensors; `training.batch_iter` indexes those tensors. This is fine at the 16h lookback (dataset prep peaked at 6.7 GB) but memory grows linearly with lookback, so any longer-input variant needs lazy windowing first.
- Run **one training job at a time.**
- Long jobs must be **detached from the Claude Code session**, logged to a file, checkpointed every epoch and resumable. They should keep the machine awake while running (on Windows, `SetThreadExecutionState` via `ctypes`). Overnight runs are fine; crashes are not. `models/run_gauge.py` already does the logging, retries, keep-awake and resume (per-epoch state in `outputs/_resume/`).
- Set `torch.set_num_threads` explicitly to the number of physical cores (8). This is `config.NUM_THREADS` (override with `TIDAL_NUM_THREADS`), applied by `config.describe_device()`, which every training script calls.

## Writing conventions (reports, docs, commit messages)

- Use British English and **no em dashes**.
- Keep figure captions short and factual, and put interpretation in a separate discussion section.
- Never claim a result without the number and the file it came from.
- Say "harmonic analysis" or "harmonic baseline" in prose; UTide is the implementation.

## Useful paths

- `models/run_gauge.py`: pipeline entry point. `--pipeline {main,dense,baselines,baselines_eval}`, repeatable `--gauge`, `--redo`, and for dense `--variant` and `--score-test`. Default gauge is `richmond` here, but `southend` when a train script is run directly.
  - Dense variant (validation): `python models/run_gauge.py --pipeline dense --variant dense_v2_skip --gauge richmond --gauge southend`. Final test scoring: add `--score-test`.
- `utils/models/config.py`: all configuration
- `utils/models/datasets.py`, `utils/models/windowing.py`: loading, splitting, UTide fit, windowing, scaling
- `utils/models/training.py`: training loop, masked loss and RMSE, checkpoints and resume
- `utils/models/architectures.py`: model classes; `utils/models/layers.py`: encoder and attention layers
- `utils/paths.py`: every output and checkpoint directory
- `tests/test_baselines.py`: baseline correctness and no-leakage tests
- `CHANGELOG.md`: experiment history and disclosures
