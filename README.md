# Tidal Analysis and Prediction — Southend Pier

MSc dissertation code: forecasting water level at Southend Pier on the Thames Estuary,
comparing classical harmonic analysis (UTide) with neural sequence models, in three stages
of increasing complexity.

- **Stage 1** — single-step baselines: persistence, UTide, MLP / RNN / LSTM.
- **Stage 2** — direct multi-horizon forecasting (10 min → 1 week) with an LSTM encoder,
  attention, and a BiLSTM decoder fed the known-in-advance UTide reconstruction.
- **Stage 3** — adds meteorological covariates (pressure, wind, rainfall, river discharge)
  as separate encoder branches, extends the horizon to 30 days, and attributes the
  predictions with SHAP.

Gauge data: Southend Pier, 10-minute readings, 2004–2024 (1,103,956 rows), cleaned and
patched for chatter and stuck-sensor faults. Splits are chronological throughout: train
2004–2017, validation 2018–2020, test 2021–2024.

---

## Repository layout

| Path | What it holds |
|---|---|
| `Data/` | `southend_pier_data.csv` — the cleaned gauge record every stage reads |
| `models/` | `train_stage{1,2,3}.py` (fit models, write tables) and `plot_stage{1,2,3}.py` (draw figures) |
| `models/checkpoints/` | Trained model weights (`.pt`), so results can be reproduced without retraining |
| `models/notebooks/` | The original development notebooks for each stage |
| `EDA/` | Exploratory analysis and the data-cleaning notebooks (see below) |
| `src/` | Data acquisition: build the gauge dataset from raw files, download the meteo data |
| `utils/` | Shared code: paths, plot styling, model architectures, training loop, dataset prep |
| `report_images/` | Figures used in the dissertation (PDF, written by the plot scripts) |
| `outputs/` | Scratch: metrics tables, training curves, diagnostic plots (not in the repo) |

### EDA and cleaning notebooks

| Notebook | Purpose |
|---|---|
| `EDA/gauge_ts.ipynb` | Temporal EDA: stationarity, ACF/PACF, periodogram, seasonal decomposition, completeness |
| `EDA/meteo_eda.ipynb` | Meteo covariates vs. the tidal residual: lagged correlations, inverse barometer, storm case study |
| `EDA/utide_test.ipynb` | The cleaning pipeline: chatter and stuck-sensor detection, then flag → null → refit → patch |
| `EDA/tide_gauge_vis.R` | Map of the Thames tide gauges (R; helpers in `utils/mapper.R`) |

---

## Setup

Python **3.12** (developed on 3.12.7).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r utils/requirements.txt
```

`utils/requirements.txt` covers everything: numpy, pandas, scipy, matplotlib, PyTorch,
UTide (harmonic analysis), SHAP, statsmodels (EDA), and the ingestion extras
(openpyxl, requests, cdsapi, xarray, netCDF4).

**GPU is optional.** `pip install torch` gives the CPU build, which is enough to run every
plotting script and to load the supplied checkpoints. For training, install the CUDA build
from [pytorch.org](https://pytorch.org). The models in this repository were trained on
Google Colab, on A100 and T4 high-RAM runtimes.

---

## Data

Only the gauge record is published here.

- **`Data/southend_pier_data.csv`** (93 MB, in the repo) — 10-minute water level, already
  datum-corrected to ODN and patched for chatter and stuck-sensor faults, with the
  `is_chatter_flagged` / `is_stuck_flagged` / `is_imputed` columns marking every value that
  was corrected. Stages 1 and 2 need nothing else.
- **`Data/richmond_lock_data.csv`** — the second gauge, Richmond Lock (tidal limit, ~8 km
  below Kingston), in the same schema. Written by `python src/clean_richmond.py` from
  `3_Cleaned/01_RICHMOND.csv`: spikes/chatter, stuck-sensor runs, a 5-day datum shift
  (Aug 2009) and out-of-range readings are flagged and patched, gaps of 6 h or less are
  filled, and anything longer is left as a real gap. Every patched value is flagged. Not
  in the repo (size) — `models/run_gauge.py` regenerates it when it is missing.
- **`Data/meteo/`** (not in the repo, for size) — the four meteorological series stage 3
  uses: ERA5 pressure and 10 m wind at Southend, plus NRFA rainfall and river discharge for
  the Thames at Kingston. Regenerate with `python src/download_meteo_data.py` (needs a free
  Copernicus CDS account; the NRFA half needs no credentials), or drop your own copies into
  `Data/meteo/`. Stage 3 fails with a message pointing here if they are absent.
- **`2_Deliverables/` and `3_Cleaned/`** (not in the repo) — the raw Port of London
  Authority files for all 16 gauges and the intermediate cleaned set. Only needed to
  rebuild the dataset from scratch with `src/build_gauge_dataset.py`.

---

## Running things

Every script has a **CONFIG block at the top** — edit those values rather than passing
arguments. Scripts are also split into `# %%` cells, so they run either way:

```bash
python models/plot_stage2.py     # top to bottom, as a normal script
```

or open the file in VS Code and run the cells one at a time in the interactive window
(the usual notebook workflow, but on a plain `.py` file).

### Reproducing the figures without training

The checkpoints in `models/checkpoints/` and the tables under `outputs/` are all the plot
scripts need:

```bash
python models/plot_stage1.py     # stage 1 figures
python models/plot_stage2.py     # stage 2 figures (reads tables only — takes seconds)
python models/plot_stage3.py     # stage 3 figures, including SHAP
```

Figures are written to `report_images/` as PDFs. Each plot script has a `FIGURES_TO_PLOT`
list — trim it to redraw just one figure. A plot script never trains: if a checkpoint is
missing it stops and tells you which training script to run.

`plot_stage3.py` reuses the saved SHAP tables by default; set `RECOMPUTE_SHAP = True` to
re-run `GradientExplainer` from the model itself.

### Retraining

```bash
python models/train_stage1.py
python models/train_stage2.py
python models/train_stage3.py
```

Each training script skips any model whose checkpoint already exists — set
`LOAD_FROM_CHECKPOINT = False` in the CONFIG block to force a full retrain. Stage 2 and
stage 3 also have flags for the expensive robustness checks (`RUN_TIDE_ABLATION`,
`RUN_REDUCED_TRAINING`, …); turn them off for a quicker run. CPU training works but takes
hours per stage — see the next section.

### Another gauge, a shorter horizon, or a laptop

Every train and plot script has `GAUGE = 'southend'` in its CONFIG block; set it to
`'richmond'` (see `GAUGES` in `utils/models/config.py`). A non-Southend gauge writes its
tables to `outputs/<gauge>/` and its weights to `models/checkpoints/<gauge>/`, so it
never overwrites Southend's. It trains the headline models only: the robustness retrains
are Southend-only, and stage 3 reuses Southend's search-winning hyperparameters rather
than searching again. Stage 3 also reuses Southend's ERA5 pressure/wind, the only
meteorological point downloaded.

`MAX_HORIZON_H = 72` in `plot_stage2.py` / `plot_stage3.py` draws only horizons up to 72h.
The models are direct multi-horizon predictors, so this is an evaluation to 72h with the
same trained models. The figure filenames get the gauge and cap appended
(`stage2_rmse_vs_horizon_richmond_72h.pdf`); set it to `None` for the full horizon set and
the original filenames. Every RMSE/MAE/loss axis carries a small "better" arrow.

Training runs fine on a CPU-only laptop (measured on an 8-core Intel Core Ultra, 32 GB):
data preparation peaks at ~7–9 GB of RAM, and the headline Richmond models take roughly
6–7 hours in total. Every training run writes a per-epoch resume file to
`outputs/_resume/`, so an interrupted run continues from its last finished epoch.

**`models/run_gauge.py` runs a whole gauge unattended** — set `GAUGE` / `MAX_HORIZON_H`
in its CONFIG block, then from a terminal (laptop plugged in, lid open):

```powershell
& C:\Users\alfat\anaconda3\envs\geospatial\python.exe models/run_gauge.py
```

It cleans the data if needed, then runs train/plot for stage 2, then stage 3, then stage 1.
It passes the gauge to each script through `TIDAL_GAUGE` / `TIDAL_MAX_HORIZON_H`, so the
scripts' own CONFIG defaults stay at Southend. A failed step is retried (up to 3 times),
resuming rather than restarting. Logs go to `logs/<gauge>/`. Re-launching skips the steps
already done. See `CHANGELOG.md` for the full set-up record.

### Rebuilding the data

```bash
python src/build_gauge_dataset.py   # raw PLA files -> one cleaned CSV per gauge
python src/download_meteo_data.py   # ERA5 + NRFA -> the four meteo CSVs
```

Then re-run the cleaning notebook `EDA/utide_test.ipynb`, which produces the patched
Southend record that becomes `Data/southend_pier_data.csv`.

---

## Where the shared code lives

`utils/` holds everything used by more than one stage, so the three stages cannot drift
apart:

| Module | Contents |
|---|---|
| `utils/paths.py` | Every data and output location |
| `utils/plot_config.py` | Figure styling for the dissertation, `save_fig`, shared series colours |
| `utils/models/config.py` | Seeds, split boundaries, lookbacks, forecast horizons |
| `utils/models/architectures.py` | All model classes for stages 1–3 |
| `utils/models/layers.py` | The encoder and attention blocks shared by stages 2 and 3 |
| `utils/models/windowing.py` | Sliding-window construction (never crossing a data gap) |
| `utils/models/datasets.py` | Loading, splitting, UTide fitting, scaling → ready-to-train tensors |
| `utils/models/training.py` | Training loop, masked losses, metrics, checkpoint handling |
