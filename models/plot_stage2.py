"""Stage 2 -- figures from an already-trained run.

Script form of the plotting sections of `models/notebooks/stage2_utide_attention.ipynb`.
Every stage-2 figure is derived from the tables `models/train_stage2.py` wrote, so this
script loads no model weights, needs no GPU, and takes seconds: it reads the CSVs (plus
stage2_history.json for the loss curves) and redraws into `report_images/`.

    python models/plot_stage2.py
    # or run the `# %%` cells interactively in VS Code

Set FIGURES_TO_PLOT to a subset of FIGURES to redraw only some of them. A figure whose
source CSV is missing is skipped with a message naming what to run.
"""
# %% Bootstrap
import pathlib
import sys

_start = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / '_')).resolve()
REPO_ROOT = next(p for p in _start.parents if (p / 'utils' / 'paths.py').exists())
sys.path.insert(0, str(REPO_ROOT))

# Run as a plain script, matplotlib must not block on an interactive window; figures are
# written by save_fig either way. Interactive sessions (VS Code cells, IPython) keep their
# own backend so the plots still appear inline.
import matplotlib
if not hasattr(sys, 'ps1') and 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')

# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import paths
from utils.models import config as C
from utils.models import training
from utils.plot_config import SERIES_COLOURS, save_fig, set_report_dir

# %% ------------------------------------------------------------------ CONFIG
FIGURES = [
    'loss_curves_combined',      # all models' train/val curves on one axis
    'loss_curves',               # one panel per model
    'rmse_vs_horizon',           # headline comparison, all comparators
    'ablation_rmse_vs_horizon',  # 13.1 tide-input ablation
    'rmse_by_calendar_year',     # 13.2 accuracy later in the test window
    'reduced_training',          # 13.3 RMSE vs horizon per training window
    'rmse_vs_training_years',    # 13.3 RMSE vs years of training data
]
FIGURES_TO_PLOT = FIGURES

HORIZON_NAMES = list(C.STAGE2_HORIZONS)
OUTPUT_DIR = paths.STAGE2_OUTPUT_DIR
set_report_dir(str(paths.ensure_dir(paths.REPORT_IMAGES_DIR)))


def _read(name):
    """Read one of train_stage2.py's tables, or return None with a note if it is missing."""
    path = OUTPUT_DIR / name
    if not path.exists():
        print(f"skipping figures that need {name} -- not found in {OUTPUT_DIR}. "
              f"Run models/train_stage2.py first.")
        return None
    return pd.read_csv(path)


# %% Loss curves ------------------------------------------------------------
histories = training.load_history(str(OUTPUT_DIR / 'stage2_history.json'))
if not histories:
    print("no stage2_history.json -- every model was loaded from a checkpoint when "
          "train_stage2.py last ran, so there are no loss curves to draw.")

if 'loss_curves_combined' in FIGURES_TO_PLOT and histories:
    # one axis for every model, so convergence speed and final loss level compare directly
    fig = plt.figure(figsize=(8, 5))
    for name, history in histories.items():
        epochs_arr = np.arange(1, len(history['train_loss']) + 1)
        color = SERIES_COLOURS.get(name)
        plt.plot(epochs_arr, history['train_loss'], label=f'{name} -- train', color=color, linestyle='-')
        plt.plot(epochs_arr, history['val_loss'], label=f'{name} -- val', color=color, linestyle='--')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss (scaled, log scale)')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3, which='both')
    plt.tight_layout()
    save_fig(fig, 'stage2_loss_curves_combined')
    plt.show()

if 'loss_curves' in FIGURES_TO_PLOT and histories:
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    axes = axes[0]
    for ax, (name, history) in zip(axes, histories.items()):
        epochs_arr = np.arange(1, len(history['train_loss']) + 1)
        ax.plot(epochs_arr, history['train_loss'], label='train')
        ax.plot(epochs_arr, history['val_loss'], label='val')
        ax.set_yscale('log')
        ax.set_xlabel('epoch')
        ax.set_title(name, fontsize=11)
        ax.legend()
        ax.grid(alpha=0.3, which='both')
    axes[0].set_ylabel('MSE loss (scaled, log scale)')
    plt.tight_layout()
    save_fig(fig, 'stage2_loss_curves', width_in=6.3, height_in=6.3 * (4 / (6 * n)))
    plt.show()

# %% RMSE vs horizon --------------------------------------------------------
metrics_df = _read('stage2_metrics.csv')
if 'rmse_vs_horizon' in FIGURES_TO_PLOT and metrics_df is not None:
    rmse_pivot = metrics_df.pivot(index='model', columns='horizon', values='rmse_m')[HORIZON_NAMES]
    print("\nRMSE (m) per horizon:")
    print(rmse_pivot.round(4))
    print("\nMAE (m) per horizon:")
    print(metrics_df.pivot(index='model', columns='horizon', values='mae_m')[HORIZON_NAMES].round(4))

    fig = plt.figure(figsize=(8, 5))
    for name in rmse_pivot.index:
        style = '--' if 'UTide' in name or 'persistence' in name else '-'
        plt.plot(rmse_pivot.columns, rmse_pivot.loc[name], marker='o', linestyle=style,
                 label=name, color=SERIES_COLOURS.get(name))
    plt.xlabel('Forecast horizon')
    plt.ylabel('RMSE (m)')
    plt.legend()
    plt.grid(alpha=0.3)
    save_fig(fig, 'stage2_rmse_vs_horizon')
    plt.show()

# %% Tide-input ablation ----------------------------------------------------
ablation_df = _read('stage2_ablation_metrics.csv')
if 'ablation_rmse_vs_horizon' in FIGURES_TO_PLOT and ablation_df is not None:
    ablation_pivot = ablation_df.pivot(index='model', columns='horizon', values='rmse_m')[HORIZON_NAMES]
    order = ['Naive persistence', 'Tide-aware persistence', 'UTide standalone',
             'Stage-2 (no tide, ablation)', 'Stage-2 (with tide)']
    ablation_pivot = ablation_pivot.loc[[m for m in order if m in ablation_pivot.index]]
    print("\nTide-input ablation, RMSE (m):")
    print(ablation_pivot.round(4))

    fig = plt.figure(figsize=(8, 5))
    for name in ablation_pivot.index:
        style = '--' if name in ('Naive persistence', 'Tide-aware persistence', 'UTide standalone') else '-'
        plt.plot(HORIZON_NAMES, ablation_pivot.loc[name], marker='o', linestyle=style,
                 label=name, color=SERIES_COLOURS.get(name))
    plt.xlabel('Forecast horizon')
    plt.ylabel('RMSE (m)')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    save_fig(fig, 'stage2_ablation_rmse_vs_horizon')
    plt.show()

    if {'Stage-2 (with tide)', 'Stage-2 (no tide, ablation)'} <= set(ablation_pivot.index):
        with_tide = ablation_pivot.loc['Stage-2 (with tide)']
        no_tide = ablation_pivot.loc['Stage-2 (no tide, ablation)']
        for h in HORIZON_NAMES:
            pct = 100 * (no_tide[h] - with_tide[h]) / no_tide[h]
            print(f"  {h:>6s}: removing the tidal input increases RMSE by {pct:.1f}%")

# %% Accuracy by calendar year ---------------------------------------------
year_df = _read('stage2_rmse_by_calendar_year.csv')
if 'rmse_by_calendar_year' in FIGURES_TO_PLOT and year_df is not None:
    year_pivot = year_df.pivot(index='year', columns='horizon', values='rmse_m')[HORIZON_NAMES]
    print("\nRMSE (m) by calendar year of the forecast target:")
    print(year_pivot.round(4))

    fig = plt.figure(figsize=(8, 5))
    for hname in HORIZON_NAMES:
        plt.plot(year_pivot.index, year_pivot[hname], marker='o', label=hname)
    plt.xlabel('Calendar year of forecast target (test period)')
    plt.ylabel('RMSE (m)')
    plt.legend(title='horizon', fontsize=8)
    plt.grid(alpha=0.3)
    plt.xticks(year_pivot.index)
    save_fig(fig, 'stage2_rmse_by_calendar_year')
    plt.show()

week_df = _read('stage2_rmse_by_test_week.csv')
if week_df is not None:
    print("\nRMSE (m) for individual test weeks:")
    print(week_df.pivot(index='week', columns='horizon', values='rmse_m')[HORIZON_NAMES].round(4))

# %% Sensitivity to training-data volume ------------------------------------
reduced_df = _read('stage2_reduced_training_data_metrics.csv')
utide_only_df = _read('stage2_reduced_training_utide_only_metrics.csv')
if reduced_df is not None:
    train_years_order = reduced_df.drop_duplicates('model').set_index('model')['train_years'].sort_values()
    reduced_pivot = reduced_df.pivot(index='model', columns='horizon',
                                     values='rmse_m')[HORIZON_NAMES].loc[train_years_order.index]
    print("\nRMSE (m) by training-data window:")
    print(reduced_pivot.round(4))
    if utide_only_df is not None:
        print("\nUTide-standalone RMSE per training window "
              "(isolates constituent-fit quality from the neural net):")
        print(utide_only_df.pivot(index='model', columns='horizon',
                                  values='rmse_m')[HORIZON_NAMES].loc[train_years_order.index].round(4))

    if 'reduced_training' in FIGURES_TO_PLOT:
        fig = plt.figure(figsize=(8, 5))
        for name in reduced_pivot.index:
            plt.plot(HORIZON_NAMES, reduced_pivot.loc[name], marker='o', label=name)
        plt.xlabel('Forecast horizon')
        plt.ylabel('RMSE (m)')
        plt.legend(title='training window', fontsize=8)
        plt.grid(alpha=0.3)
        save_fig(fig, 'stage2_reduced_training_rmse_vs_horizon')
        plt.show()

    if 'rmse_vs_training_years' in FIGURES_TO_PLOT:
        fixed_horizons = ['1h', '24h', '72h', '168h']
        fig = plt.figure(figsize=(8, 5))
        for hname in fixed_horizons:
            ys = [reduced_pivot.loc[name, hname] for name in train_years_order.index]
            plt.plot(train_years_order.values, ys, marker='o', label=hname)
        plt.xlabel('Years of training data')
        plt.ylabel('RMSE (m)')
        plt.legend(title='horizon', fontsize=8)
        plt.grid(alpha=0.3)
        plt.xticks(sorted(train_years_order.unique()))
        save_fig(fig, 'stage2_rmse_vs_training_years')
        plt.show()

print(f"\nFigures written to: {paths.REPORT_IMAGES_DIR}")
