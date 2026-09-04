"""Stage 1 -- figures and tables from an already-trained run.

Script form of the plotting sections of `models/notebooks/initial_modelling.ipynb`.
Loads the checkpoints and metric tables written by `models/train_stage1.py` and redraws
every figure into `report_images/`. It never trains: if a checkpoint is missing it fails
with a message telling you to run the training script first.

Runs fine on CPU -- checkpoints are loaded with map_location, and only forward passes are
needed for the prediction figure.

    python models/plot_stage1.py
    # or run the `# %%` cells interactively in VS Code

Set FIGURES_TO_PLOT to a subset of FIGURES below to redraw only some of them.
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
import torch

from utils import paths
from utils.models import config as C
from utils.models import datasets, training
from utils.models.architectures import MLP, LSTMModel, RNNModel
from utils.plot_config import SERIES_COLOURS, save_fig, set_report_dir

# %% ------------------------------------------------------------------ CONFIG
FIGURES = [
    'loss_curves',            # train/val MSE per model (needs initial_history.json)
    'split_distributions',    # level and 10-min step distributions per split
    'yearly_volatility',      # std of the 10-min step change, year by year
    'predictions_vs_observed',  # test-set sample window, predictions and error
]
FIGURES_TO_PLOT = FIGURES     # e.g. ['predictions_vs_observed'] to redraw just one

# window shown by the predictions figure
PLOT_WINDOW_START = '2022-01-01'
PLOT_WINDOW_DAYS = 14

LOOKBACK = C.LOOKBACK
OUTPUT_DIR = paths.STAGE1_OUTPUT_DIR
CHECKPOINT_DIR = paths.CHECKPOINTS_DIR
set_report_dir(str(paths.ensure_dir(paths.REPORT_IMAGES_DIR)))
DEVICE = torch.device('cpu')   # plotting is forward-pass only; no GPU required

# %% Data and trained models ------------------------------------------------
data = datasets.prepare_stage1(DEVICE, lookback=LOOKBACK)

model_factories = {
    'MLP': lambda: MLP(LOOKBACK),
    'RNN': lambda: RNNModel(hidden_size=64, num_layers=1),
    'LSTM': lambda: LSTMModel(hidden_size=64, num_layers=1),
}
predict_flat = lambda model, batch: model(batch['X'])

trained = {}
for name, factory in model_factories.items():
    model, _ = training.load_checkpoint_or_raise(
        factory().to(DEVICE), CHECKPOINT_DIR / f'initial_{name.lower()}.pt', label=name, device=DEVICE)
    trained[name] = model

test_preds = {}
for name, model in trained.items():
    _, _, test_preds[name] = training.compute_metrics(
        model, predict_flat, data.test_tensors, data.y_test, data.flag_test,
        data.train_mean, data.train_std)

metrics_df = pd.read_csv(OUTPUT_DIR / 'initial_metrics.csv')
print("\nOne-step-ahead metrics (from initial_metrics.csv):")
print(metrics_df.pivot(index='model', columns='split', values=['rmse_m', 'mae_m']).round(4))

rollout_path = OUTPUT_DIR / 'initial_rollout_rmse.csv'
if rollout_path.exists():
    print("\nRecursive-rollout RMSE (m):")
    print(pd.read_csv(rollout_path, index_col=0).round(4))

# %% Figure: loss curves ----------------------------------------------------
if 'loss_curves' in FIGURES_TO_PLOT:
    histories = training.load_history(str(OUTPUT_DIR / 'initial_history.json'))
    if not histories:
        print("no initial_history.json -- every model was loaded from a checkpoint when "
              "train_stage1.py last ran, so there are no loss curves to draw.")
    else:
        n = len(histories)
        fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), squeeze=False)
        for col, (name, history) in enumerate(histories.items()):
            epochs = np.arange(1, len(history['train_loss']) + 1)
            # second row repeats the curve from epoch 2, where the first epoch's much
            # larger loss no longer compresses the rest of the axis
            for row, start_epoch in enumerate([1, 2]):
                ax = axes[row, col]
                mask = epochs >= start_epoch
                ax.plot(epochs[mask], np.array(history['train_loss'])[mask], label='train')
                ax.plot(epochs[mask], np.array(history['val_loss'])[mask], label='val')
                ax.set_yscale('log')
                ax.set_xlabel('epoch')
                if col == 0:
                    ax.legend(fontsize=8)
                ax.grid(alpha=0.3, which='both')
            axes[0, col].set_title(name, fontsize=11)

        axes[0, 0].set_ylabel('MSE loss\n(log scale)', fontsize=9)
        axes[1, 0].set_ylabel('MSE loss\n(log, epoch >= 2)', fontsize=9)
        save_fig(fig, 'initial_loss_curves', width_in=6.3, height_in=4.4, w_pad=2.0, h_pad=2.5)
        plt.show()

# %% Figure: split distributions -------------------------------------------
# Context for "why is val loss lower than train loss": val is an easier period, not
# better generalisation -- train has ~35x the flagged rate of val, and flagged points are
# the hardest to fit.
splits = {'train': data.train_df, 'val': data.val_df, 'test': data.test_df}
step_diffs = {}
print(f"\n{'split':6s} {'n':>10s} {'mean_m':>8s} {'std_m':>8s} {'min_m':>8s} {'max_m':>8s} "
      f"{'step_mean_abs':>14s} {'step_std':>10s} {'flagged%':>9s}")
for name, d in splits.items():
    level = d['Observed_ODN']
    # genuine 10-min step, both endpoints unflagged -- so neither a removed/flagged row
    # nor either side of a data gap is ever counted as a real step
    is_real_step = ((d['DateTime'].diff() == C.STEP) & ~d['is_flagged']
                    & ~d['is_flagged'].shift(1, fill_value=True))
    step = d['Observed_ODN'].diff()[is_real_step]
    step_diffs[name] = step
    print(f"{name:6s} {len(d):>10,} {level.mean():>8.4f} {level.std():>8.4f} {level.min():>8.4f} "
          f"{level.max():>8.4f} {step.abs().mean():>14.5f} {step.std():>10.5f} "
          f"{100 * d['is_flagged'].mean():>8.3f}%")

if 'split_distributions' in FIGURES_TO_PLOT:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, d in splits.items():
        axes[0].hist(d['Observed_ODN'], bins=100, histtype='step', density=True, label=name, linewidth=1.5)
    axes[0].set_xlabel('Water level (m ODN)')
    axes[0].set_ylabel('density')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for name, step in step_diffs.items():
        axes[1].hist(step, bins=200, range=(-0.15, 0.15), histtype='step', density=True,
                     label=name, linewidth=1.5)
    axes[1].set_xlabel('10-min change in water level (m)')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(alpha=0.3, which='both')
    plt.tight_layout()
    save_fig(fig, 'initial_split_distributions', width_in=6.3, height_in=2.6)
    plt.show()

# %% Figure: year-by-year volatility ----------------------------------------
if 'yearly_volatility' in FIGURES_TO_PLOT:
    df = data.df
    is_real_step_full = ((df['DateTime'].diff() == C.STEP) & ~df['is_flagged']
                         & ~df['is_flagged'].shift(1, fill_value=True))
    full_step = df['Observed_ODN'].diff()[is_real_step_full]
    yearly_step_std = full_step.groupby(df.loc[is_real_step_full, 'DateTime'].dt.year).std()

    train_end_year = pd.Timestamp(C.TRAIN_END).year
    val_end_year = pd.Timestamp(C.VAL_END).year
    split_colors = {'train': '#d62728', 'val': '#1f77b4', 'test': '#2ca02c'}
    bar_colors = [split_colors['train'] if y < train_end_year
                  else split_colors['val'] if y < val_end_year
                  else split_colors['test'] for y in yearly_step_std.index]

    fig = plt.figure(figsize=(14, 4))
    plt.bar(yearly_step_std.index, yearly_step_std.values, color=bar_colors)
    plt.axvline(train_end_year - 0.5, color='k', linestyle='--', linewidth=1)
    plt.axvline(val_end_year - 0.5, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Year (red = train, blue = validation, green = test)')
    plt.ylabel('std of 10-min step (m)')
    plt.grid(alpha=0.3, axis='y')
    save_fig(fig, 'initial_yearly_volatility', width_in=6.3, height_in=2.4)
    plt.show()

# %% Figure: predictions vs observed ----------------------------------------
if 'predictions_vs_observed' in FIGURES_TO_PLOT:
    window_start = pd.Timestamp(PLOT_WINDOW_START)
    window_end = window_start + pd.Timedelta(days=PLOT_WINDOW_DAYS)
    plot_mask = ((data.time_test >= np.datetime64(window_start))
                 & (data.time_test < np.datetime64(window_end)))

    fig, (ax_level, ax_err) = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True, height_ratios=[2, 1])

    ax_level.plot(data.time_test[plot_mask], data.y_test[plot_mask], label='Observed',
                  color='black', linewidth=1)
    for name, preds_m in test_preds.items():
        ax_level.plot(data.time_test[plot_mask], preds_m[plot_mask], label=name, linewidth=1,
                      color=SERIES_COLOURS.get(name))
    ax_level.set_ylabel('Water level (m ODN)')
    ax_level.legend()
    ax_level.grid(alpha=0.3)

    # Predictions sit within ~1-2cm of Observed at this horizon, so on the raw level axis
    # every line overlaps and whichever is drawn last hides the rest. The error panel is
    # where the differences are actually visible.
    for name, preds_m in test_preds.items():
        ax_err.plot(data.time_test[plot_mask], preds_m[plot_mask] - data.y_test[plot_mask],
                    linewidth=1, color=SERIES_COLOURS.get(name))
    ax_err.axhline(0, color='black', linewidth=0.8)
    ax_err.set_xlabel('Date')
    ax_err.set_ylabel('Error (m)')
    ax_err.grid(alpha=0.3)

    save_fig(fig, 'initial_predictions_vs_observed', width_in=6.3, height_in=4.2)
    plt.show()

print(f"\nFigures written to: {paths.REPORT_IMAGES_DIR}")
