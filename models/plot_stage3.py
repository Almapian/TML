"""Stage 3 -- figures from an already-trained run, including SHAP interpretability.

Script form of the plotting and interpretability sections of
`models/notebooks/stage3_meteo_fusion.ipynb`. It never trains: model weights come from the
checkpoints `models/train_stage3.py` wrote, and a missing checkpoint raises rather than
silently starting a training run.

Two tiers of figure:

* The RMSE figures are pure table reads -- they need only the CSVs in
  outputs/stage3_outputs/, so they draw in seconds.
* The SHAP figures need the trained model and the test windows, so selecting any of them
  loads the data (UTide fit included, a few minutes) and the checkpoint. GradientExplainer
  itself runs at most once per session: with RECOMPUTE_SHAP = False the saved SHAP tables
  are reused and every SHAP view is re-derived from them.

    python models/plot_stage3.py
    # or run the `# %%` cells interactively in VS Code
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
from utils.models import training
from utils.plot_config import SERIES_COLOURS, save_fig, set_report_dir

# %% ------------------------------------------------------------------ CONFIG
FIGURES = [
    'loss_curve',                   # both models' train/val curves (needs stage3_history.json)
    'rmse_vs_horizon_full',         # 30-horizon RMSE curve, log hours
    'reduced_training',             # training-volume ablation, RMSE vs horizon
    'rmse_vs_training_years',       # training-volume ablation, RMSE vs years
    'shap_branch_contributions',    # stacked bar, raw mean |SHAP| per branch
    'shap_branch_contributions_pct',  # the same as a share of each horizon's total
    'shap_covariates_only',         # meteo/river covariates alone, tide/UTide dropped
    'shap_tide_lookback_heatmap',   # importance by lookback position, one shared scale
    'shap_tide_lookback_normalized',  # the same, row-normalised, as small multiples
    'shap_vs_attention',            # SHAP importance vs the decoder's own attention weights
]
FIGURES_TO_PLOT = FIGURES

# SHAP settings (only used when a shap_* figure is selected)
RECOMPUTE_SHAP = False    # True re-runs GradientExplainer instead of reusing the saved tables
SHAP_HORIZONS = ['10min', '6h', '24h', '168h', '14d', '30d']
SHAP_BACKGROUND_N = 150
SHAP_EXPLAIN_N = 300

HORIZONS = C.STAGE3_HORIZONS
HORIZON_NAMES = list(HORIZONS)
HORIZON_STEPS = list(HORIZONS.values())
HEADLINE_NAMES = list(C.HEADLINE_HORIZONS)

OUTPUT_DIR = paths.STAGE3_OUTPUT_DIR
CHECKPOINT_DIR = paths.CHECKPOINTS_DIR
set_report_dir(str(paths.ensure_dir(paths.REPORT_IMAGES_DIR)))
NEEDS_MODEL = any(f.startswith('shap') for f in FIGURES_TO_PLOT)


def _read(name, **kwargs):
    path = OUTPUT_DIR / name
    if not path.exists():
        print(f"skipping figures that need {name} -- not found in {OUTPUT_DIR}. "
              f"Run models/train_stage3.py first.")
        return None
    return pd.read_csv(path, **kwargs)


# %% Loss curves ------------------------------------------------------------
histories = training.load_history(str(OUTPUT_DIR / 'stage3_history.json'))
if 'loss_curve' in FIGURES_TO_PLOT:
    if not histories:
        print("no stage3_history.json -- both models were loaded from checkpoints when "
              "train_stage3.py last ran, so there are no loss curves to draw.")
    else:
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
        save_fig(fig, 'stage3_loss_curve')
        plt.show()

# %% RMSE vs horizon, full 30-horizon curve ---------------------------------
metrics_df = _read('stage3_metrics.csv')
observed_df = _read('stage3_observed_reference.csv')
horizon_hours = np.array(HORIZON_STEPS) / 6.0   # 10-minute steps -> hours

if metrics_df is not None:
    rmse_pivot = metrics_df.pivot(index='model', columns='horizon', values='rmse_m')[HORIZON_NAMES]
    mae_pivot = metrics_df.pivot(index='model', columns='horizon', values='mae_m')[HORIZON_NAMES]

    def with_observed_row(pivot_table, horizon_names):
        """Prepend the observed-level reference row to a (model x horizon) pivot table."""
        if observed_df is None:
            return pivot_table
        obs = observed_df.set_index('horizon')['observed_mean_m']
        obs_row = pd.DataFrame([obs.reindex(horizon_names).to_numpy()],
                               index=['Observed (mean, m)'], columns=horizon_names)
        return pd.concat([obs_row, pivot_table])

    print("\nRMSE (m), headline 7 horizons -- directly comparable to stage 2's own table:")
    print(with_observed_row(rmse_pivot[HEADLINE_NAMES], HEADLINE_NAMES).round(4))
    print("\nMAE (m), headline 7 horizons:")
    print(with_observed_row(mae_pivot[HEADLINE_NAMES], HEADLINE_NAMES).round(4))

if 'rmse_vs_horizon_full' in FIGURES_TO_PLOT and metrics_df is not None:
    fig = plt.figure(figsize=(10, 5))
    for name in rmse_pivot.index:
        style = '--' if 'UTide' in name else '-'
        plt.plot(horizon_hours, rmse_pivot.loc[name], marker='o', markersize=3, linestyle=style,
                 label=name, color=SERIES_COLOURS.get(name))
    plt.axvline(168, color='grey', linewidth=0.8, linestyle=':', label='stage-2 horizon limit (168h)')
    plt.xscale('log')
    plt.xlabel('Forecast horizon (hours, log scale)')
    plt.ylabel('RMSE (m)')
    plt.legend()
    plt.grid(alpha=0.3, which='both')
    plt.tight_layout()
    save_fig(fig, 'stage3_rmse_vs_horizon_full')
    plt.show()

    if {'Stage-3 (meteo fusion)', 'Stage-3 (tide-only, no meteo)'} <= set(rmse_pivot.index):
        full = rmse_pivot.loc['Stage-3 (meteo fusion)']
        tide_only = rmse_pivot.loc['Stage-3 (tide-only, no meteo)']
        for h in HEADLINE_NAMES:
            pct = 100 * (tide_only[h] - full[h]) / tide_only[h]
            print(f"  {h:>6s}: full model vs. tide-only RMSE: {pct:+.1f}% "
                  f"(positive = adding meteo/river helped)")

# %% Training-data-volume ablation ------------------------------------------
reduced_df = _read('stage3_reduced_training_data_metrics.csv')
if reduced_df is not None:
    train_years_order = reduced_df.drop_duplicates('model').set_index('model')['train_years'].sort_values()
    reduced_pivot = reduced_df.pivot(index='model', columns='horizon',
                                     values='rmse_m')[HORIZON_NAMES].loc[train_years_order.index]
    print("\nRMSE (m) by training-data window, headline 7 horizons:")
    print(reduced_pivot[HEADLINE_NAMES].round(4))

    if 'reduced_training' in FIGURES_TO_PLOT:
        fig = plt.figure(figsize=(10, 5))
        for name in reduced_pivot.index:
            plt.plot(horizon_hours, reduced_pivot.loc[name], marker='o', markersize=3, label=name)
        plt.xscale('log')
        plt.xlabel('Forecast horizon (hours, log scale)')
        plt.ylabel('RMSE (m)')
        plt.legend(title='training window', fontsize=8)
        plt.grid(alpha=0.3, which='both')
        plt.tight_layout()
        save_fig(fig, 'stage3_reduced_training_rmse_vs_horizon')
        plt.show()

    if 'rmse_vs_training_years' in FIGURES_TO_PLOT:
        fixed_horizons_check = [h for h in ['1h', '24h', '72h', '168h', '30d'] if h in HORIZON_NAMES]
        fig = plt.figure(figsize=(8, 5))
        for hname in fixed_horizons_check:
            ys = [reduced_pivot.loc[name, hname] for name in train_years_order.index]
            plt.plot(train_years_order.values, ys, marker='o', label=hname)
        plt.xlabel('Years of training data')
        plt.ylabel('RMSE (m)')
        plt.legend(title='horizon', fontsize=8)
        plt.grid(alpha=0.3)
        plt.xticks(sorted(train_years_order.unique()))
        save_fig(fig, 'stage3_rmse_vs_training_years')
        plt.show()

# %% SHAP: data and model (only loaded when a shap_* figure is selected) -----
shap_df = None
lookback_matrix = None

if NEEDS_MODEL:
    from utils.models import datasets
    from utils.models.architectures import TidalSeq2SeqMulti

    DEVICE = torch.device('cpu')   # forward/backward passes only; no GPU required
    data = datasets.prepare_stage3(DEVICE, HORIZON_STEPS, C.DECODER_SEQ_STEPS,
                                   C.TARGET_SLICE_IN_DECODER, lookback=C.LOOKBACK)
    stage3_model, _ = training.load_checkpoint_or_raise(
        TidalSeq2SeqMulti(C.DECODER_SEQ_LEN, C.TARGET_SLICE_IN_DECODER).to(DEVICE),
        CHECKPOINT_DIR / 'stage3_tidal_seq2seq_multi.pt', label='stage3-multi', device=DEVICE)

    # Same seed and draw as the notebook, so the explained sample is reproducible and the
    # attention comparison below lines up with the saved SHAP tables.
    C.set_seed()
    bg_idx = np.random.choice(data.train_tensors['X'].shape[0], size=SHAP_BACKGROUND_N, replace=False)
    explain_idx = np.random.choice(data.test_tensors['X'].shape[0], size=SHAP_EXPLAIN_N, replace=False)
    shap_background = [data.train_tensors[k][bg_idx] for k in ('X', 'Xm', 'Xr', 'Ydec')]
    shap_explain_inputs = [data.test_tensors[k][explain_idx] for k in ('X', 'Xm', 'Xr', 'Ydec')]

# %% SHAP: compute once, or reuse the saved tables --------------------------
shap_summary_path = OUTPUT_DIR / 'stage3_shap_branch_contributions.csv'
shap_lookback_path = OUTPUT_DIR / 'stage3_shap_tide_lookback_importance.csv'

if NEEDS_MODEL:
    have_saved = shap_summary_path.exists() and shap_lookback_path.exists()
    if not RECOMPUTE_SHAP and have_saved:
        shap_df = pd.read_csv(shap_summary_path, index_col=0)
        SHAP_HORIZONS = [h for h in SHAP_HORIZONS if h in shap_df.index] or list(shap_df.index)
        shap_df = shap_df.loc[SHAP_HORIZONS]
        lookback_matrix = pd.read_csv(shap_lookback_path, index_col=0).loc[SHAP_HORIZONS].to_numpy()
        print(f"reusing saved SHAP tables ({shap_summary_path.name}, {shap_lookback_path.name}); "
              f"set RECOMPUTE_SHAP = True to re-run GradientExplainer.")
    else:
        import shap
        import torch.nn as nn

        class HorizonWrapper(nn.Module):
            """SHAP explains one fixed-size output per call; this selects a single horizon
            out of the model's (B, 30) prediction so GradientExplainer sees a (B, 1) output."""

            def __init__(self, base_model, horizon_idx):
                super().__init__()
                self.base_model = base_model
                self.horizon_idx = horizon_idx

            def forward(self, x_tide, x_meteo, x_river, x_utide):
                preds, _ = self.base_model(x_tide, x_meteo, x_river, x_utide)
                return preds[:, self.horizon_idx:self.horizon_idx + 1]

        # cuDNN's fused RNN kernel only retains backward-pass state when the forward call
        # runs in training mode, so GradientExplainer's backward() through the LSTMs fails
        # on a .eval()-mode model. Keeping the model in genuine eval() mode -- correct,
        # since SHAP is explaining a trained model, not training one -- and disabling
        # cuDNN's fused kernel just for this block routes PyTorch to the generic RNN
        # backward implementation instead, which has no such restriction.
        stage3_model.eval()
        shap_rows = []
        tide_lookback_importance = {}
        with torch.backends.cudnn.flags(enabled=False):
            for hname in SHAP_HORIZONS:
                j = HORIZON_NAMES.index(hname)
                wrapped = HorizonWrapper(stage3_model, horizon_idx=j)
                explainer = shap.GradientExplainer(wrapped, shap_background)
                sv_tide, sv_meteo, sv_river, sv_utide = explainer.shap_values(shap_explain_inputs)
                sv_tide, sv_meteo, sv_river, sv_utide = [
                    np.abs(s).squeeze(-1) for s in (sv_tide, sv_meteo, sv_river, sv_utide)]

                shap_rows.append({
                    'horizon': hname,
                    'tide_history': sv_tide.sum(axis=1).mean(),
                    'pressure': sv_meteo[..., 0].sum(axis=1).mean(),
                    'wind_u10': sv_meteo[..., 1].sum(axis=1).mean(),
                    'wind_v10': sv_meteo[..., 2].sum(axis=1).mean(),
                    'rainfall': sv_river[..., 0].sum(axis=1).mean(),
                    'discharge': sv_river[..., 1].sum(axis=1).mean(),
                    'river_availability_flags': sv_river[..., 2:4].sum(axis=(1, 2)).mean(),
                    'utide_decoder': sv_utide.sum(axis=1).mean(),
                })
                tide_lookback_importance[hname] = sv_tide.mean(axis=0)   # (96,)
                print(f"SHAP done for horizon {hname}")

        shap_df = pd.DataFrame(shap_rows).set_index('horizon').loc[SHAP_HORIZONS]
        shap_df.to_csv(shap_summary_path)
        print(f"saved {shap_summary_path}")

        lookback_matrix = np.stack([tide_lookback_importance[h] for h in SHAP_HORIZONS])
        pd.DataFrame(lookback_matrix, index=SHAP_HORIZONS).to_csv(shap_lookback_path)
        print(f"saved {shap_lookback_path}")

    print(shap_df.round(4))

covariate_cols = ['pressure', 'wind_u10', 'wind_v10', 'rainfall', 'discharge', 'river_availability_flags']

# %% SHAP figure: branch contributions --------------------------------------
if 'shap_branch_contributions' in FIGURES_TO_PLOT and shap_df is not None:
    ax = shap_df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab10')
    ax.set_ylabel('mean |SHAP| (contribution to prediction, scaled units)')
    ax.set_xlabel('horizon')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_fig(ax.figure, 'stage3_shap_branch_contributions')
    plt.show()

# The stacked bar above shrinks in total height as the horizon grows, which confounds any
# comparison of *composition* across horizons; the percentage view fixes that, and the
# covariates-only view drops the two large tide/UTide bars that otherwise dominate.
if 'shap_branch_contributions_pct' in FIGURES_TO_PLOT and shap_df is not None:
    shap_pct = shap_df.div(shap_df.sum(axis=1), axis=0) * 100
    ax = shap_pct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab10')
    ax.set_ylabel('share of total mean |SHAP| (%)')
    ax.set_xlabel('horizon')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_fig(ax.figure, 'stage3_shap_branch_contributions_pct')
    plt.show()

if 'shap_covariates_only' in FIGURES_TO_PLOT and shap_df is not None:
    ax = shap_df[covariate_cols].plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab10')
    ax.set_ylabel('mean |SHAP| (contribution to prediction, scaled units)')
    ax.set_xlabel('horizon')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_fig(ax.figure, 'stage3_shap_covariates_only')
    plt.show()

# %% SHAP figure: importance by tide-lookback position ----------------------
if 'shap_tide_lookback_heatmap' in FIGURES_TO_PLOT and lookback_matrix is not None:
    fig = plt.figure(figsize=(12, 4))
    plt.imshow(lookback_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='mean |SHAP|')
    plt.yticks(range(len(SHAP_HORIZONS)), SHAP_HORIZONS)
    plt.xlabel('lookback position (steps before anchor; 0 = 16h before, 95 = the anchor itself)')
    plt.ylabel('horizon')
    plt.tight_layout()
    save_fig(fig, 'stage3_shap_tide_lookback_heatmap')
    plt.show()

# One shared colour scale across rows that differ by roughly an order of magnitude in peak
# size washes out everything but the 10min row. Row-normalising fixes the scale; with only
# six rows, small-multiple lines then read more precisely than colour, so that is the
# version saved (the row-normalised heatmap is drawn for comparison but not written out).
if 'shap_tide_lookback_normalized' in FIGURES_TO_PLOT and lookback_matrix is not None:
    row_normed = lookback_matrix / lookback_matrix.max(axis=1, keepdims=True)

    fig_heat = plt.figure(figsize=(12, 4))
    plt.imshow(row_normed, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label="mean |SHAP| (normalised to each horizon's own max)")
    plt.yticks(range(len(SHAP_HORIZONS)), SHAP_HORIZONS)
    plt.xlabel('lookback position (steps before anchor; 0 = 16h before, 95 = the anchor itself)')
    plt.ylabel('horizon')
    plt.tight_layout()

    fig_lines, axes = plt.subplots(len(SHAP_HORIZONS), 1, figsize=(8, 9), sharex=True)
    for ax, hname in zip(np.atleast_1d(axes), SHAP_HORIZONS):
        ax.plot(row_normed[SHAP_HORIZONS.index(hname)], color='#1f77b4')
        ax.set_ylabel(hname, rotation=0, ha='right', va='center', fontsize=9)
        ax.set_ylim(0, 1.05)
    np.atleast_1d(axes)[-1].set_xlabel(
        'lookback position (steps before anchor; 0 = 16h before, 95 = the anchor itself)')
    plt.tight_layout()
    plt.show()

    save_fig(fig_lines, 'stage3_shap_tide_lookback_normalized')

# %% SHAP vs the decoder's own attention weights ----------------------------
# Does the decoder's tide-branch attention (learned during training) agree with what SHAP
# independently attributes to the same lookback positions? A single forward pass (no
# gradients) reads off the attention weights the trained model already computes.
if 'shap_vs_attention' in FIGURES_TO_PLOT and lookback_matrix is not None and NEEDS_MODEL:
    stage3_model.eval()
    with torch.no_grad():
        _, (w_tide, w_meteo, w_river) = stage3_model(*shap_explain_inputs)
    w_tide = w_tide.cpu().numpy()   # (N_explain, 30, 96)
    print(f"w_tide shape: {w_tide.shape}")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, hname in zip(axes.flat, SHAP_HORIZONS):
        shap_curve = lookback_matrix[SHAP_HORIZONS.index(hname)]
        shap_curve = shap_curve / shap_curve.max()
        attn_curve = w_tide[:, HORIZON_NAMES.index(hname), :].mean(axis=0)
        attn_curve = attn_curve / attn_curve.max()

        ax.plot(shap_curve, label='SHAP')
        ax.plot(attn_curve, label='attention')
        ax.text(0.03, 0.92, hname, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')

    axes[0, 0].legend(fontsize=8, loc='upper right')
    fig.supxlabel('lookback position')
    plt.tight_layout()
    save_fig(fig, 'stage3_shap_vs_attention', width_in=6.3, height_in=4.5)
    plt.show()

    print("\nCovariate share of total mean |SHAP|, largest to smallest per horizon:")
    for hname in SHAP_HORIZONS:
        row = shap_df.loc[hname, covariate_cols]
        top = row.idxmax()
        print(f"  {hname:>6s}: {row.sum():.4f} total covariate |SHAP| -- largest single covariate: "
              f"{top} ({row[top]:.4f}, {100 * row[top] / row.sum():.0f}% of the covariate share)")

    print("\nSHAP vs. attention agreement, by horizon "
          "(Pearson correlation of the two normalised lookback curves):")
    for hname in SHAP_HORIZONS:
        shap_curve = lookback_matrix[SHAP_HORIZONS.index(hname)]
        shap_curve = shap_curve / shap_curve.max()
        attn_curve = w_tide[:, HORIZON_NAMES.index(hname), :].mean(axis=0)
        attn_curve = attn_curve / attn_curve.max()
        corr = np.corrcoef(shap_curve, attn_curve)[0, 1]
        verdict = 'agree' if corr > 0.5 else ('diverge' if corr < 0.0 else 'partially agree')
        print(f"  {hname:>6s}: r={corr:+.3f} -- {verdict}")

print(f"\nFigures written to: {paths.REPORT_IMAGES_DIR}")
