"""Figures for the dense (continuous 0-72h) pipeline. Reads tables only -- no training.

    python models/plot_dense.py      # after models/train_dense.py, for the same GAUGE

Two figures per gauge, as 300-dpi PNG in report_images/ (suffix _<gauge> for non-Southend):

    stages_rmse_at_stage1_leads   every model from stage 1 to the dense models, at stage 1's
                                  five lead times (10min, 1h, 6h, 24h, 48h); also as a CSV
    dense_72h_forecast            one continuous 72h forecast issued in Feb 2022 against the
                                  observed water level

Colour = model family (stage 1 / stage 2 / stage 3, UTide grey); filled = dense, hollow =
the sparse original. The family palette passed the dataviz colour checks (CVD, normal vision).
"""
# %% Bootstrap
import pathlib
import sys

_start = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / '_')).resolve()
REPO_ROOT = next(p for p in _start.parents if (p / 'utils' / 'paths.py').exists())
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
if not hasattr(sys, 'ps1') and 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')

# %% Imports
import matplotlib.dates as mdates
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from utils import paths
from utils.models import config as C
from utils.plot_config import ANNOTATION_INK, figure_suffix, save_fig, set_name_suffix, set_report_dir

# %% ------------------------------------------------------------------ CONFIG
GAUGE = C.from_env('TIDAL_GAUGE', 'southend')
SMOKE = C.from_env('TIDAL_SMOKE', '0').strip().lower() not in ('', '0', 'false')
LEADS = list(C.STAGE1_ROLLOUT_HORIZONS)            # 10min, 1h, 6h, 24h, 48h

DENSE_DIR = pathlib.Path(str(paths.dense_output_dir(GAUGE)) + ('_smoke' if SMOKE else ''))
set_report_dir(str(DENSE_DIR if SMOKE else paths.ensure_dir(paths.REPORT_IMAGES_DIR)))
set_name_suffix(figure_suffix(GAUGE))
LABEL = C.gauge_info(GAUGE)['label']

FAMILY = {'UTide': '#52514e', 'stage1': '#2a78d6', 'stage2': '#eb6834', 'stage3': '#1baf7a'}
INK = '#0b0b0b'

# (row label, family, marker, filled, source, model name in that source)
ROWS = [
    ('UTide (harmonic baseline)', 'UTide', 'D', False, 'stage2', 'UTide standalone'),
    ('Stage 1 LSTM, recursive', 'stage1', 'o', False, 'stage1', 'LSTM'),
    ('Stage 1 RNN, recursive', 'stage1', 'v', False, 'stage1', 'RNN'),
    ('Stage 1 MLP, recursive', 'stage1', 's', False, 'stage1', 'MLP'),
    ('Flat LSTM, direct', 'stage1', 'o', True, 'stage2', 'Stage-1 baseline (flat LSTM)'),
    ('Flat MLP, direct', 'stage1', 's', True, 'stage2', 'Stage-1 baseline (flat MLP)'),
    ('Stage 2 seq2seq (sparse)', 'stage2', 'o', False, 'stage2', 'Stage-2 (attn decoder)'),
    ('Stage 2 seq2seq (dense)', 'stage2', 'o', True, 'dense', 'Dense stage 2'),
    ('Stage 3 meteo fusion (sparse)', 'stage3', '^', False, 'stage3', 'Stage-3 (meteo fusion)'),
    ('Stage 3 tide-only (sparse)', 'stage3', 'o', False, 'stage3', 'Stage-3 (tide-only, no meteo)'),
    ('Stage 3 tide-only (dense)', 'stage3', 'o', True, 'dense', 'Dense stage 3 (tide-only)'),
]


# %% Load ---------------------------------------------------------------------
def long(df):
    return df.pivot_table(index='model', columns='horizon', values='rmse_m')


tables = {
    'stage1': pd.read_csv(paths.stage_output_dir(1, GAUGE) / 'initial_rollout_rmse.csv').set_index('model'),
    'stage2': long(pd.read_csv(paths.stage_output_dir(2, GAUGE) / 'stage2_metrics.csv')),
    'stage3': long(pd.read_csv(paths.stage_output_dir(3, GAUGE) / 'stage3_metrics.csv')),
    'dense': long(pd.read_csv(DENSE_DIR / 'dense_metrics.csv')),
}
grid = pd.DataFrame([[tables[src].loc[name, lead] for lead in LEADS] for _, _, _, _, src, name in ROWS],
                    index=[r[0] for r in ROWS], columns=LEADS)
grid.index.name = 'model'
grid_path = DENSE_DIR / f'stages_rmse_at_stage1_leads.csv'
grid.round(4).to_csv(grid_path)
print(f"saved {grid_path}\n{grid.round(4)}")

# %% Figure 1: every stage at stage 1's five lead times -----------------------
fig, axes = plt.subplots(1, len(LEADS), sharey=True, figsize=(6.3, 4.4))
y = np.arange(len(ROWS))[::-1]
lo, hi = np.nanmin(grid.values), np.nanmax(grid.values)
for ax, lead in zip(axes, LEADS):
    best = grid[lead].idxmin()
    for yi, (label, fam, mk, filled, _, _) in zip(y, ROWS):
        v = grid.loc[label, lead]
        col = FAMILY[fam]
        ax.scatter(v, yi, marker=mk, s=34, facecolors=col if filled else 'white', edgecolors=col,
                   linewidths=1.4, zorder=3)
        if label == best:
            ax.scatter(v, yi, marker='o', s=150, facecolors='none', edgecolors=INK, linewidths=0.8, zorder=2)
    ax.set_xscale('log')
    ax.set_xlim(lo / 1.6, hi * 1.6)
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=4))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(lead, fontsize=10)
    ax.grid(axis='y', visible=False)
    ax.tick_params(axis='x', labelsize=7)
    for sep in (len(ROWS) - 1.5, len(ROWS) - 4.5, len(ROWS) - 6.5, len(ROWS) - 8.5):   # between family blocks
        ax.axhline(sep, color='0.85', lw=0.6, zorder=0)
axes[0].set_yticks(y, [r[0] for r in ROWS], fontsize=8)
fig.supxlabel('Test RMSE (m, log scale)', fontsize=10, y=0.07)
handles = [Line2D([], [], ls='', marker='o', mfc='white', mec='0.3', label='sparse / original'),
           Line2D([], [], ls='', marker='o', color='0.3', label='direct or dense'),
           Line2D([], [], ls='', marker='o', mfc='none', mec=INK, ms=11, label='lowest at that lead')]
fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.6, 1.0))
fig.text(0.99, 0.005, f'{LABEL}. Each stage is scored on its own test windows (2021-2024).',
         ha='right', va='bottom', fontsize=7, color=ANNOTATION_INK)
save_fig(fig, 'stages_rmse_at_stage1_leads', width_in=6.3, height_in=4.6, rect=(0, 0.08, 1, 0.93))
plt.close(fig)

# %% Figure 2: one continuous 72h forecast ------------------------------------
ex = pd.read_csv(DENSE_DIR / 'dense_72h_example.csv', parse_dates=['issue_time', 'valid_time'])
obs = pd.read_csv(DENSE_DIR / 'dense_72h_example_observed.csv', parse_dates=['DateTime'])
issue = ex['issue_time'].iloc[0]

fig, ax = plt.subplots(figsize=(6.3, 3.6))
ax.plot(obs['DateTime'], obs['Observed_ODN'], color=INK, lw=1.1, label='Observed')
ax.plot(obs['DateTime'], obs['utide_recon'], color=FAMILY['UTide'], lw=1.0, ls=(0, (4, 3)), label='UTide')
for name, col, lab in [('Dense stage 2', FAMILY['stage2'], 'Stage 2 (dense)'),
                       ('Dense stage 3 (tide-only)', FAMILY['stage3'], 'Stage 3 tide-only (dense)')]:
    d = ex[ex['model'] == name].sort_values('lead_h')
    ax.plot(d['valid_time'], d['forecast_m'], color=col, lw=1.8, label=lab, zorder=3)
sp = ex[ex['model'] == 'Sparse stage 2']
ax.scatter(sp['valid_time'], sp['forecast_m'], s=42, facecolors='white', edgecolors=FAMILY['stage2'],
           linewidths=1.6, zorder=4, label='Stage 2 (sparse): its 6 outputs')
ax.axvline(issue, color=ANNOTATION_INK, lw=0.8, ls=':')
ax.text(issue, 1.0, ' issue time', transform=ax.get_xaxis_transform(), fontsize=8, color=ANNOTATION_INK,
        va='top', ha='left')
ax.text(issue - pd.Timedelta(hours=12), 1.0, 'history ', transform=ax.get_xaxis_transform(), fontsize=8,
        color=ANNOTATION_INK, va='top', ha='center')
ax.set_ylabel('Water level (m ODN)')
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
ax.tick_params(axis='x', labelsize=7.5)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=8)
fig.text(0.01, 0.005, f'{LABEL}: forecast issued {issue:%d %b %Y %H:%M} UTC, 72h ahead.',
         fontsize=7, color=ANNOTATION_INK)
save_fig(fig, 'dense_72h_forecast', width_in=6.3, height_in=3.9)
plt.close(fig)

print("\nDone.")
