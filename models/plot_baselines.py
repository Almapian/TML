"""Figures for the classical baselines. Reads tables only -- no fitting.

    python models/plot_baselines.py      # after models/eval_baselines.py, for the same GAUGE

Seven figures per gauge, as 300-dpi PNG in report_images/ (suffix _<gauge> for non-Southend):

    baselines_rmse_vs_lead      every tuned baseline against UTide and the dense models, per track
    baselines_rmse_vs_history   validation RMSE against history length: the parity gap
    baselines_bootstrap_diff    RMSE(best baseline) - RMSE(dense stage 2) with 95% intervals
    baselines_72h_forecast      the Feb 2022 forecast against the observed level
    baselines_rmse_classical    the tuned baselines against each other and UTide
    baselines_rmse_vs_best_nn   baselines close to the best neural model (CLOSE_RATIO), if any
    baselines_forecast_panels   the Feb 2022 forecast, one panel per model

Two tables in outputs[/<gauge>]/baselines/: baseline_horizon_table.csv (RMSE at REPORT_LEADS)
and baseline_close_to_nn.csv (the closeness check).
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
import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import paths
from utils.models import config as C
from utils.plot_config import ANNOTATION_INK, figure_suffix, mark_better, save_fig, set_name_suffix, set_report_dir

# %% ------------------------------------------------------------------ CONFIG
GAUGE = C.from_env('TIDAL_GAUGE', 'southend')
SMOKE = C.from_env('TIDAL_SMOKE', '0').strip().lower() not in ('', '0', 'false')

OUT = pathlib.Path(str(paths.baseline_output_dir(GAUGE)) + ('_smoke' if SMOKE else ''))
set_report_dir(str(OUT if SMOKE else paths.ensure_dir(paths.REPORT_IMAGES_DIR)))
set_name_suffix(figure_suffix(GAUGE))
LABEL = C.gauge_info(GAUGE)['label']

METHOD = {'ar': ('Long AR', '#2a78d6', 'o'), 'sarima': ('SARIMA', '#1baf7a', 's'), 'gp': ('Gaussian process', '#eb6834', '^'),
          'dhr': ('Dynamic harmonic', '#8a5cd1', 'D'), 'ridge': ('Direct ridge', '#b7791f', 'v')}
INK = '#0b0b0b'
UTIDE = '#52514e'
TRACK_LABEL = {'level': 'Full water level', 'residual': 'UTide residual hybrid'}
MARK_EVERY = 12
REPORT_LEADS = ['10min', '1h', '6h', '24h', '48h', '72h']
CLOSE_RATIO = 1.10    # compared with the best NN only if within this at 72h and on average over leads

metrics = pd.read_csv(OUT / 'baseline_metrics.csv')
winners = json.loads((OUT / 'winners.json').read_text(encoding='utf-8'))


def curve(model):
    return metrics[metrics.model == model].sort_values('lead_h')


# %% Figure 1: RMSE against lead ----------------------------------------------
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6.3, 3.9))
for ax, track in zip(axes, C.BASELINE_TRACKS):
    for method, (name, colour, marker) in METHOD.items():
        d = curve(f'{method}_{track}')
        if len(d):
            ax.plot(d.lead_h, d.rmse_m, color=colour, lw=1.6, marker=marker, ms=4, markevery=MARK_EVERY, label=name)
    for model, colour, style, name in [('Dense stage 2', INK, '-', 'Dense stage 2'),
                                       ('UTide standalone', UTIDE, (0, (4, 3)), 'UTide')]:
        d = curve(model)
        ax.plot(d.lead_h, d.rmse_m, color=colour, ls=style, lw=1.8, label=name)
    ax.set_xlabel('Lead time (h)')
    ax.set_xticks([0, 24, 48, 72])
    ax.text(0.03, 0.97, TRACK_LABEL[track], transform=ax.transAxes, va='top', fontsize=9, color=ANNOTATION_INK)
axes[0].set_ylabel('Test RMSE (m)')
axes[0].set_ylim(bottom=0)
mark_better(axes[1])
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=8)
save_fig(fig, 'baselines_rmse_vs_lead', width_in=6.3, height_in=3.9, rect=(0, 0.13, 1, 1))
plt.close(fig)

# %% Figure 2: validation RMSE against history ---------------------------------
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6.3, 3.3))
for ax, track in zip(axes, C.BASELINE_TRACKS):
    for method, column in [('ar', 'params_order'), ('gp', 'params_history'), ('ridge', 'params_history')]:
        frames = []
        for tag in ('', '_x', '_parity', '_parity_x'):
            path = OUT / 'search' / f'search_{method}_{track}{tag}.csv'
            if path.exists():
                t = pd.read_csv(path)
                t = t[t.state == 'COMPLETE']
                if column in t and t[column].notna().any():
                    frames.append(t[[column, 'value']].rename(columns={column: 'history'}))
                elif len(t):
                    frames.append(pd.DataFrame({'history': C.LOOKBACK, 'value': t['value']}))
        if frames:
            best = pd.concat(frames).groupby('history')['value'].min()
            name, colour, marker = METHOD[method]
            ax.plot(best.index / 144, np.sqrt(best.values), color=colour, marker=marker, ms=5, lw=1.6, label=name)
    ax.axvline(C.LOOKBACK / 144, color=ANNOTATION_INK, lw=0.8, ls=':')
    ax.set_xscale('log')
    ax.set_xlabel('History used (days)')
    ax.text(0.97, 0.97, TRACK_LABEL[track], transform=ax.transAxes, va='top', ha='right', fontsize=9,
            color=ANNOTATION_INK)
axes[0].set_ylabel('Validation RMSE, all leads (m)')
axes[0].legend(frameon=False, fontsize=8, loc='lower left')
mark_better(axes[1])
save_fig(fig, 'baselines_rmse_vs_history', width_in=6.3, height_in=3.3)
plt.close(fig)

# %% Figure 3: bootstrap difference --------------------------------------------
boot = pd.read_csv(OUT / 'baseline_bootstrap.csv')
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6.3, 3.2))
for ax, track in zip(axes, C.BASELINE_TRACKS):
    d = boot[boot.track == track].sort_values('lead_h')
    method = winners[track].split('_')[0]
    name, colour, _ = METHOD[method]
    ax.fill_between(d.lead_h, d.ci_low, d.ci_high, color=colour, alpha=0.25, lw=0)
    ax.plot(d.lead_h, d.rmse_diff_m, color=colour, lw=1.8, label=f'{name}, 95% interval')
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel('Lead time (h)')
    ax.set_xticks([0, 24, 48, 72])
    ax.legend(frameon=False, fontsize=8, loc='upper left', title=TRACK_LABEL[track], title_fontsize=8)
axes[0].set_ylabel('RMSE difference (m)\nbaseline minus dense stage 2')
mark_better(axes[1])
save_fig(fig, 'baselines_bootstrap_diff', width_in=6.3, height_in=3.2)
plt.close(fig)

# %% Figure 4: one 72h forecast -----------------------------------------------
ex = pd.read_csv(OUT / 'baseline_72h_example.csv', parse_dates=['issue_time', 'valid_time'])
obs = pd.read_csv(OUT / 'baseline_72h_example_observed.csv', parse_dates=['DateTime'])
issue = ex['issue_time'].iloc[0]
fig, ax = plt.subplots(figsize=(6.3, 3.6))
ax.plot(obs['DateTime'], obs['Observed_ODN'], color=INK, lw=1.1, label='Observed')
ax.plot(obs['DateTime'], obs['utide_recon'], color=UTIDE, lw=1.0, ls=(0, (4, 3)), label='UTide')
d = ex[ex.model == 'Dense stage 2'].sort_values('lead_h')
ax.plot(d.valid_time, d.forecast_m, color='#d62728', lw=1.8, label='Dense stage 2', zorder=3)
for track in C.BASELINE_TRACKS:
    name, colour, _ = METHOD[winners[track].split('_')[0]]
    d = ex[ex.model == winners[track]].sort_values('lead_h')
    ax.plot(d.valid_time, d.forecast_m, color=colour, lw=1.6, ls='-' if track == 'residual' else (0, (2, 1)),
            label=f"{name} ({TRACK_LABEL[track].lower()})", zorder=3)
ax.axvline(issue, color=ANNOTATION_INK, lw=0.8, ls=':')
ax.set_ylabel('Water level (m ODN)')
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
ax.tick_params(axis='x', labelsize=7.5)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=8)
save_fig(fig, 'baselines_72h_forecast', width_in=6.3, height_in=3.9)
plt.close(fig)

# %% Figure 5: the baselines against each other ---------------------------------
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6.3, 3.9))
for ax, track in zip(axes, C.BASELINE_TRACKS):
    for method, (name, colour, marker) in METHOD.items():
        d = curve(f'{method}_{track}')
        ax.plot(d.lead_h, d.rmse_m, color=colour, lw=1.6, marker=marker, ms=4, markevery=MARK_EVERY, label=name)
    d = curve('UTide standalone')
    ax.plot(d.lead_h, d.rmse_m, color=UTIDE, ls=(0, (4, 3)), lw=1.8, label='UTide')
    ax.set_xlabel('Lead time (h)')
    ax.set_xticks([0, 24, 48, 72])
    ax.text(0.03, 0.97, TRACK_LABEL[track], transform=ax.transAxes, va='top', fontsize=9, color=ANNOTATION_INK)
axes[0].set_ylabel('Test RMSE (m)')
axes[0].set_ylim(bottom=0)
mark_better(axes[1])
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=8)
save_fig(fig, 'baselines_rmse_classical', width_in=6.3, height_in=3.9, rect=(0, 0.13, 1, 1))
plt.close(fig)

# %% The closeness check and the horizon table ------------------------------------
dense = pd.read_csv(paths.dense_output_dir(GAUGE) / 'dense_metrics.csv')
neural = {m: d.sort_values('lead_h') for m, d in dense[dense.model.str.startswith('Dense')].groupby('model')}
best_nn = min(neural, key=lambda m: neural[m].rmse_m.mean())
nn = neural[best_nn]
nn_72, nn_mean = float(nn.rmse_m.iloc[-1]), float(nn.rmse_m.mean())
close = []
for track in C.BASELINE_TRACKS:
    for method in METHOD:
        d = curve(f'{method}_{track}')
        ratio_72, ratio_mean = d.rmse_m.iloc[-1] / nn_72, d.rmse_m.mean() / nn_mean
        close.append({'model': f'{method}_{track}', 'best_nn': best_nn, 'ratio_72h': ratio_72, 'ratio_mean': ratio_mean,
                      'growth_24h_to_72h': d.set_index('horizon').rmse_m['72h'] / d.set_index('horizon').rmse_m['24h'],
                      'close': bool(ratio_72 <= CLOSE_RATIO and ratio_mean <= CLOSE_RATIO)})
close = pd.DataFrame(close)
close.to_csv(OUT / 'baseline_close_to_nn.csv', index=False)
print(f"best NN {best_nn}; close to it: {close[close.close].model.tolist()}")

table = pd.concat([metrics, dense[dense.model == 'Dense stage 3 (tide-only)'].assign(family='neural')])
table = table[table.horizon.isin(REPORT_LEADS)].pivot_table(index='model', columns='horizon', values='rmse_m')[REPORT_LEADS]
table.round(3).to_csv(OUT / 'baseline_horizon_table.csv')

# %% Figure 6: the close ones against the best neural model -----------------------
if close.close.any():
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    ax.plot(nn.lead_h, nn.rmse_m, color=INK, lw=2.0, label=best_nn)
    d = curve('UTide standalone')
    ax.plot(d.lead_h, d.rmse_m, color=UTIDE, ls=(0, (4, 3)), lw=1.8, label='UTide')
    for model in close[close.close].model:
        method, track = model.split('_')
        name, colour, marker = METHOD[method]
        d = curve(model)
        ax.plot(d.lead_h, d.rmse_m, color=colour, lw=1.6, marker=marker, ms=4, markevery=MARK_EVERY,
                ls='-' if track == 'residual' else (0, (2, 1)), label=f'{name} ({TRACK_LABEL[track].lower()})')
    ax.set_xlabel('Lead time (h)')
    ax.set_xticks([0, 24, 48, 72])
    ax.set_ylabel('Test RMSE (m)')
    ax.set_ylim(bottom=0)
    mark_better(ax)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    save_fig(fig, 'baselines_rmse_vs_best_nn', width_in=6.3, height_in=3.6)
    plt.close(fig)

# %% Figure 7: one forecast per panel ---------------------------------------------
panels = [('Dense stage 2', 'Dense stage 2', INK)] + [(f'{m}_{t}', f'{METHOD[m][0]}, {t}', METHOD[m][1])
                                                      for m in METHOD for t in C.BASELINE_TRACKS]
rows_n = int(np.ceil(len(panels) / 2))
fig, axes = plt.subplots(rows_n, 2, sharex=True, sharey=True, figsize=(6.3, 1.35 * rows_n))
for ax in axes.flat:
    ax.axis('off')
for ax, (model, name, colour) in zip(axes.flat, panels):
    ax.axis('on')
    d = ex[ex.model == model].sort_values('lead_h')
    ax.plot(obs['DateTime'], obs['Observed_ODN'], color='#8a8986', lw=0.8, label='Observed')
    ax.plot(d.valid_time, d.forecast_m, color=colour if colour != INK else '#d62728', lw=1.2, zorder=3)
    ax.axvline(issue, color=ANNOTATION_INK, lw=0.6, ls=':')
    ax.set_title(name, fontsize=8, loc='left', pad=2)
    ax.tick_params(labelsize=7, labelbottom=True)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
fig.supylabel('Water level (m ODN)', fontsize=9)
save_fig(fig, 'baselines_forecast_panels', width_in=6.3, height_in=1.35 * rows_n)
plt.close(fig)

print('\nDone.')
