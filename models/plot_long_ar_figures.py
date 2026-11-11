"""Figures for reports/2026-11-nov-fixes/04_long_ar_explained.md. No training; frozen configs only.

    python models/plot_long_ar_figures.py

Writes to reports/2026-11-nov-fixes/figures/:
    L1_ar_level_vs_residual_by_lead.png   test RMSE by lead: the long AR on the water level itself
                                          against the long AR on the harmonic residual, each also
                                          at 16h of history, plus the harmonic prediction alone and
                                          (Southend) the dissertation's stage 2 network without the
                                          tide input (its sparse leads)
    L2_ar_level_forecast_<gauge>.png      one 72h forecast every 10 min from both AR versions, on
                                          the 'typical' issue times of plot_supervisor_figures.py
    L3_ar_weights_<gauge>.png             the AR's weights on past values for a 1h and a 24h forecast
and the numbers behind L1 to outputs/diagnostics_2026-11/long_ar_L1_by_lead.csv.

Test data: all configs are frozen and were already scored on test by eval_baselines (disclosed in
CHANGELOG.md); this script reads those tables and draws single test forecasts.
"""
import pathlib
import sys

_start = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / '_')).resolve()
REPO_ROOT = next(p for p in _start.parents if (p / 'utils' / 'paths.py').exists())
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use('Agg')

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import paths
from utils.models import baselines as B
from utils.models import config as C

FIG_DIR = paths.ensure_dir(REPO_ROOT / 'reports' / '2026-11-nov-fixes' / 'figures')
OUT = paths.ensure_dir(paths.OUTPUTS_DIR / 'diagnostics_2026-11')
LABEL = {'southend': 'Southend Pier', 'richmond': 'Richmond Lock'}
TYPICAL = {'southend': np.datetime64('2023-05-25T12:00'), 'richmond': np.datetime64('2024-02-19T12:00')}
STEPS = np.arange(1, 433)
RESID, LEVEL, NN = '#eb6834', '#1baf7a', '#2a78d6'   # validated palette slots (orange, aqua, blue)
HARM, OBS = '0.55', '0.1'
DPI = 170
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '0.9', 'grid.linewidth': 0.6, 'legend.frameon': False})


def cfgs(gauge):
    return json.loads(paths.baseline_config_path(gauge).read_text(encoding='utf-8'))


def history_label(order):
    days = order / C.STEPS_PER_DAY
    return f'{order * 10 / 60:.0f} h' if days < 1 else f'{days:.0f} d'


def fig_by_lead():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    rows = []
    for ax, gauge in zip(axes, C.GAUGES):
        b = pd.read_csv(paths.baseline_output_dir(gauge) / 'baseline_metrics.csv')
        c = cfgs(gauge)
        series = [('ar_residual', f"AR on residual, {history_label(c['ar_residual']['config']['order'])}", RESID, '-'),
                  ('ar_level', f"AR on water level, {history_label(c['ar_level']['config']['order'])}", LEVEL, '-'),
                  ('ar_residual_parity', 'AR on residual, 16 h', RESID, '--'),
                  ('ar_level_parity', 'AR on water level, 16 h', LEVEL, '--'),
                  ('UTide standalone', 'Harmonic analysis alone', HARM, ':')]
        for key, name, col, ls in series:
            d = b[b.model == key].sort_values('lead_h')
            ax.plot(d.lead_h, d.rmse_m * 100, color=col, ls=ls, lw=1.8 if key != 'UTide standalone' else 1.4, label=name)
            rows += [{'gauge': gauge, 'model': name, 'horizon': r.horizon, 'lead_h': r.lead_h, 'test_rmse_cm': r.rmse_m * 100}
                     for r in d.itertuples()]
        if gauge == 'southend':
            a = pd.read_csv(paths.OUTPUTS_DIR / 'stage2_outputs' / 'stage2_ablation_metrics.csv')
            a = a[a.model == 'Stage-2 (no tide, ablation)']
            lead = a.horizon.map({'10min': 1 / 6, '1h': 1, '6h': 6, '24h': 24, '48h': 48, '72h': 72, '168h': 168})
            keep = lead <= 72
            ax.plot(lead[keep], a.rmse_m[keep] * 100, 'D', color=NN, ms=5,
                    label='Neural, water level only, 16 h (dissertation)')
            rows += [{'gauge': gauge, 'model': 'Neural, water level only, 16 h (dissertation)', 'horizon': h,
                      'lead_h': l, 'test_rmse_cm': r * 100} for h, l, r in zip(a.horizon[keep], lead[keep], a.rmse_m[keep])]
        ax.set_xlim(0, 72); ax.set_xticks([0, 12, 24, 36, 48, 60, 72]); ax.set_ylim(0, None)
        ax.set_xlabel('Forecast lead time (hours)'); ax.set_ylabel('Test RMSE (cm)')
        ax.set_title(LABEL[gauge], loc='left', fontsize=10)
        ax.legend(fontsize=7, loc='upper left' if gauge == 'southend' else 'lower right')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'L1_ar_level_vs_residual_by_lead.png', dpi=DPI)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / 'long_ar_L1_by_lead.csv', index=False)
    print('wrote L1')


def fit(gauge, s, key):
    c = cfgs(gauge)[key]['config']
    track = 'level' if key.startswith('ar_level') else 'residual'
    y_train = s.track(track)[:s.n_train]
    phi = B.fit_ar(y_train, c['order'], c['estimator'], c.get('jitter', 1e-5))
    return phi, B.linear_from_pi(phi, y_train, steps=STEPS), track


def fig_example(gauge, s):
    a = np.array([int(np.searchsorted(s.time, TYPICAL[gauge]))])
    assert s.time[a[0]] == TYPICAL[gauge]
    preds = {}
    for key in ('ar_residual', 'ar_level'):
        _, model, track = fit(gauge, s, key)
        f = model.forecast(s, track, a, steps=STEPS)[0]
        preds[key] = f + (s.utide[a[0] + STEPS] if track == 'residual' else 0.0)
    span = np.arange(a[0] - 36, a[0] + 433)
    tt = s.time[span].astype('datetime64[m]').astype(object)
    tf = s.time[a[0] + STEPS].astype('datetime64[m]').astype(object)
    c = cfgs(gauge)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(tt, np.where(s.flagged[span], np.nan, s.level[span]), color=OBS, lw=1.3, label='Observed')
    ax.plot(tf, s.utide[a[0] + STEPS], color=HARM, ls=':', lw=1.2, label='Harmonic analysis alone')
    ax.plot(tf, preds['ar_residual'], color=RESID, lw=1.4,
            label=f"AR on residual ({history_label(c['ar_residual']['config']['order'])}) + harmonic")
    ax.plot(tf, preds['ar_level'], color=LEVEL, lw=1.4,
            label=f"AR on water level only ({history_label(c['ar_level']['config']['order'])}), no harmonic analysis")
    ax.axvline(s.time[a[0]].astype('datetime64[m]').astype(object), color='0.4', lw=0.9)
    ax.set_ylabel('Water level (m ODN)')
    when = pd.Timestamp(TYPICAL[gauge]).strftime('%d %B %Y, %H:%M')
    ax.set_title(f'{LABEL[gauge]}: one 72h forecast issued {when} (test period)', loc='left', fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(FIG_DIR / f'L2_ar_level_forecast_{gauge}.png', dpi=DPI)
    plt.close(fig)
    print('wrote L2', gauge)


def fig_weights(gauge, s):
    _, model, _ = fit(gauge, s, 'ar_residual')
    days = model.lags / C.STEPS_PER_DAY
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    for ax, (lead_steps, name) in zip(axes, [(6, '1 hour ahead'), (144, '24 hours ahead')]):
        w = model.weights[lead_steps - 1]
        ax.plot(days, w, color=RESID, lw=0.7)
        ax.axhline(0, color='0.6', lw=0.6)
        ax.set_title(f'{LABEL[gauge]}: weights for the forecast {name} (sum {w.sum():.2f})', loc='left', fontsize=9.5)
        ax.set_ylabel('Weight')
        ax.set_xscale('log')
    axes[1].set_xlabel('How far back the past value is (days, log scale)')
    fig.tight_layout()
    fig.savefig(FIG_DIR / f'L3_ar_weights_{gauge}.png', dpi=DPI)
    plt.close(fig)
    print('wrote L3', gauge)


if __name__ == '__main__':
    fig_by_lead()
    for gauge in C.GAUGES:
        s = B.load_series(gauge, verbose=False)
        fig_example(gauge, s)
        fig_weights(gauge, s)
