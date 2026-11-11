"""Figures for the November 2026 supervisor update (neural model only). No training.

    python models/plot_supervisor_update.py

Writes PNGs to reports/2026-11-nov-fixes/figures/:
    <gauge>_rmse_vs_lead.png          test RMSE against lead, 10 min to 72h: harmonic analysis
                                      alone and the chosen dense model (every 10-min lead, from
                                      outputs/[richmond/]dense_v2_skip_wloss/postprocess/test_rmse_433_leads.csv),
                                      and the dissertation's fixed-horizon model (sparse stage 2) at its
                                      own leads, on the anchors it shares with the dense model
                                      (dense_vs_sparse_matched.csv)
    <gauge>_forecast_{typical,20220217}.png
                                      one 72h forecast every 10 min against observations. '20220217' is
                                      issued 2022-02-17 12:00 (the dissertation's example date); 'typical' is the 12:00 issue
                                      whose 72h RMSE is the test median (rule in plot_supervisor_figures.py)

No classical models are drawn. Test data: the chosen dense model is frozen and was already scored
on test (CHANGELOG 2026-11-10); these figures re-draw those frozen forecasts.
"""
import pathlib
import sys

_start = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / '_')).resolve()
REPO_ROOT = next(p for p in _start.parents if (p / 'utils' / 'paths.py').exists())
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'models'))

import matplotlib
matplotlib.use('Agg')

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import paths
from utils.models import config as C
import postprocess_hw as P
import plot_supervisor_figures as S

VARIANT = S.VARIANT
FIG_DIR = S.FIG_DIR
LABEL = S.LABEL
NN, DISS = '#2a78d6', '#eb6834'   # validated palette slots (blue, orange)
HARM, OBS = '0.5', '0.1'
DPI = 150
SHARED = {'10min': 1 / 6, '1h': 1, '6h': 6, '24h': 24, '48h': 48, '72h': 72}

plt.rcParams.update({'font.size': 11.5, 'axes.titlesize': 12.5, 'axes.labelsize': 12, 'legend.fontsize': 11,
                     'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': '0.9', 'grid.linewidth': 0.6,
                     'legend.frameon': False})


def fig_rmse(gauge):
    d = pd.read_csv(paths.dense_output_dir(gauge, VARIANT) / 'postprocess' / 'test_rmse_433_leads.csv')
    m = pd.read_csv(paths.dense_output_dir(gauge, VARIANT) / 'dense_vs_sparse_matched.csv')
    sp = m[m.model == 'Sparse stage 2'].set_index('horizon').loc[list(SHARED)]
    harm = d[d.model == 'UTide alone']
    nn = d[d.model == 'dense, cubic interp']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={'width_ratios': [1, 1.6]})
    for ax, hi in zip(axes, (6, 72)):
        for src, kw in ((harm, dict(color=HARM, ls=':', lw=2, label='Harmonic analysis alone')),
                        (nn, dict(color=NN, lw=2.2, label='New continuous model (every 10 min)'))):
            x = src.lead_min / 60
            sel = (x > 0) & (x <= hi)
            ax.plot(x[sel], src.rmse_m[sel] * 100, **kw)
        lead = np.array([SHARED[h] for h in sp.index])
        keep = lead <= hi
        ax.plot(lead[keep], sp.rmse_m.to_numpy()[keep] * 100, 'D', color=DISS, ms=8, mec='white', mew=1.2,
                label='Dissertation model (fixed horizons)')
        ax.set_xlim(0, hi)
        ax.set_ylim(0, None)
        ax.set_xlabel('Forecast lead time (hours)')
        ax.set_ylabel('Test RMSE (cm)')
        ax.set_title('First 6 hours' if hi == 6 else 'Whole 72 hours', loc='left')
        if hi == 72:
            ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    fig.suptitle(f'{LABEL[gauge]}: forecast error against lead time (test, 2021 to 2024)', x=0.01, ha='left',
                 fontsize=13.5)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG_DIR / f'{gauge}_rmse_vs_lead.png', dpi=DPI)
    plt.close(fig)
    print('wrote', f'{gauge}_rmse_vs_lead.png')


def fig_example(gauge, s, anchors, preds, t, issue, tag, what):
    i = int(np.where(t == issue.astype('datetime64[ns]'))[0][0])
    a = anchors[i:i + 1]
    nn_c = P.trajectories(s, a, preds[i:i + 1], 'cubic')[0]
    nn_n = P.trajectories(s, a, preds[i:i + 1], 'native')[0]
    span = np.arange(a[0] - 36, a[0] + 433)
    tt = s.time[span].astype('datetime64[m]').astype(object)
    tf = s.time[a[0] + P.FULL_STEPS].astype('datetime64[m]').astype(object)
    obs = np.where(s.flagged[span], np.nan, s.level[span])
    harm_span, harm = s.utide[span], s.utide[a[0] + P.FULL_STEPS]
    t0 = s.time[a[0]].astype('datetime64[m]').astype(object)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True, gridspec_kw={'height_ratios': [1.6, 1]})
    ax1.plot(tt, obs, color=OBS, lw=1.5, label='Observed')
    ax1.plot(tf, harm, color=HARM, lw=1.5, ls=':', label='Harmonic analysis alone')
    ax1.plot(tf, nn_c, color=NN, lw=1.7, label='New model, every 10 min')
    ax1.plot(tf, nn_n, 'o', color=NN, ms=3.2, label='New model, its hourly outputs')
    ax1.set_ylabel('Water level (m ODN)')
    when = pd.Timestamp(issue).strftime('%d %B %Y, %H:%M').lstrip('0')
    ax1.set_title(f'{LABEL[gauge]}: 72h forecast issued {when} ({what})', loc='left')
    ax2.axhline(0, color=HARM, lw=1.3, ls=':')
    ax2.plot(tt, obs - harm_span, color=OBS, lw=1.5)
    ax2.plot(tf, nn_c - harm, color=NN, lw=1.7)
    ax2.set_ylabel('Minus harmonic\nprediction (m)')
    for ax in (ax1, ax2):
        ax.axvline(t0, color='0.4', lw=1)
    ax1.annotate('forecast issued', (t0, 1), xycoords=('data', 'axes fraction'), xytext=(4, -14),
                 textcoords='offset points', fontsize=10, color='0.3')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG_DIR / f'{gauge}_forecast_{tag}.png', dpi=DPI)
    plt.close(fig)
    print('wrote', f'{gauge}_forecast_{tag}.png', issue)


if __name__ == '__main__':
    for gauge in C.GAUGES:
        fig_rmse(gauge)
        s, anchors, preds, t = S.load(gauge)
        fig_example(gauge, s, anchors, preds, t, S.ISSUE, '20220217', 'dissertation example date')
        typ, e = S.typical_issue(s, anchors, preds)
        print(gauge, 'typical issue', typ, f'72h RMSE {e * 100:.1f} cm')
        fig_example(gauge, s, anchors, preds, t, typ, 'typical', 'typical, median error')
