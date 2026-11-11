"""Figures for the November 2026 supervisor summary. No training; frozen configs only.

    python models/plot_supervisor_figures.py

Writes PNGs to reports/2026-11-nov-fixes/figures/:
    S1_ar_history_length.png        validation RMSE of the AR surge model against how much past
                                    record it uses (the Optuna search's own trials), with the new
                                    neural model for reference
    S2_same_history_by_lead.png     test RMSE by lead: the new neural model against the classical
                                    models held to the same 16h of history, plus the full-history AR
    S3_continuous_forecast_{storm,typical}_<g>.png
                                    one 72h forecast, every 10 min: neural (hourly output, cubic
                                    interpolation of its correction to the harmonic prediction) and
                                    AR (forecast directly every 10 min). 'storm' is issued
                                    2022-02-17 12:00 (the dissertation's example date, Storm Eunice);
                                    'typical' is the 12:00 issue whose neural 72h RMSE is the test median
    S4_continuous_rmse_by_lead.png  test RMSE at every 10-min lead 0 to 72h, hourly anchors
Also writes the numbers behind S1, S2 and S4 to outputs/diagnostics_2026-11/supervisor_figs_*.csv.

Test data: the neural model (dense_v2_skip_wloss) and the AR configs are frozen and already scored
on test; S3 and S4 re-use those frozen forecasts and are disclosed in CHANGELOG.md.
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
from utils.models import baselines as B
from utils.models import config as C
import postprocess_hw as P

VARIANT = 'dense_v2_skip_wloss'
FIG_DIR = paths.ensure_dir(REPO_ROOT / 'reports' / '2026-11-nov-fixes' / 'figures')
OUT = paths.ensure_dir(paths.OUTPUTS_DIR / 'diagnostics_2026-11')
ISSUE = np.datetime64('2022-02-17T12:00')
LABEL = {'southend': 'Southend Pier', 'richmond': 'Richmond Lock'}
# validated with the dataviz palette checker (CVD and normal vision pass; contrast relief = legends)
NN, AR, RIDGE, GP = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
HARM, OBS = '0.55', '0.1'
DPI = 170

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '0.9', 'grid.linewidth': 0.6, 'legend.frameon': False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=DPI)
    plt.close(fig)
    print('wrote', FIG_DIR / name)


def nn_val_sqrt_mean_mse(gauge):
    r = pd.read_csv(OUT / f'val_vs_ar_with_parity_{gauge}.csv', index_col=0)
    return float(np.sqrt((r[VARIANT] ** 2).mean()))


# ---------------------------------------------------------------- S1: history length
def fig_history():
    labels = {96: '16 h', 449: '3 d', 2160: '15 d', 4320: '30 d', 8640: '60 d', 13000: '90 d',
              17280: '120 d', 26000: '180 d', 52560: '1 yr'}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    rows = []
    for ax, gauge in zip(axes, C.GAUGES):
        d = pd.read_csv(paths.baseline_output_dir(gauge) / 'search' / 'search_ar_residual_x.csv')
        d = d[d.state == 'COMPLETE']
        best = (np.sqrt(d.groupby('params_order').value.min()) * 100).sort_index()
        x = np.arange(len(best))
        ax.plot(x, best.values, color=AR, lw=2, marker='o', ms=5, label='Autoregressive surge model')
        k = int(best.values.argmin())
        ax.plot(x[k], best.values[k], 'o', ms=11, mfc='none', mec=OBS, mew=1.4)
        ax.annotate('chosen', (x[k], best.values[k]), xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=8, color=OBS)
        nn = nn_val_sqrt_mean_mse(gauge) * 100
        ax.axhline(nn, color=NN, lw=1.8, ls='--', label='Neural model (always 16 h)')
        ax.set_xticks(x, [labels[o] for o in best.index], rotation=0, fontsize=7.5)
        ax.set_xlabel('Past record the model may use')
        ax.set_title(LABEL[gauge], loc='left', fontsize=10)
        ax.set_ylabel('Validation RMSE, 0 to 72h (cm)')
        for o, v in best.items():
            rows.append({'gauge': gauge, 'order_10min_steps': o, 'history': labels[o], 'val_rmse_cm': v})
        rows.append({'gauge': gauge, 'order_10min_steps': 96, 'history': 'neural 16 h', 'val_rmse_cm': nn})
    axes[0].legend(loc='upper right', fontsize=8)
    save(fig, 'S1_ar_history_length.png')
    pd.DataFrame(rows).to_csv(OUT / 'supervisor_figs_S1_history.csv', index=False)


# ---------------------------------------------------------------- S2: same history, by lead
def fig_same_history():
    series = [('Dense stage 2', 'Neural model (16 h)', NN, '-'),
              ('ar_residual_parity', 'Autoregressive (16 h)', AR, '--'),
              ('ridge_residual_parity', 'Ridge regression (16 h)', RIDGE, '--'),
              ('gp_residual_parity', 'Gaussian process (16 h)', GP, '--'),
              ('ar_residual', 'Autoregressive (30 d / 180 d)', AR, '-'),
              ('UTide standalone', 'Harmonic analysis alone', HARM, ':')]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    rows = []
    for ax, gauge in zip(axes, C.GAUGES):
        bm = pd.read_csv(paths.baseline_output_dir(gauge) / 'baseline_metrics.csv')
        nm = pd.read_csv(paths.dense_output_dir(gauge, VARIANT) / 'dense_metrics.csv')
        for key, name, col, ls in series:
            src = nm if key == 'Dense stage 2' else bm
            d = src[src.model == key].sort_values('lead_h')
            ax.plot(d.lead_h, d.rmse_m * 100, color=col, ls=ls, lw=1.8 if key != 'UTide standalone' else 1.4,
                    label=name)
            for _, r in d.iterrows():
                rows.append({'gauge': gauge, 'model': name, 'horizon': r.horizon, 'lead_h': r.lead_h,
                             'test_rmse_cm': r.rmse_m * 100})
        ax.set_xlim(0, 72); ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
        ax.set_ylim(0, None)
        ax.set_xlabel('Forecast lead time (hours)')
        ax.set_ylabel('Test RMSE (cm)')
        ax.set_title(LABEL[gauge], loc='left', fontsize=10)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(FIG_DIR / 'S2_same_history_by_lead.png', dpi=DPI)
    plt.close(fig)
    print('wrote S2')
    pd.DataFrame(rows).to_csv(OUT / 'supervisor_figs_S2_same_history.csv', index=False)


# ---------------------------------------------------------------- S3 and S4: continuous output
def load(gauge):
    s = B.load_series(gauge, verbose=False)
    z = np.load(paths.dense_output_dir(gauge, VARIANT) / 'dense_test_predictions_stage2.npz')
    t = z['anchors'].astype('datetime64[ns]')
    anchors = np.searchsorted(s.time.astype('datetime64[ns]'), t)
    assert np.array_equal(s.time[anchors].astype('datetime64[ns]'), t)
    return s, anchors, z['preds'].astype(np.float64), t


def typical_issue(s, anchors, preds):
    """Issue time (12:00 issues only) whose neural 72h RMSE over the 73 leads is the median of
    the test period: a forecast of ordinary quality, chosen by rule rather than by eye."""
    tm = s.time[anchors].astype('datetime64[m]').astype(np.int64)
    keep = (tm % (24 * 60)) == 12 * 60
    a, p = anchors[keep], preds[keep]
    Y, F = B.targets(s.level, s.flagged, a)
    ok = ~F & ~np.isnan(Y)
    full = ok.all(1)
    err = np.sqrt((np.where(ok, p - Y, 0.0) ** 2).sum(1) / ok.sum(1))
    idx = np.where(full)[0]
    j = idx[np.argsort(err[idx])[len(idx) // 2]]
    return s.time[a[j]].astype('datetime64[ns]'), float(err[j])


def fig_example(gauge, s, anchors, preds, t, issue, tag, what):
    i = int(np.where(t == issue.astype('datetime64[ns]'))[0][0])
    a = anchors[i:i + 1]
    nn_c = P.trajectories(s, a, preds[i:i + 1], 'cubic')[0]
    nn_n = P.trajectories(s, a, preds[i:i + 1], 'native')[0]
    ar = P.ar_full_trajectories(gauge, s, a)[0]
    lo = a[0] - 36
    span = np.arange(lo, a[0] + 433)
    tt = s.time[span].astype('datetime64[m]').astype(object)
    tf = s.time[a[0] + P.FULL_STEPS].astype('datetime64[m]').astype(object)
    obs = np.where(s.flagged[span], np.nan, s.level[span])
    harm_span, harm = s.utide[span], s.utide[a[0] + P.FULL_STEPS]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={'height_ratios': [1.6, 1]})
    ax1.plot(tt, obs, color=OBS, lw=1.3, label='Observed')
    ax1.plot(tf, harm, color=HARM, lw=1.2, ls=':', label='Harmonic analysis alone')
    ax1.plot(tf, ar, color=AR, lw=1.4, label='Autoregressive, every 10 min')
    ax1.plot(tf, nn_c, color=NN, lw=1.4, label='Neural, hourly output joined by a smooth curve')
    ax1.plot(tf, nn_n, 'o', color=NN, ms=2.6, label='Neural, the hourly values it actually outputs')
    ax1.set_ylabel('Water level (m ODN)')
    when = pd.Timestamp(issue).strftime('%d %B %Y, %H:%M').lstrip('0')
    ax1.set_title(f'{LABEL[gauge]}: one 72h forecast issued {when} ({what})', loc='left', fontsize=10)

    ax2.axhline(0, color=HARM, lw=1.0, ls=':')
    ax2.plot(tt, obs - harm_span, color=OBS, lw=1.3)
    ax2.plot(tf, ar - harm, color=AR, lw=1.4)
    ax2.plot(tf, nn_c - harm, color=NN, lw=1.4)
    ax2.set_ylabel('Difference from\nharmonic (m)')
    for ax in (ax1, ax2):
        ax.axvline(s.time[a[0]].astype('datetime64[m]').astype(object), color='0.4', lw=0.9)
    ax1.annotate('forecast issued', (s.time[a[0]].astype('datetime64[m]').astype(object), ax1.get_ylim()[0]),
                 xytext=(3, 3), textcoords='offset points', fontsize=7.5, color='0.35')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG_DIR / f'S3_continuous_forecast_{tag}_{gauge}.png', dpi=DPI)
    plt.close(fig)
    print('wrote S3', tag, gauge, issue)


def rmse_curves(gauge, s, anchors, preds):
    a_all, p_all = anchors[::P.HW_ANCHOR_STRIDE], preds[::P.HW_ANCHOR_STRIDE]
    names = ['harmonic', 'neural_native', 'neural_cubic', 'ar']
    sse = {k: np.zeros(433) for k in names}
    n = {k: np.zeros(433) for k in names}
    for lo in range(0, len(a_all), 4000):
        a, p = a_all[lo:lo + 4000], p_all[lo:lo + 4000]
        Y, F = B.targets(s.level, s.flagged, a, steps=P.FULL_STEPS)
        ok = ~F & ~np.isnan(Y)
        tr = {'harmonic': P.trajectories(s, a, p, 'utide'), 'neural_native': P.trajectories(s, a, p, 'native'),
              'neural_cubic': P.trajectories(s, a, p, 'cubic'), 'ar': P.ar_full_trajectories(gauge, s, a)}
        for k, x in tr.items():
            m = ok & ~np.isnan(x)
            sse[k] += (np.where(m, x - Y, 0.0) ** 2).sum(0)
            n[k] += m.sum(0)
    df = pd.DataFrame({k: np.sqrt(sse[k] / np.maximum(n[k], 1)) * 100 for k in names})
    df.loc[df.neural_native.index.difference(np.r_[0, B.DENSE_STEPS]), 'neural_native'] = np.nan
    df.insert(0, 'lead_min', P.FULL_STEPS * 10)
    df['n_anchors'] = n['ar']
    df['gauge'] = gauge
    return df


def fig_rmse(curves):
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.8), gridspec_kw={'width_ratios': [1, 1.6]})
    for row, gauge in enumerate(C.GAUGES):
        d = curves[curves.gauge == gauge]
        for col, (lo, hi) in enumerate([(0, 6), (0, 72)]):
            ax = axes[row, col]
            x = d.lead_min / 60
            sel = (x >= lo) & (x <= hi)
            ax.plot(x[sel], d.harmonic[sel], color=HARM, ls=':', lw=1.4, label='Harmonic analysis alone')
            ax.plot(x[sel], d.ar[sel], color=AR, lw=1.6, label='Autoregressive, every 10 min')
            ax.plot(x[sel], d.neural_cubic[sel], color=NN, lw=1.6, label='Neural, smooth curve between hours')
            nat = d[sel & d.neural_native.notna()]
            ax.plot(nat.lead_min / 60, nat.neural_native, 'o', color=NN, ms=3 if col == 0 else 2,
                    label='Neural, its own hourly outputs')
            ax.set_xlim(lo, hi)
            ax.set_ylim(0, None)
            ax.set_ylabel('Test RMSE (cm)')
            ax.set_title(f"{LABEL[gauge]}: {'first 6 hours' if col == 0 else 'whole 72 hours'}", loc='left', fontsize=9.5)
            if row == 1:
                ax.set_xlabel('Forecast lead time (hours)')
            if col == 1:
                ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG_DIR / 'S4_continuous_rmse_by_lead.png', dpi=DPI)
    plt.close(fig)
    print('wrote S4')


if __name__ == '__main__':
    fig_history()
    fig_same_history()
    curves = []
    for gauge in C.GAUGES:
        s, anchors, preds, t = load(gauge)
        fig_example(gauge, s, anchors, preds, t, ISSUE, 'storm', 'Storm Eunice, test period')
        typ, e = typical_issue(s, anchors, preds)
        print(gauge, 'typical issue', typ, f'neural 72h RMSE {e * 100:.1f} cm')
        fig_example(gauge, s, anchors, preds, t, typ, 'typical', 'a typical forecast, test period')
        curves.append(rmse_curves(gauge, s, anchors, preds))
    curves = pd.concat(curves)
    curves.to_csv(OUT / 'supervisor_figs_S4_rmse_433.csv', index=False)
    fig_rmse(curves)
