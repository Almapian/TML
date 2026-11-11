"""Diagnostic A (Nov 2026): bias decomposition of UTide, dense stage 2 and the AR residual baseline.

No training. Inputs:
  - the train-only UTide reconstruction cached on the 10-min grid by utils/models/baselines.py
  - dense stage 2 predictions saved by train_dense.py (dense_{val,test}_predictions_stage2.npz)
  - AR residual baseline: test predictions saved by eval_baselines.py; validation predictions
    regenerated here from the frozen best_configs.json through eval_baselines' own code path
    (deterministic Yule-Walker on train), after checking it reproduces the saved test predictions.

Error convention: e = forecast - observed. bias = mean(e), std = population std of e, so
RMSE^2 = bias^2 + std^2 exactly. Flagged or missing targets are excluded (same masks as scoring).
The UTide residual is observed - UTide = -e(UTide).

    python models/diag_bias_decomposition.py

Outputs: outputs/diagnostics_2026-11/  (tables)   and   reports/2026-11-nov-fixes/figures/ (PNGs)
Test-period numbers are computed for models whose test results were already published.
"""
# %% Bootstrap
import pathlib
import sys

_start = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / '_')).resolve()
REPO_ROOT = next(p for p in _start.parents if (p / 'utils' / 'paths.py').exists())
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use('Agg')

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import paths
from utils.models import baselines as B
from utils.models import config as C
from utils.plot_config import ANNOTATION_INK, save_fig, set_report_dir

# %% ------------------------------------------------------------------ CONFIG
GAUGES = ['southend', 'richmond']
AR_NAME = 'ar_residual'                 # the best AR residual baseline (non-parity) in best_configs.json
REPRO_EVERY = 25                        # test anchors re-forecast to check the AR reproduction
REPRO_TOL = 1e-4                        # metres (saved predictions are float32)
LEAD_NAMES = list(C.DENSE_HORIZONS)
LEAD_H = np.array([s / 6 for s in C.DENSE_HORIZONS.values()])
I72 = LEAD_NAMES.index('72h')

OUT = paths.ensure_dir(paths.OUTPUTS_DIR / 'diagnostics_2026-11')
FIG_DIR = paths.ensure_dir(REPO_ROOT / 'reports' / '2026-11-nov-fixes' / 'figures')
set_report_dir(str(FIG_DIR))

INK = '#0b0b0b'
COL = {'UTide': '#52514e', 'AR residual': '#2a78d6', 'Dense stage 2': '#eb6834'}
SPLIT_SHADE = {'train': '#f2f2f2', 'val': '#dbe8f7', 'test': '#fbe3d6'}
C.configure_threads()


def decompose(e, mask, axis=0):
    """bias, std, rmse, n of e over the unmasked entries (masked = excluded)."""
    e = np.where(mask, e, np.nan)
    n = np.sum(mask, axis=axis)
    bias = np.nanmean(e, axis=axis)
    rmse = np.sqrt(np.nanmean(e ** 2, axis=axis))
    std = np.sqrt(np.maximum(rmse ** 2 - bias ** 2, 0))
    return bias, std, rmse, n


def slow_component(e, mask, months):
    """Mean over samples of the squared calendar-month mean error, per lead: the part of the
    MSE carried by an offset that is constant within each month but can change between months."""
    out = np.zeros(e.shape[1])
    for j in range(e.shape[1]):
        df = pd.DataFrame({'e': e[mask[:, j], j], 'm': months[mask[:, j]]})
        mmean = df.groupby('m')['e'].transform('mean').to_numpy()
        out[j] = np.mean(mmean ** 2)
    return out


utide_year_rows, lead_rows, gap_rows, monthly_rows = [], [], [], []
fig_data = {}

for gauge in GAUGES:
    label = C.gauge_info(gauge)['label']
    print(f"\n===== {label}")
    s = B.load_series(gauge, verbose=False)
    valid = ~np.isnan(s.level) & ~s.flagged
    years = s.time.astype('datetime64[Y]').astype(int) + 1970
    split_of = np.where(s.time < np.datetime64(C.TRAIN_END), 'train',
                        np.where(s.time < np.datetime64(C.VAL_END), 'val', 'test'))

    # ---- 1. UTide alone, every valid 10-min point, per split and per calendar year
    e_ut = s.utide - s.level
    for key, sel in [(sp, split_of == sp) for sp in ('train', 'val', 'test')] + \
                    [(int(y), years == y) for y in np.unique(years)]:
        m = valid & sel
        b, sd, r, n = decompose(e_ut, m)
        utide_year_rows.append({'gauge': gauge, 'period': key,
                                'split': key if isinstance(key, str) else split_of[np.argmax(sel)],
                                'bias_m': b, 'std_m': sd, 'rmse_m': r, 'mean_residual_m': -b, 'n': int(n)})
    month = pd.Series(np.where(valid, s.level - s.utide, np.nan), index=pd.DatetimeIndex(s.time))
    monthly = month.resample('MS').mean()
    for t, v in monthly.items():
        monthly_rows.append({'gauge': gauge, 'month': t, 'mean_residual_m': v})

    # ---- 2. Dense stage 2 vs AR residual, per lead, on val and test
    best = json.loads(paths.baseline_config_path(gauge).read_text(encoding='utf-8'))[AR_NAME]
    ranges = json.loads((paths.baseline_output_dir(gauge) / 'identification' / 'identified_ranges.json')
                        .read_text(encoding='utf-8'))
    print(f"AR residual config: {best['config']}")
    ar = B.build_model('ar', best['config'], s, 'residual', ranges['residual']['dhr_frequencies'], B.ModelCache())

    fig_data[gauge] = {}
    for split in ('val', 'test'):
        anchors = B.split_anchors(s, split)
        times = s.time[anchors].astype('datetime64[ns]')
        Y, F = B.targets(s.level, s.flagged, anchors)
        U = s.utide[anchors[:, None] + B.DENSE_STEPS[None, :]]
        mask = ~F & ~np.isnan(Y)

        dz = np.load(paths.dense_output_dir(gauge) / f'dense_{split}_predictions_stage2.npz')
        assert np.array_equal(dz['anchors'].astype('datetime64[ns]'), times), f'{split} anchors differ (dense)'
        p_dense = dz['preds'].astype(np.float64)

        if split == 'test':
            az = np.load(paths.baseline_output_dir(gauge) / f'baseline_test_predictions_{AR_NAME}.npz')
            assert np.array_equal(az['anchors'].astype('datetime64[ns]'), times), 'test anchors differ (AR)'
            p_ar = az['preds'].astype(np.float64)
            sub = anchors[::REPRO_EVERY]
            redo = ar.forecast(s, 'residual', sub) + s.utide[sub[:, None] + B.DENSE_STEPS[None, :]]
            worst = float(np.max(np.abs(redo - p_ar[::REPRO_EVERY])))
            print(f"AR reproduction on {len(sub):,} test anchors: worst |diff| {worst:.2e} m")
            assert worst < REPRO_TOL, 'regenerated AR forecasts do not reproduce the saved test predictions'
            gap_rows.append({'gauge': gauge, 'check': 'AR test reproduction worst abs diff (m)', 'value': worst})
        else:
            p_ar = ar.forecast(s, 'residual', anchors) + U
            np.savez_compressed(OUT / f'{gauge}_ar_residual_val_predictions.npz', preds=p_ar.astype(np.float32),
                                anchors=times, lead_names=np.array(LEAD_NAMES))

        months = times.astype('datetime64[M]')
        res = {}
        for model, p in [('Dense stage 2', p_dense), ('AR residual', p_ar), ('UTide', U)]:
            e = p - Y
            b, sd, r, n = decompose(e, mask)
            slow = slow_component(e, mask, months)
            res[model] = {'bias': b, 'std': sd, 'rmse': r, 'slow': slow}
            for j, h in enumerate(LEAD_NAMES):
                lead_rows.append({'gauge': gauge, 'split': split, 'model': model, 'horizon': h,
                                  'lead_h': LEAD_H[j], 'bias_m': b[j], 'std_m': sd[j], 'rmse_m': r[j],
                                  'monthly_offset_mse_m2': slow[j], 'n': int(n[j])})
        fig_data[gauge][split] = res

        d, a = res['Dense stage 2'], res['AR residual']
        gap_mse = d['rmse'][I72] ** 2 - a['rmse'][I72] ** 2
        row = {'gauge': gauge, 'split': split,
               'rmse_dense_72h': d['rmse'][I72], 'rmse_ar_72h': a['rmse'][I72],
               'bias_dense_72h': d['bias'][I72], 'bias_ar_72h': a['bias'][I72],
               'std_dense_72h': d['std'][I72], 'std_ar_72h': a['std'][I72],
               'gap_rmse_m': d['rmse'][I72] - a['rmse'][I72], 'gap_mse_m2': gap_mse,
               'share_gap_global_bias': (d['bias'][I72] ** 2 - a['bias'][I72] ** 2) / gap_mse,
               'share_gap_monthly_offset': (d['slow'][I72] - a['slow'][I72]) / gap_mse,
               'rmse_dense_72h_minus_monthly_offset': np.sqrt(d['rmse'][I72] ** 2 - d['slow'][I72]),
               'rmse_ar_72h_minus_monthly_offset': np.sqrt(a['rmse'][I72] ** 2 - a['slow'][I72])}
        # the same, averaged over leads 24h..71h (dense stage 2 has an edge spike at its last lead, 72h)
        mid = (LEAD_H >= 24) & (LEAD_H <= 71)
        g_mid = np.mean(d['rmse'][mid] ** 2) - np.mean(a['rmse'][mid] ** 2)
        row.update({'rmse_dense_24_71h': np.sqrt(np.mean(d['rmse'][mid] ** 2)),
                    'rmse_ar_24_71h': np.sqrt(np.mean(a['rmse'][mid] ** 2)),
                    'gap_mse_24_71h_m2': g_mid,
                    'share_gap_24_71h_global_bias': (np.mean(d['bias'][mid] ** 2) - np.mean(a['bias'][mid] ** 2)) / g_mid,
                    'share_gap_24_71h_monthly_offset': (np.mean(d['slow'][mid]) - np.mean(a['slow'][mid])) / g_mid,
                    'dense_rmse_70h': d['rmse'][LEAD_NAMES.index('70h')], 'dense_rmse_71h': d['rmse'][LEAD_NAMES.index('71h')]})
        gap_rows.append({'gauge': gauge, 'check': f'{split} 72h decomposition', **row})
        print(f"[{split}] 72h RMSE dense {row['rmse_dense_72h']:.4f} vs AR {row['rmse_ar_72h']:.4f}; "
              f"bias {row['bias_dense_72h']:+.4f} vs {row['bias_ar_72h']:+.4f}; "
              f"gap share: global bias {row['share_gap_global_bias']:.1%}, "
              f"monthly offset {row['share_gap_monthly_offset']:.1%}")

# %% Tables -------------------------------------------------------------------
pd.DataFrame(utide_year_rows).to_csv(OUT / 'A_utide_bias_by_split_and_year.csv', index=False)
pd.DataFrame(monthly_rows).to_csv(OUT / 'A_utide_monthly_mean_residual.csv', index=False)
pd.DataFrame(lead_rows).to_csv(OUT / 'A_bias_std_rmse_by_lead.csv', index=False)
gap_df = pd.DataFrame(gap_rows)
gap_df.to_csv(OUT / 'A_72h_gap_decomposition.csv', index=False)
print(f"\nsaved tables to {OUT}")

# %% Figure A1: annual mean UTide residual, whole record ----------------------
uy = pd.DataFrame(utide_year_rows)
uy = uy[uy['period'].apply(lambda k: not isinstance(k, str))]
mm = pd.DataFrame(monthly_rows)
fig, axes = plt.subplots(len(GAUGES), 1, figsize=(6.3, 5.4), sharex=True)
for ax, gauge in zip(axes, GAUGES):
    for sp, (lo, hi) in {'train': ('2004-01-01', C.TRAIN_END), 'val': (C.TRAIN_END, C.VAL_END),
                         'test': (C.VAL_END, '2025-01-01')}.items():
        ax.axvspan(pd.Timestamp(lo), pd.Timestamp(hi), color=SPLIT_SHADE[sp], lw=0, zorder=0)
        ax.text(pd.Timestamp(lo) + (pd.Timestamp(hi) - pd.Timestamp(lo)) / 2, 1.0, sp,
                transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=8, color=ANNOTATION_INK)
    m = mm[mm.gauge == gauge]
    ax.plot(m['month'], m['mean_residual_m'], color='#9a9993', lw=0.7, label='monthly mean', zorder=2)
    g = uy[uy.gauge == gauge]
    ax.plot([pd.Timestamp(f'{int(y)}-07-01') for y in g['period']], g['mean_residual_m'], color=INK, lw=2,
            marker='o', ms=4, label='annual mean', zorder=3)
    ax.axhline(0, color=ANNOTATION_INK, lw=0.6, zorder=1)
    ax.set_ylabel('Observed - UTide (m)')
    ax.set_title(C.gauge_info(gauge)['label'], fontsize=10, loc='left')
axes[0].legend(loc='upper left', frameon=False, fontsize=8, ncol=2)
save_fig(fig, 'A1_utide_residual_annual_mean', width_in=6.3, height_in=5.4)
plt.close(fig)

# %% Figure A2/A3: RMSE, |bias| and std against lead, dense vs AR -------------
for gauge in GAUGES:
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.6), sharex=True, sharey='row',
                             gridspec_kw={'height_ratios': [1.6, 1]})
    for col, split in enumerate(('val', 'test')):
        ax, axd = axes[0, col], axes[1, col]
        res = fig_data[gauge][split]
        for model in ('Dense stage 2', 'AR residual'):
            r = res[model]
            ax.plot(LEAD_H, r['rmse'], color=COL[model], lw=2, ls='-')
            ax.plot(LEAD_H, r['std'], color=COL[model], lw=1.4, ls=(0, (4, 2)))
            ax.plot(LEAD_H, np.abs(r['bias']), color=COL[model], lw=1.4, ls=(0, (1, 1.5)))
        d, a = res['Dense stage 2'], res['AR residual']
        axd.axhline(0, color=ANNOTATION_INK, lw=0.6)
        axd.plot(LEAD_H, d['rmse'] - a['rmse'], color=INK, lw=2, ls='-')
        axd.plot(LEAD_H, d['std'] - a['std'], color=INK, lw=1.4, ls=(0, (4, 2)))
        axd.plot(LEAD_H, np.abs(d['bias']) - np.abs(a['bias']), color=INK, lw=1.4, ls=(0, (1, 1.5)))
        axd.set_xlabel('Lead time (h)')
        axd.set_xticks([0, 12, 24, 36, 48, 60, 72])
        ax.set_title({'val': 'Validation 2018-2020', 'test': 'Test 2021-2024'}[split], fontsize=10)
        ax.set_ylim(bottom=0)
    axes[0, 0].set_ylabel('Metres')
    axes[1, 0].set_ylabel('Dense minus AR (m)')
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COL['Dense stage 2'], lw=2, label='Dense stage 2'),
               Line2D([], [], color=COL['AR residual'], lw=2, label='AR on UTide residual'),
               Line2D([], [], color='0.3', lw=2, ls='-', label='RMSE'),
               Line2D([], [], color='0.3', lw=1.4, ls=(0, (4, 2)), label='error std'),
               Line2D([], [], color='0.3', lw=1.4, ls=(0, (1, 1.5)), label='|bias|')]
    fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.005, f"{C.gauge_info(gauge)['label']}. RMSE$^2$ = bias$^2$ + std$^2$.",
             fontsize=7, color=ANNOTATION_INK)
    save_fig(fig, f'A2_bias_std_rmse_by_lead_{gauge}', width_in=6.3, height_in=5.8, rect=(0, 0.03, 1, 0.94))
    plt.close(fig)

print("done")
