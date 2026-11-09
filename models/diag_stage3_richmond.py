"""Diagnostic B (Nov 2026): why sparse stage 3 (meteo fusion) is so poor at short leads at Richmond.

No training. VALIDATION period only. Loads the existing Richmond stage 3 checkpoints
(meteo fusion and tide-only) and:
  1. scores them on validation, unablated and with each input branch/channel replaced by its
     training-set mean (in the scaled units the model sees);
  2. audits the inputs: scalers (recomputed vs stored in the checkpoint), ERA5 point, time
     alignment (empirical DST test against UTide), daily river mapping, missingness per split,
     covariate shift train -> validation, and hyperparameters;
  3. draws a two-week validation sanity plot.

    python models/diag_stage3_richmond.py

Outputs: outputs/diagnostics_2026-11/B_*  and  reports/2026-11-nov-fixes/figures/B*.png
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
import torch

from utils import paths
from utils.models import baselines as B
from utils.models import config as C
from utils.models import datasets, training, windowing
from utils.models.architectures import TidalSeq2SeqMulti, TidalSeq2SeqTideOnly
from utils.plot_config import ANNOTATION_INK, save_fig, set_report_dir

# %% ------------------------------------------------------------------ CONFIG
GAUGE = 'richmond'
REPORT_LEADS = ['10min', '1h', '6h', '24h', '72h']
HORIZONS = C.STAGE3_HORIZONS
NAMES = list(HORIZONS)
LEAD_IDX = [NAMES.index(h) for h in REPORT_LEADS]
METEO_CH = ['pressure', 'wind_u10', 'wind_v10']
RIVER_CH = ['rainfall', 'discharge', 'rainfall_available', 'discharge_available']
SANITY_DAYS = 14
DST_LAGS = np.arange(-12, 13)          # 10-min steps, i.e. -2h .. +2h

OUT = paths.ensure_dir(paths.OUTPUTS_DIR / 'diagnostics_2026-11')
FIG_DIR = paths.ensure_dir(REPO_ROOT / 'reports' / '2026-11-nov-fixes' / 'figures')
set_report_dir(str(FIG_DIR))
INK = '#0b0b0b'
COL = {'multi': '#1baf7a', 'tideonly': '#2a78d6', 'obs': INK, 'cov': '#52514e'}

DEVICE = C.describe_device()
C.set_seed()
audit = {}

# %% Data: exactly the training pipeline's preparation --------------------------
# the checkpoints were trained with the original calendar-day river mapping, so score them with it
data = datasets.prepare_stage3(DEVICE, list(HORIZONS.values()), C.DECODER_SEQ_STEPS, C.TARGET_SLICE_IN_DECODER,
                               gauge=GAUGE, river_lag_hours=C.LEGACY_RIVER_LAG_HOURS)
X, Xm, Xr, Yd, Y, F, T, A = windowing.build_windows_stage3(
    data.val_df, C.LOOKBACK, list(HORIZONS.values()), data.grids, data.utide_full, data.utide_start,
    C.DECODER_SEQ_STEPS)
assert len(X) == len(data.val_tensors['X']), 'validation windows differ from the training pipeline'
sc = data.scalers
val = data.val_tensors              # scaled, as during training
train = data.train_tensors
print(f"validation windows: {len(X):,}")

# %% Checkpoints ----------------------------------------------------------------
ckdir = paths.checkpoint_dir(GAUGE)
ck_multi = torch.load(ckdir / 'stage3_tidal_seq2seq_multi.pt', map_location='cpu', weights_only=False)
ck_tide = torch.load(ckdir / 'stage3_tidal_seq2seq_tideonly.pt', map_location='cpu', weights_only=False)
multi = TidalSeq2SeqMulti(C.DECODER_SEQ_LEN, C.TARGET_SLICE_IN_DECODER)
multi.load_state_dict(ck_multi['model_state_dict']); multi.eval()
tide = TidalSeq2SeqTideOnly(C.DECODER_SEQ_LEN, C.TARGET_SLICE_IN_DECODER)
tide.load_state_dict(ck_tide['model_state_dict']); tide.eval()

# scaler audit: stored at training time vs recomputed now from Richmond's train split
scaler_rows = []
for key, stored in ck_multi['scalers'].items():
    now = sc[key]
    scaler_rows.append({'covariate': key, 'stored_mean': stored['mean'], 'stored_std': stored['std'],
                        'recomputed_mean': now['mean'], 'recomputed_std': now['std'],
                        'southend_mean': None, 'southend_std': None})
sth = torch.load(paths.checkpoint_dir('southend') / 'stage3_tidal_seq2seq_multi.pt', map_location='cpu',
                 weights_only=False)['scalers']
for r in scaler_rows:
    r['southend_mean'], r['southend_std'] = sth[r['covariate']]['mean'], sth[r['covariate']]['std']
pd.DataFrame(scaler_rows).to_csv(OUT / 'B_scalers.csv', index=False)
audit['hyperparameters_multi'] = {k: v for k, v in ck_multi['config'].items() if k not in ('horizons', 'decoder_seq_steps')}
audit['hyperparameters_tideonly'] = {k: v for k, v in ck_tide['config'].items() if k not in ('horizons', 'decoder_seq_steps')}


# %% 1. Ablation at inference -------------------------------------------------
@torch.no_grad()
def predict(model_kind, tensors):
    preds = []
    for s in range(0, len(tensors['X']), 4096):
        b = {k: v[s:s + 4096] for k, v in tensors.items()}
        if model_kind == 'multi':
            p = multi(b['X'], b['Xm'], b['Xr'], b['Ydec'])[0]
        else:
            p = tide(b['X'], b['Ydec'])[0]
        preds.append(p)
    return torch.cat(preds).numpy() * sc['tide_utide']['std'] + sc['tide_utide']['mean']


def score(p):
    rmse, _ = training.masked_rmse_mae(p, Y, F)
    return rmse


def ablate(key, channels=None):
    """Copy of the validation tensors with tensor `key` (optionally only `channels` of its
    last axis) set to its training-set mean, in scaled units."""
    t = {k: v for k, v in val.items() if k in ('X', 'Xm', 'Xr', 'Ydec')}
    x = t[key].clone()
    if channels is None:
        x[:] = float(train[key].mean())
    else:
        for c in channels:
            x[..., c] = float(train[key][..., c].mean())
    t[key] = x
    return t


base_inputs = {k: v for k, v in val.items() if k in ('X', 'Xm', 'Xr', 'Ydec')}
p_multi = predict('multi', base_inputs)
p_tide = predict('tideonly', base_inputs)
r_multi, r_tide = score(p_multi), score(p_tide)

# consistency: scaled masked MSE equals the best validation loss in the training history
hist = json.loads((paths.stage_output_dir(3, GAUGE) / 'stage3_history.json').read_text())
for label, p, hk in [('multi', p_multi, 'Stage-3 (meteo fusion)'), ('tideonly', p_tide, 'Stage-3 (tide-only, no meteo)')]:
    e = ((p - Y) / sc['tide_utide']['std'])[~F]
    audit[f'val_mse_check_{label}'] = {'scored': float(np.mean(e ** 2)), 'history_best': float(min(hist[hk]['val_loss'])),
                                       'best_epoch': int(np.argmin(hist[hk]['val_loss']) + 1),
                                       'epochs_run': len(hist[hk]['val_loss'])}
print(json.dumps({k: v for k, v in audit.items() if k.startswith('val_mse')}, indent=1))

ablations = [('none (unablated)', None, None),
             ('tide_history', 'X', None), ('utide_decoder', 'Ydec', None),
             ('pressure', 'Xm', [0]), ('wind_u10', 'Xm', [1]), ('wind_v10', 'Xm', [2]),
             ('all meteo (pressure, u10, v10)', 'Xm', [0, 1, 2]),
             ('rainfall', 'Xr', [0]), ('discharge', 'Xr', [1]),
             ('river_availability_flags', 'Xr', [2, 3]),
             ('all river (rain, discharge, flags)', 'Xr', [0, 1, 2, 3])]
rows = []
for name, key, ch in ablations:
    p = p_multi if key is None else predict('multi', ablate(key, ch))
    r = score(p)
    rows.append({'model': 'stage 3 meteo fusion', 'branch_removed': name,
                 **{h: r[i] for h, i in zip(REPORT_LEADS, LEAD_IDX)}})
# same model, river inputs re-gathered without the future leak (water day must have ended)
import dataclasses
grids33 = dataclasses.replace(data.grids, river_lag_hours=C.RIVER_AVAILABILITY_LAG_HOURS)
_, _, Xr33, _, Y33, _, _, A33 = windowing.build_windows_stage3(
    data.val_df, C.LOOKBACK, list(HORIZONS.values()), grids33, data.utide_full, data.utide_start, C.DECODER_SEQ_STEPS)
assert np.array_equal(A33, A), 'leak-free river windows changed the validation anchors'
xr33 = Xr33.copy()
xr33[..., 0] = (xr33[..., 0] - sc['rainfall']['mean']) / sc['rainfall']['std']
xr33[..., 1] = (xr33[..., 1] - sc['discharge']['mean']) / sc['discharge']['std']
t33 = dict(base_inputs, Xr=datasets.to_tensor(xr33, 'cpu'))
r = score(predict('multi', t33))
rows.append({'model': 'stage 3 meteo fusion', 'branch_removed': 'none, river leak removed (33h lag)',
             **{h: r[i] for h, i in zip(REPORT_LEADS, LEAD_IDX)}})
for name, key in [('none (unablated)', None), ('tide_history', 'X'), ('utide_decoder', 'Ydec')]:
    p = p_tide if key is None else predict('tideonly', ablate(key))
    r = score(p)
    rows.append({'model': 'stage 3 tide-only', 'branch_removed': name,
                 **{h: r[i] for h, i in zip(REPORT_LEADS, LEAD_IDX)}})
abl = pd.DataFrame(rows)
abl.to_csv(OUT / 'B_ablation_val_rmse.csv', index=False)
print(abl.round(4).to_string())
pd.DataFrame({'horizon': NAMES, 'multi_val_rmse': r_multi, 'tideonly_val_rmse': r_tide}).to_csv(
    OUT / 'B_val_rmse_all_horizons.csv', index=False)

# %% 2. Input audit ---------------------------------------------------------------
g = data.grids
hourly_t = g.hourly_start + np.arange(len(g.pressure_hpa)) * np.timedelta64(1, 'h')
daily_t = g.daily_start + np.arange(len(g.rain)) * np.timedelta64(1, 'D')


def split_of(t):
    return np.where(t < np.datetime64(C.TRAIN_END), 'train', np.where(t < np.datetime64(C.VAL_END), 'val', 'test'))


miss = []
gauge_df = data.df
gsplit = split_of(gauge_df['DateTime'].to_numpy())
full_grid = pd.date_range(gauge_df['DateTime'].min(), gauge_df['DateTime'].max(), freq='10min')
for sp in ('train', 'val', 'test'):
    rows_sp = gauge_df[gsplit == sp]
    grid_sp = full_grid[split_of(full_grid.values) == sp]
    miss.append({'covariate': 'gauge level (rows absent from 10-min grid)', 'split': sp,
                 'fraction': 1 - len(rows_sp) / len(grid_sp)})
    miss.append({'covariate': 'gauge level (flagged: chatter/stuck/imputed)', 'split': sp,
                 'fraction': float(rows_sp['is_flagged'].mean())})
    in_gauge_span = (daily_t >= full_grid[0].to_datetime64()) & (daily_t <= full_grid[-1].to_datetime64())
    d = (split_of(daily_t) == sp) & in_gauge_span
    miss.append({'covariate': 'rainfall (climatology-filled days)', 'split': sp, 'fraction': float(1 - g.rain_avail[d].mean())})
    miss.append({'covariate': 'discharge (climatology-filled days)', 'split': sp, 'fraction': float(1 - g.discharge_avail[d].mean())})
    h = split_of(hourly_t) == sp
    for nm, arr in [('pressure', g.pressure_hpa), ('u10', g.u10), ('v10', g.v10)]:
        miss.append({'covariate': f'ERA5 {nm} (NaN hours)', 'split': sp, 'fraction': float(np.isnan(arr[h]).mean())})
pd.DataFrame(miss).to_csv(OUT / 'B_missingness.csv', index=False)

# covariate shift: validation inputs in the scaled units the model sees (train = mean 0, std 1)
shift = []
for key, chans in [('Xm', METEO_CH), ('Xr', RIVER_CH)]:
    for c, nm in enumerate(chans):
        tr, va = train[key][..., c], val[key][..., c]
        shift.append({'channel': nm, 'train_mean': float(tr.mean()), 'train_std': float(tr.std()),
                      'val_mean': float(va.mean()), 'val_std': float(va.std()),
                      'val_p99': float(torch.quantile(va.flatten()[::97], 0.99)),
                      'train_p99': float(torch.quantile(tr.flatten()[::97], 0.99))})
pd.DataFrame(shift).to_csv(OUT / 'B_covariate_shift_scaled.csv', index=False)

# empirical DST test: lag of observed level vs UTide, by month of year, both gauges
dst_rows = []
for gg in ('richmond', 'southend'):
    s = B.load_series(gg, verbose=False)
    lvl = np.where(s.flagged, np.nan, s.level)
    months = pd.DatetimeIndex(s.time).month.to_numpy()
    for m in range(1, 13):
        sel = np.flatnonzero(months == m)
        sel = sel[(sel > 20) & (sel < len(s.time) - 20)]
        best, best_rmse = None, np.inf
        for L in DST_LAGS:
            # observed at t compared with UTide at t+L: L>0 means the observed clock runs behind UTide
            e = lvl[sel] - s.utide[sel + L]
            r = np.sqrt(np.nanmean(e ** 2))
            if r < best_rmse:
                best, best_rmse = int(L), r
        dst_rows.append({'gauge': gg, 'month': m, 'best_lag_min': best * 10, 'rmse_at_best_m': best_rmse})
dst = pd.DataFrame(dst_rows)
dst.to_csv(OUT / 'B_timezone_lag_by_month.csv', index=False)
print(dst.pivot(index='month', columns='gauge', values='best_lag_min').T.to_string())

# ERA5 point: only 'Southend Pier' exists in the consolidated CSVs
audit['era5_columns'] = list(pd.read_csv(paths.METEO_DIR / 'pressure_all_gauges.csv', nrows=1).columns)
audit['era5_columns_wind'] = list(pd.read_csv(paths.METEO_DIR / 'wind_all_gauges.csv', nrows=1).columns)
(OUT / 'B_audit.json').write_text(json.dumps(audit, indent=2, default=str), encoding='utf-8')

# %% 3. Sanity plot: two validation weeks around the highest Kingston discharge ----------
val_days = (daily_t >= np.datetime64(C.TRAIN_END)) & (daily_t < np.datetime64(C.VAL_END))
peak_day = daily_t[val_days][np.argmax(g.discharge[val_days])]
t0 = peak_day - np.timedelta64(SANITY_DAYS // 2, 'D')
t1 = t0 + np.timedelta64(SANITY_DAYS, 'D')
tgt = T[:, 0]                                       # target time of the 10-min lead
w = (tgt >= t0) & (tgt < t1)
obs = data.val_df[(data.val_df.DateTime >= pd.Timestamp(t0)) & (data.val_df.DateTime < pd.Timestamp(t1))]
hw = (hourly_t >= t0) & (hourly_t < t1)
dw = (daily_t >= t0 - np.timedelta64(1, 'D')) & (daily_t < t1)

fig, axes = plt.subplots(6, 1, figsize=(6.3, 8.6), sharex=True,
                         gridspec_kw={'height_ratios': [2.4, 1, 1, 1, 1, 1]})
ax = axes[0]
ax.plot(obs.DateTime, obs.Observed_ODN, color=COL['obs'], lw=1.0, label='Observed')
ax.plot(tgt[w], p_tide[w, 0], color=COL['tideonly'], lw=1.0, ls=(0, (4, 2)), label='Tide-only, 10 min ahead')
ax.plot(tgt[w], p_multi[w, 0], color=COL['multi'], lw=1.2, label='Meteo fusion, 10 min ahead')
ax.set_ylabel('Level (m ODN)')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7)
for ax, (lab, x, y, unit) in zip(axes[1:], [
        ('Pressure', hourly_t[hw], g.pressure_hpa[hw], 'hPa'), ('u10', hourly_t[hw], g.u10[hw], 'm/s'),
        ('v10', hourly_t[hw], g.v10[hw], 'm/s'), ('Rainfall', daily_t[dw], g.rain[dw], 'mm/day'),
        ('Discharge', daily_t[dw], g.discharge[dw], 'm$^3$/s')]):
    if lab in ('Rainfall', 'Discharge'):
        ax.step(x, y, where='post', color=COL['cov'], lw=1.2)
    else:
        ax.plot(x, y, color=COL['cov'], lw=1.0)
    ax.set_ylabel(f'{lab}\n({unit})', fontsize=8)
axes[-1].tick_params(axis='x', labelsize=7.5)
fig.text(0.01, 0.003, f"Richmond Lock, validation, {pd.Timestamp(t0):%d %b} to {pd.Timestamp(t1):%d %b %Y}. "
         f"ERA5 at the Southend Pier grid point; NRFA daily values drawn from the start of their day.",
         fontsize=6.5, color=ANNOTATION_INK)
save_fig(fig, 'B1_richmond_stage3_sanity_two_weeks', width_in=6.3, height_in=8.6)
plt.close(fig)

# per-lead RMSE of the two models on validation
fig, ax = plt.subplots(figsize=(6.3, 3.2))
lead_h = np.array([HORIZONS[h] / 6 for h in NAMES])
ax.plot(lead_h, r_multi, color=COL['multi'], lw=2, marker='o', ms=4, label='Meteo fusion (epoch-1 weights)')
ax.plot(lead_h, r_tide, color=COL['tideonly'], lw=2, marker='s', ms=4, ls=(0, (4, 2)), label='Tide-only')
ax.set_xscale('log')
ax.set_xticks([1 / 6, 1, 6, 24, 72, 168, 720], ['10min', '1h', '6h', '24h', '72h', '7d', '30d'])
ax.set_ylabel('Validation RMSE (m)')
ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=8)
save_fig(fig, 'B2_richmond_stage3_val_rmse_by_lead', width_in=6.3, height_in=3.2)
plt.close(fig)
print(f"sanity window: {pd.Timestamp(t0)} to {pd.Timestamp(t1)} (peak discharge {pd.Timestamp(peak_day):%Y-%m-%d})")
print("done")
