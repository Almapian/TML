"""Final test evaluation of the classical baselines, from the frozen best_configs.json.

Before any baseline is scored, the saved dense-model forecasts and UTide are re-scored here and
must reproduce dense_metrics.csv, which shows the scoring is the one the neural models had.

    python models/eval_baselines.py        # after models/search_baselines.py

Outputs (outputs[/<gauge>]/baselines/):
    baseline_metrics.csv          RMSE/MAE at the 73 lead times: baselines, references, UTide, dense
    baseline_vs_nn_matched.csv    the same on the anchors shared with dense stage 3 tide-only,
                                  plus the sparse models' own tables at their shared leads
    baseline_bootstrap.csv        moving-block bootstrap of RMSE(best baseline) - RMSE(dense stage 2)
    baseline_by_year.csv          RMSE per test year at the shared leads
    baseline_extremes.csv         RMSE on the largest 1% of residuals (threshold from validation)
    baseline_incomplete_history.csv  RMSE on anchors whose 30-day history has gaps, and the rest
    baseline_residual_diagnostics.csv  one-step error autocorrelation and Ljung-Box, on validation
    baseline_72h_example.csv      the Feb 2022 forecast, for plot_baselines.py
    baseline_test_predictions_<name>.npz
"""
# %% Bootstrap
import pathlib
import sys

_start = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / '_')).resolve()
REPO_ROOT = next(p for p in _start.parents if (p / 'utils' / 'paths.py').exists())
sys.path.insert(0, str(REPO_ROOT))

# %% Imports
import json

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from utils import paths
from utils.models import baselines as B
from utils.models import config as C
from utils.models import training

# %% ------------------------------------------------------------------ CONFIG
GAUGE = C.from_env('TIDAL_GAUGE', 'southend')    # run_gauge.py sets it
SMOKE = C.from_env('TIDAL_SMOKE', '0').strip().lower() not in ('', '0', 'false')

LEAD_NAMES = list(C.DENSE_HORIZONS)
SHARED_LEADS = ['10min', '1h', '6h', '24h', '48h', '72h']
LEAD_IDX = [LEAD_NAMES.index(h) for h in SHARED_LEADS]
EXAMPLE_ISSUE = np.datetime64('2022-02-17T12:00')    # as train_dense.py
BOOT_REPS = 50 if SMOKE else C.BASELINE_BOOTSTRAP_REPS
BLOCK_DAYS = C.BASELINE_BOOTSTRAP_BLOCK_DAYS
HISTORY_CHECK_DAYS = 30
EXTREME_QUANTILE = 0.99
PARITY_TOLERANCE = 5e-4     # metres; dense_metrics.csv came from float32 predictions

suffix = '_smoke' if SMOKE else ''
OUT = paths.ensure_dir(str(paths.baseline_output_dir(GAUGE)) + suffix)
DENSE_DIR = paths.dense_output_dir(GAUGE)
CONFIG_PATH = OUT / 'search' / 'best_configs.json' if SMOKE else paths.baseline_config_path(GAUGE)
BEST = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
RANGES = json.loads((OUT / 'identification' / 'identified_ranges.json').read_text(encoding='utf-8'))

series = B.load_series(GAUGE)
anchors = B.split_anchors(series, 'test')
times = series.time[anchors]
Y, FLAG = B.targets(series.level, series.flagged, anchors)
UTIDE = series.utide[anchors[:, None] + B.DENSE_STEPS[None, :]]
print(f"{C.gauge_info(GAUGE)['label']}: {len(anchors):,} test anchors")

# %% The scoring must match the neural models' ---------------------------------
dense2 = np.load(DENSE_DIR / 'dense_test_predictions_stage2.npz')
dense3 = np.load(DENSE_DIR / 'dense_test_predictions_stage3_tideonly.npz')
assert np.array_equal(dense2['anchors'].astype('datetime64[ns]'), times.astype('datetime64[ns]')), \
    'test anchors differ from the dense stage 2 set'
assert list(dense2['lead_names']) == LEAD_NAMES

metrics = []


def add(label, family, pred, Y_=Y, FLAG_=FLAG):
    rmse, mae = training.masked_rmse_mae(pred, Y_, FLAG_)
    for j, h in enumerate(LEAD_NAMES):
        metrics.append({'model': label, 'family': family, 'horizon': h, 'lead_h': B.DENSE_STEPS[j] / 6,
                        'rmse_m': float(rmse[j]), 'mae_m': float(mae[j])})
    return rmse


published = pd.read_csv(DENSE_DIR / 'dense_metrics.csv')
published = published[published.model == 'Dense stage 2'].set_index('horizon').loc[LEAD_NAMES, 'rmse_m'].to_numpy()
rescored = add('Dense stage 2', 'neural', dense2['preds'].astype(np.float64))
worst = float(np.max(np.abs(rescored - published)))
print(f"dense stage 2 re-scored: worst difference from dense_metrics.csv {worst:.2e} m")
assert worst < PARITY_TOLERANCE, 'scoring differs from dense_metrics.csv'
published_u = pd.read_csv(DENSE_DIR / 'dense_metrics.csv')
published_u = published_u[published_u.model == 'UTide standalone'].set_index('horizon').loc[LEAD_NAMES, 'rmse_m'].to_numpy()
rescored_u = add('UTide standalone', 'reference', UTIDE)
print(f"UTide re-scored: worst difference {np.max(np.abs(rescored_u - published_u)):.2e} m")
assert np.max(np.abs(rescored_u - published_u)) < PARITY_TOLERANCE, 'UTide differs from the neural pipeline'

# %% Forecast ------------------------------------------------------------------
preds = {}
for track in C.BASELINE_TRACKS:
    add(f'Persistence ({track})', 'reference',
        B.persistence(series, track, anchors) + (UTIDE if track == 'residual' else 0))
    add(f'Seasonal naive ({track})', 'reference',
        B.seasonal_naive(series, track, anchors) + (UTIDE if track == 'residual' else 0))

cache = B.ModelCache()
models = {}
for name, entry in BEST.items():
    track, method = entry['track'], entry['method']
    model = B.build_model(method, entry['config'], series, track, RANGES[track]['dhr_frequencies'], cache)
    pred = model.forecast(series, track, anchors)
    pred = pred + (UTIDE if track == 'residual' else 0)
    preds[name] = pred
    models[name] = model
    rmse = add(name, method, pred)
    np.savez_compressed(OUT / f'baseline_test_predictions_{name}.npz', preds=pred.astype(np.float32),
                        anchors=times, lead_names=np.array(LEAD_NAMES))
    print(f"{name}: RMSE at {SHARED_LEADS}: {np.round(rmse[LEAD_IDX], 3)}", flush=True)

metric_df = pd.DataFrame(metrics)
metric_df.to_csv(OUT / 'baseline_metrics.csv', index=False)

# %% Matched against dense stage 3 and the sparse tables ------------------------
common, i_b, i_3 = np.intersect1d(times.astype('datetime64[ns]'), dense3['anchors'].astype('datetime64[ns]'),
                                  return_indices=True)
print(f"matched anchors with dense stage 3: {len(common):,} of {len(times):,}")
matched = []
rows = [('Dense stage 2', dense2['preds'][i_b]), ('Dense stage 3 (tide-only)', dense3['preds'][i_3]),
        ('UTide standalone', UTIDE[i_b])] + [(k, v[i_b]) for k, v in preds.items()]
for label, p in rows:
    rmse, mae = training.masked_rmse_mae(p[:, LEAD_IDX], Y[i_b][:, LEAD_IDX], FLAG[i_b][:, LEAD_IDX])
    for j, h in enumerate(SHARED_LEADS):
        matched.append({'model': label, 'horizon': h, 'rmse_m': float(rmse[j]), 'mae_m': float(mae[j]),
                        'scored_on': 'matched anchors', 'n_anchors': len(common)})
for stage, table, names in [(2, 'stage2_metrics.csv', ['Stage-1 baseline (flat MLP)', 'Stage-1 baseline (flat LSTM)',
                                                       'Stage-2 (attn decoder)']),
                            (3, 'stage3_metrics.csv', ['Stage-3 (meteo fusion)', 'Stage-3 (tide-only, no meteo)'])]:
    source = pd.read_csv(paths.stage_output_dir(stage, GAUGE) / table)
    source = source[source.model.isin(names) & source.horizon.isin(SHARED_LEADS)]
    for row in source.itertuples():
        matched.append({'model': row.model, 'horizon': row.horizon, 'rmse_m': row.rmse_m, 'mae_m': row.mae_m,
                        'scored_on': 'own test windows', 'n_anchors': np.nan})
pd.DataFrame(matched).to_csv(OUT / 'baseline_vs_nn_matched.csv', index=False)

# %% Paired bootstrap against dense stage 2 --------------------------------------
# Winners are chosen on validation score alone, one per track.
winners = {t: min((k for k, v in BEST.items() if v['track'] == t and not v['parity']),
                  key=lambda k: BEST[k]['val_mean_lead_mse']) for t in C.BASELINE_TRACKS}
day = times.astype('datetime64[D]')
unique_days, day_index = np.unique(day, return_inverse=True)
rng = np.random.default_rng(C.SEED)
n_days = len(unique_days)
block_starts = rng.integers(0, n_days, size=(BOOT_REPS, int(np.ceil(n_days / BLOCK_DAYS))))
sample = ((block_starts[:, :, None] + np.arange(BLOCK_DAYS)[None, None, :]) % n_days).reshape(BOOT_REPS, -1)[:, :n_days]

err2 = {'nn': (dense2['preds'].astype(np.float64) - Y) ** 2}
boot_rows = []


def daily_sums(sq):
    out = np.zeros((n_days, sq.shape[1]))
    count = np.zeros((n_days, sq.shape[1]))
    for j in range(sq.shape[1]):
        ok = ~FLAG[:, j]
        out[:, j] = np.bincount(day_index[ok], sq[ok, j], n_days)
        count[:, j] = np.bincount(day_index[ok], minlength=n_days)
    return out, count


nn_sum, count = daily_sums(err2['nn'])
for track, name in winners.items():
    base_sum, _ = daily_sums((preds[name] - Y) ** 2)
    observed = np.sqrt(base_sum.sum(0) / count.sum(0)) - np.sqrt(nn_sum.sum(0) / count.sum(0))
    draws = np.zeros((BOOT_REPS, base_sum.shape[1]))
    for rep in range(BOOT_REPS):
        take = sample[rep]
        n = count[take].sum(0)
        draws[rep] = np.sqrt(base_sum[take].sum(0) / n) - np.sqrt(nn_sum[take].sum(0) / n)
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    for j, h in enumerate(LEAD_NAMES):
        boot_rows.append({'track': track, 'baseline': name, 'horizon': h, 'lead_h': B.DENSE_STEPS[j] / 6,
                          'rmse_diff_m': observed[j], 'ci_low': low[j], 'ci_high': high[j],
                          'significant': bool(low[j] > 0 or high[j] < 0)})
pd.DataFrame(boot_rows).to_csv(OUT / 'baseline_bootstrap.csv', index=False)

# %% Stability, extremes and history gaps ----------------------------------------
everything = {'Dense stage 2': dense2['preds'].astype(np.float64), 'UTide standalone': UTIDE, **preds}
years = times.astype('datetime64[Y]').astype(int) + 1970
by_year, extremes, history = [], [], []

val_anchors = B.split_anchors(series, 'val')
threshold = float(np.nanquantile(np.abs(series.level[val_anchors] - series.utide[val_anchors]), EXTREME_QUANTILE))
residual_target = np.abs(Y - UTIDE)

missing = np.cumsum(np.isnan(series.level))
window = HISTORY_CHECK_DAYS * C.STEPS_PER_DAY
incomplete = (missing[anchors] - missing[anchors - window]) > 0

for label, p in everything.items():
    for year in np.unique(years):
        keep = years == year
        rmse, _ = training.masked_rmse_mae(p[keep][:, LEAD_IDX], Y[keep][:, LEAD_IDX], FLAG[keep][:, LEAD_IDX])
        by_year += [{'model': label, 'year': int(year), 'horizon': h, 'rmse_m': float(r)}
                    for h, r in zip(SHARED_LEADS, rmse)]
    mask = FLAG | (residual_target < threshold)
    rmse, _ = training.masked_rmse_mae(p[:, LEAD_IDX], Y[:, LEAD_IDX], mask[:, LEAD_IDX])
    extremes += [{'model': label, 'horizon': h, 'rmse_m': float(r), 'threshold_m': threshold}
                 for h, r in zip(SHARED_LEADS, rmse)]
    for subset, keep in [('incomplete', incomplete), ('complete', ~incomplete)]:
        if keep.any():
            rmse, _ = training.masked_rmse_mae(p[keep][:, LEAD_IDX], Y[keep][:, LEAD_IDX], FLAG[keep][:, LEAD_IDX])
            history += [{'model': label, 'subset': subset, 'share': float(keep.mean()), 'horizon': h,
                         'rmse_m': float(r)} for h, r in zip(SHARED_LEADS, rmse)]
pd.DataFrame(by_year).to_csv(OUT / 'baseline_by_year.csv', index=False)
pd.DataFrame(extremes).to_csv(OUT / 'baseline_extremes.csv', index=False)
pd.DataFrame(history).to_csv(OUT / 'baseline_incomplete_history.csv', index=False)

# %% Diagnostics on validation one-step errors -----------------------------------
series_val = B.load_series(GAUGE, upto=C.VAL_END, verbose=False)
diag_anchors = B.split_anchors(series_val, 'val')
diag = []
for name, entry in BEST.items():
    track = entry['track']
    model = models[name]
    one = model.forecast(series_val, track, diag_anchors)[:, 0]
    err = one - series_val.track(track)[diag_anchors + 1]
    err = err[~np.isnan(err)]
    lb = acorr_ljungbox(err[-100_000:], lags=[6, 36, 144])
    row = {'model': name, 'error_std': float(err.std())}
    for lag, stat, p in zip([6, 36, 144], lb.lb_stat, lb.lb_pvalue):
        row[f'lb_stat_{lag}'], row[f'lb_p_{lag}'] = float(stat), float(p)
    for lag in (1, 6, 36, 144):
        row[f'acf_{lag}'] = float(np.corrcoef(err[:-lag], err[lag:])[0, 1])
    diag.append(row)
pd.DataFrame(diag).to_csv(OUT / 'baseline_residual_diagnostics.csv', index=False)

# %% One 72h forecast, for plot_baselines.py --------------------------------------
issue_i = int(np.argmin(np.abs(times.astype('datetime64[m]') - EXAMPLE_ISSUE)))
issue = times[issue_i]
rows = []
tuned = [(k, v) for k, v in preds.items() if not BEST[k]['parity']]
for label, p in [('Dense stage 2', dense2['preds'])] + tuned:
    for j, h in enumerate(LEAD_NAMES):
        rows.append({'model': label, 'issue_time': issue, 'horizon': h, 'lead_h': B.DENSE_STEPS[j] / 6,
                     'valid_time': issue + np.timedelta64(int(B.DENSE_STEPS[j]) * 10, 'm'),
                     'forecast_m': float(p[issue_i, j])})
pd.DataFrame(rows).to_csv(OUT / 'baseline_72h_example.csv', index=False)
span = (series.time >= issue - np.timedelta64(24, 'h')) & (series.time <= issue + np.timedelta64(73, 'h'))
pd.DataFrame({'DateTime': series.time[span], 'Observed_ODN': series.level[span],
              'utide_recon': series.utide[span]}).to_csv(OUT / 'baseline_72h_example_observed.csv', index=False)

(OUT / 'winners.json').write_text(json.dumps(winners, indent=2), encoding='utf-8')
print(f"\nvalidation winners: {winners}\nAll baseline outputs are under: {OUT}")
