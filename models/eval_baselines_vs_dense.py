"""Re-evaluate the classical baselines against a chosen dense variant on TEST (no re-search, no refit).

The baselines' test forecasts are deterministic functions of the frozen best_configs.json, and
eval_baselines.py saved them (baseline_test_predictions_<name>.npz). They are loaded here, after a
spot check that the AR residual baseline still reproduces them, so the comparison takes minutes
instead of the ~2.5h a full refit of every method needs. Everything is compared against the
variant's own test predictions (dense_test_predictions_stage2.npz, written by train_dense.py
--score-test), which must reproduce that variant's dense_metrics.csv.

Adds two JOINT tests to the per-lead paired block bootstrap: the bootstrap CI of the mean RMSE
difference (best baseline per track minus dense) over all 73 leads, and over leads 24h..72h.

    TIDAL_GAUGE=southend TIDAL_DENSE_VARIANT=dense_v2_skip python models/eval_baselines_vs_dense.py

Outputs: outputs/[<gauge>/]baselines_vs_<variant>/   (never the original baselines folder)
"""
# %% Bootstrap
import pathlib
import sys

_start = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / '_')).resolve()
REPO_ROOT = next(p for p in _start.parents if (p / 'utils' / 'paths.py').exists())
sys.path.insert(0, str(REPO_ROOT))

import json

import numpy as np
import pandas as pd

from utils import paths
from utils.models import baselines as B
from utils.models import config as C
from utils.models import training

# %% ------------------------------------------------------------------ CONFIG
GAUGE = C.from_env('TIDAL_GAUGE', 'southend')
VARIANT = C.dense_variant()
SMOKE = C.from_env('TIDAL_SMOKE', '0').strip().lower() not in ('', '0', 'false')
suffix = '_smoke' if SMOKE and VARIANT != C.DENSE_LEGACY_VARIANT else ''

LEAD_NAMES = list(C.DENSE_HORIZONS)
LEAD_H = B.DENSE_STEPS / 6
SHARED_LEADS = ['10min', '1h', '6h', '24h', '48h', '72h']
LEAD_IDX = [LEAD_NAMES.index(h) for h in SHARED_LEADS]
BOOT_REPS = 50 if SMOKE else C.BASELINE_BOOTSTRAP_REPS
BLOCK_DAYS = C.BASELINE_BOOTSTRAP_BLOCK_DAYS
EXTREME_QUANTILE = 0.99
HISTORY_CHECK_DAYS = 30
PARITY_TOLERANCE = 5e-4
REPRO_EVERY = 500
JOINT_SETS = {'all 73 leads': np.ones(len(LEAD_NAMES), bool), '24h to 72h': LEAD_H >= 24}

C.configure_threads()
DENSE_DIR = pathlib.Path(str(paths.dense_output_dir(GAUGE, VARIANT)) + suffix)
BASE_DIR = paths.baseline_output_dir(GAUGE)
OUT = paths.ensure_dir(pathlib.Path(str(BASE_DIR) + f'_vs_{VARIANT}{suffix}'))
BEST = json.loads(paths.baseline_config_path(GAUGE).read_text(encoding='utf-8'))
LABEL = f'Dense stage 2 ({VARIANT})'
print(f"{C.gauge_info(GAUGE)['label']}: baselines vs {VARIANT}{suffix} -> {OUT}")

series = B.load_series(GAUGE, verbose=False)
anchors = B.split_anchors(series, 'test')
times = series.time[anchors].astype('datetime64[ns]')
Y, FLAG = B.targets(series.level, series.flagged, anchors)
UTIDE = series.utide[anchors[:, None] + B.DENSE_STEPS[None, :]]

dense = np.load(DENSE_DIR / 'dense_test_predictions_stage2.npz')
assert np.array_equal(dense['anchors'].astype('datetime64[ns]'), times), 'test anchors differ from the dense set'
NN = dense['preds'].astype(np.float64)

metrics = []


def add(label, family, pred):
    rmse, mae = training.masked_rmse_mae(pred, Y, FLAG)
    for j, h in enumerate(LEAD_NAMES):
        metrics.append({'model': label, 'family': family, 'horizon': h, 'lead_h': LEAD_H[j],
                        'rmse_m': float(rmse[j]), 'mae_m': float(mae[j])})
    return rmse


published = pd.read_csv(DENSE_DIR / 'dense_metrics.csv')
published = published[published.model == 'Dense stage 2'].set_index('horizon').loc[LEAD_NAMES, 'rmse_m'].to_numpy()
worst = float(np.max(np.abs(add(LABEL, 'neural', NN) - published)))
print(f"{VARIANT} re-scored: worst difference from its dense_metrics.csv {worst:.2e} m")
assert worst < PARITY_TOLERANCE, 'scoring differs from the variant dense_metrics.csv'
add('UTide standalone', 'reference', UTIDE)
for track in C.BASELINE_TRACKS:
    add(f'Persistence ({track})', 'reference', B.persistence(series, track, anchors) + (UTIDE if track == 'residual' else 0))
    add(f'Seasonal naive ({track})', 'reference', B.seasonal_naive(series, track, anchors) + (UTIDE if track == 'residual' else 0))

preds = {}
for name, entry in BEST.items():
    z = np.load(BASE_DIR / f'baseline_test_predictions_{name}.npz')
    assert np.array_equal(z['anchors'].astype('datetime64[ns]'), times), f'{name}: anchors differ'
    preds[name] = z['preds'].astype(np.float64)
    add(name, entry['method'], preds[name])

# spot check: the saved AR residual forecasts are still what the frozen config produces
ranges = json.loads((BASE_DIR / 'identification' / 'identified_ranges.json').read_text(encoding='utf-8'))
ar = B.build_model('ar', BEST['ar_residual']['config'], series, 'residual', ranges['residual']['dhr_frequencies'],
                   B.ModelCache())
sub = np.arange(0, len(anchors), REPRO_EVERY)
redo = ar.forecast(series, 'residual', anchors[sub]) + UTIDE[sub]
repro = float(np.max(np.abs(redo - preds['ar_residual'][sub])))
print(f"AR residual reproduction on {len(sub)} anchors: worst |diff| {repro:.2e} m")
assert repro < 1e-4, 'saved baseline predictions no longer reproduce'
pd.DataFrame(metrics).to_csv(OUT / 'baseline_metrics.csv', index=False)

# %% Paired block bootstrap per lead, plus the joint tests ----------------------
winners = {t: min((k for k, v in BEST.items() if v['track'] == t and not v['parity']),
                  key=lambda k: BEST[k]['val_mean_lead_mse']) for t in C.BASELINE_TRACKS}
day = times.astype('datetime64[D]')
unique_days, day_index = np.unique(day, return_inverse=True)
n_days = len(unique_days)
rng = np.random.default_rng(C.SEED)
block_starts = rng.integers(0, n_days, size=(BOOT_REPS, int(np.ceil(n_days / BLOCK_DAYS))))
sample = ((block_starts[:, :, None] + np.arange(BLOCK_DAYS)[None, None, :]) % n_days).reshape(BOOT_REPS, -1)[:, :n_days]


def daily_sums(sq):
    out = np.zeros((n_days, sq.shape[1]))
    count = np.zeros((n_days, sq.shape[1]))
    for j in range(sq.shape[1]):
        ok = ~FLAG[:, j]
        out[:, j] = np.bincount(day_index[ok], sq[ok, j], n_days)
        count[:, j] = np.bincount(day_index[ok], minlength=n_days)
    return out, count


nn_sum, count = daily_sums((NN - Y) ** 2)
boot_rows, joint_rows = [], []
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
        boot_rows.append({'track': track, 'baseline': name, 'horizon': h, 'lead_h': LEAD_H[j],
                          'rmse_diff_m': observed[j], 'ci_low': low[j], 'ci_high': high[j],
                          'significant': bool(low[j] > 0 or high[j] < 0)})
    for set_name, sel in JOINT_SETS.items():
        obs = float(observed[sel].mean())
        d = draws[:, sel].mean(1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        joint_rows.append({'track': track, 'baseline': name, 'comparator': VARIANT, 'leads': set_name,
                           'n_leads': int(sel.sum()), 'mean_rmse_diff_m': obs, 'ci_low': float(lo),
                           'ci_high': float(hi), 'significant': bool(lo > 0 or hi < 0),
                           'favours': 'dense' if obs > 0 else 'baseline', 'boot_reps': BOOT_REPS})
pd.DataFrame(boot_rows).to_csv(OUT / 'baseline_bootstrap.csv', index=False)
joint = pd.DataFrame(joint_rows)
joint.to_csv(OUT / 'baseline_joint_tests.csv', index=False)
print('\njoint tests (mean over leads of RMSE(baseline) - RMSE(dense); > 0 favours dense):')
print(joint.round(4).to_string())

# %% Stability, extremes and history gaps ----------------------------------------
everything = {LABEL: NN, 'UTide standalone': UTIDE, **preds}
years = times.astype('datetime64[Y]').astype(int) + 1970
val_anchors = B.split_anchors(series, 'val')
threshold = float(np.nanquantile(np.abs(series.level[val_anchors] - series.utide[val_anchors]), EXTREME_QUANTILE))
residual_target = np.abs(Y - UTIDE)
missing = np.cumsum(np.isnan(series.level))
window = HISTORY_CHECK_DAYS * C.STEPS_PER_DAY
incomplete = (missing[anchors] - missing[anchors - window]) > 0
by_year, extremes, history = [], [], []
for label, p in everything.items():
    for year in np.unique(years):
        keep = years == year
        rmse, _ = training.masked_rmse_mae(p[keep][:, LEAD_IDX], Y[keep][:, LEAD_IDX], FLAG[keep][:, LEAD_IDX])
        by_year += [{'model': label, 'year': int(year), 'horizon': h, 'rmse_m': float(r)} for h, r in zip(SHARED_LEADS, rmse)]
    mask = FLAG | (residual_target < threshold)
    rmse, _ = training.masked_rmse_mae(p[:, LEAD_IDX], Y[:, LEAD_IDX], mask[:, LEAD_IDX])
    extremes += [{'model': label, 'horizon': h, 'rmse_m': float(r), 'threshold_m': threshold} for h, r in zip(SHARED_LEADS, rmse)]
    for subset, keep in [('incomplete', incomplete), ('complete', ~incomplete)]:
        if keep.any():
            rmse, _ = training.masked_rmse_mae(p[keep][:, LEAD_IDX], Y[keep][:, LEAD_IDX], FLAG[keep][:, LEAD_IDX])
            history += [{'model': label, 'subset': subset, 'share': float(keep.mean()), 'horizon': h,
                         'rmse_m': float(r)} for h, r in zip(SHARED_LEADS, rmse)]
pd.DataFrame(by_year).to_csv(OUT / 'baseline_by_year.csv', index=False)
pd.DataFrame(extremes).to_csv(OUT / 'baseline_extremes.csv', index=False)
pd.DataFrame(history).to_csv(OUT / 'baseline_incomplete_history.csv', index=False)
(OUT / 'winners.json').write_text(json.dumps({'baseline_winners': winners, 'comparator': VARIANT,
                                              'ar_reproduction_worst_m': repro, 'parity_worst_m': worst},
                                             indent=2), encoding='utf-8')
print(f"\nAll outputs under: {OUT}")
