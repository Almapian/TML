"""Post-processing of dense forecasts: hourly -> 10-min trajectory, and high/low-water errors.

No training. For each anchor t the dense model predicts the level at 10 min and 1h..72h. Its
correction to the harmonic prediction, c(k) = forecast(t+k) - UTide(t+k), is interpolated onto
every 10-min lead 0..72h (433 leads; the knot at lead 0 is the observed residual at the anchor,
obs(t) - UTide(t), which is known at issue time), and UTide at 10-min resolution is added back.
Linear and cubic (scipy CubicSpline, not-a-knot) interpolation are both scored.

High water (HW): peaks of the observed 10-min series, at least HW_MIN_SEP apart and with enough
prominence, whose +-MATCH_WINDOW neighbourhood is gap- and flag-free. Each observed HW inside a
forecast's 0..72h span (with the whole matching window inside it) is matched to the highest value
of the forecast trajectory within +-MATCH_WINDOW: height error = predicted max - observed HW,
timing error = time of predicted max - time of observed HW. Low water (LW) the same with minima,
scored only where LWs are not truncated (LW_SCORED). At Richmond they are: LWs pile up against a
hard lower edge near -1.0 m ODN (almost none below -1.06 m; summer medians sit on it), and the
level stays within 5 cm of LW for a median 60 min against 30 min at HW, consistent with a control
(the half-tide barrier, or the bed/weir at the tidal limit) holding low water up. The statistics
are written to <split>_extremes_check.json and the histogram to <split>_low_water_histogram.png.

Compared: harmonic prediction (UTide) alone; the dense model at its 73 native leads only (no
interpolation); the dense model interpolated linearly and cubically; and the AR residual
baseline forecast directly at every 10-min lead (its frozen config, refit on train exactly as
eval_baselines does), when --with-ar.

    python models/postprocess_hw.py --gauge southend --variant dense --split val [--with-ar]

Outputs: outputs/[<gauge>/]<variant>/postprocess/  (CSV tables, one LW histogram PNG)
"""
# %% Bootstrap
import pathlib
import sys

_start = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / '_')).resolve()
REPO_ROOT = next(p for p in _start.parents if (p / 'utils' / 'paths.py').exists())
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use('Agg')

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

from utils import paths
from utils.models import baselines as B
from utils.models import config as C

# %% ------------------------------------------------------------------ CONFIG
FULL_STEPS = np.arange(0, 433)                 # every 10-min lead 0..72h
KNOTS = np.concatenate([[0], B.DENSE_STEPS])   # lead 0 (observed residual) + the 73 native leads
HW_MIN_SEP = 48                                # 8h in 10-min steps: one HW per semidiurnal cycle
HW_PROMINENCE = 0.5                            # m; spring and neap tides both exceed this at both gauges
MATCH_WINDOW = 18                              # +-3h in 10-min steps
HW_ANCHOR_STRIDE = 6                           # HW/LW scoring on hourly anchors (all HWs still used)
LW_SCORED = {'southend': True, 'richmond': False}   # see the module docstring
CHUNK = 20000
REPORT_LEADS = ['10min', '30min', '1h', '6h', '24h', '48h', '72h']
LEAD_BINS = [(0, 24), (24, 48), (48, 72)]


def interpolate(corr_knots, kind):
    """(N, 74) corrections at KNOTS -> (N, 433) at every 10-min lead."""
    if kind == 'linear':
        out = np.empty((len(corr_knots), len(FULL_STEPS)))
        for i in range(len(corr_knots)):
            out[i] = np.interp(FULL_STEPS, KNOTS, corr_knots[i])
        return out
    return CubicSpline(KNOTS, corr_knots, axis=1)(FULL_STEPS)


def trajectories(s, anchors, preds, kind):
    """(N, 433) forecast trajectories in metres. kind: 'linear', 'cubic', 'native' (NaN off the
    native leads), or 'utide'."""
    u_full = s.utide[anchors[:, None] + FULL_STEPS[None, :]]
    if kind == 'utide':
        return u_full
    if kind == 'native':
        out = np.full(u_full.shape, np.nan)
        out[:, 0] = s.level[anchors]
        out[:, B.DENSE_STEPS] = preds
        return out
    c = np.empty((len(anchors), len(KNOTS)))
    c[:, 0] = s.level[anchors] - s.utide[anchors]
    c[:, 1:] = preds - s.utide[anchors[:, None] + B.DENSE_STEPS[None, :]]
    return u_full + interpolate(c, kind)


_AR_MODELS = {}


def ar_full_trajectories(gauge, s, anchors):
    """The AR residual baseline (frozen best config, refit on train as eval_baselines does)
    forecast at every 10-min lead 1..432, plus UTide; lead 0 is the observation."""
    if gauge not in _AR_MODELS:
        best = json.loads(paths.baseline_config_path(gauge).read_text(encoding='utf-8'))['ar_residual']['config']
        y_train = s.track('residual')[:s.n_train]
        phi = B.fit_ar(y_train, best['order'], best['estimator'], best.get('jitter', 1e-5))
        _AR_MODELS[gauge] = B.linear_from_pi(phi, y_train, steps=FULL_STEPS[1:])
    model = _AR_MODELS[gauge]
    out = np.empty((len(anchors), len(FULL_STEPS)))
    out[:, 0] = s.level[anchors]
    out[:, 1:] = model.forecast(s, 'residual', anchors, steps=FULL_STEPS[1:]) \
        + s.utide[anchors[:, None] + FULL_STEPS[None, 1:]]
    return out


def score_full(s, anchors, preds, kinds, extra=None):
    """Per-lead RMSE/MAE/bias at all 433 leads, chunked so memory stays small."""
    acc = {k: {'sse': np.zeros(433), 'sae': np.zeros(433), 'se': np.zeros(433), 'n': np.zeros(433)}
           for k in list(kinds) + (list(extra) if extra else [])}
    for lo in range(0, len(anchors), CHUNK):
        a = anchors[lo:lo + CHUNK]
        Y, F = B.targets(s.level, s.flagged, a, steps=FULL_STEPS)
        ok = ~F & ~np.isnan(Y)
        trajs = {k: trajectories(s, a, preds[lo:lo + CHUNK], k) for k in kinds}
        if extra:
            for k, fn in extra.items():
                trajs[k] = fn(a)
        for k, tr in trajs.items():
            e = np.where(ok & ~np.isnan(tr), tr - Y, 0.0)
            m = ok & ~np.isnan(tr)
            acc[k]['sse'] += (e ** 2).sum(0); acc[k]['sae'] += np.abs(e).sum(0)
            acc[k]['se'] += e.sum(0); acc[k]['n'] += m.sum(0)
    rows = []
    for k, v in acc.items():
        n = np.maximum(v['n'], 1)
        for j, step in enumerate(FULL_STEPS):
            if v['n'][j] == 0:
                continue
            rows.append({'model': k, 'lead_min': int(step * 10), 'lead_h': step / 6,
                         'rmse_m': float(np.sqrt(v['sse'][j] / n[j])), 'mae_m': float(v['sae'][j] / n[j]),
                         'bias_m': float(v['se'][j] / n[j]), 'n': int(v['n'][j])})
    return pd.DataFrame(rows)


def extremes(s, kind='high'):
    """Grid indices of clean observed high (or low) waters."""
    lvl = np.where(s.flagged, np.nan, s.level)
    sig = lvl if kind == 'high' else -lvl
    filled = np.where(np.isnan(sig), -1e9, sig)
    peaks, _ = find_peaks(filled, distance=HW_MIN_SEP, prominence=HW_PROMINENCE)
    bad = np.isnan(lvl)
    bad_near = np.convolve(bad.astype(int), np.ones(2 * MATCH_WINDOW + 1, int), mode='same') > 0
    return peaks[~bad_near[peaks]]


def plateau_minutes(s, events):
    """Median minutes the level stays within 5 cm of each extreme (within +-6h)."""
    lvl = np.where(s.flagged, np.nan, s.level)
    d = [np.sum(np.abs(lvl[p - 36:p + 37] - lvl[p]) <= 0.05) * 10 for p in events if 36 <= p < len(lvl) - 37]
    return float(np.median(d))


def lw_floor_check(s, lows, highs):
    """Statistics describing whether low waters are truncated (reported, not a decision rule)."""
    h = s.level[lows]
    p2 = np.percentile(h, 2)
    return {'n_low_waters': int(len(h)), 'lw_min_m': float(h.min()), 'lw_p2_m': float(p2),
            'lw_p10_m': float(np.percentile(h, 10)), 'lw_p50_m': float(np.median(h)),
            'share_within_5cm_of_p2': float(np.mean(h <= p2 + 0.05)),
            'lw_plateau_median_min': plateau_minutes(s, lows), 'hw_plateau_median_min': plateau_minutes(s, highs)}, h


def match_events(s, anchors, traj_fns, events, kind):
    """Height and timing errors of every (anchor, observed extreme) pair inside the span."""
    rows = []
    a_sorted = anchors  # ascending
    for name, fn in traj_fns.items():
        herr, terr, leadh, edge = [], [], [], []
        for lo in range(0, len(anchors), CHUNK):
            a = anchors[lo:lo + CHUNK]
            tr = fn(a, lo)
            if kind == 'low':
                tr = -tr
            # events whose matching window fits inside this chunk's spans
            lo_p = a[0] + MATCH_WINDOW
            hi_p = a[-1] + 432 - MATCH_WINDOW
            for p in events[(events >= lo_p) & (events <= hi_p)]:
                rel = p - a
                k = np.flatnonzero((rel >= MATCH_WINDOW) & (rel <= 432 - MATCH_WINDOW))
                if len(k) == 0:
                    continue
                idx = rel[k][:, None] + np.arange(-MATCH_WINDOW, MATCH_WINDOW + 1)[None, :]
                win = tr[k[:, None], idx]
                if np.isnan(win).all():
                    continue
                j = np.nanargmax(win, axis=1)
                pred = win[np.arange(len(k)), j]
                obs = s.level[p] if kind == 'high' else -s.level[p]
                sign = 1 if kind == 'high' else -1
                herr.append(sign * (pred - obs)); terr.append((j - MATCH_WINDOW) * 10.0)
                leadh.append(rel[k] / 6); edge.append((j == 0) | (j == 2 * MATCH_WINDOW))
        if not herr:
            continue
        herr, terr, leadh, edge = map(np.concatenate, (herr, terr, leadh, edge))
        for lo_h, hi_h in [(0, 72)] + LEAD_BINS:
            m = (leadh >= lo_h) & (leadh <= hi_h)
            rows.append({'model': name, 'event': f'{kind} water', 'lead_bin_h': f'{lo_h}-{hi_h}',
                         'n_matches': int(m.sum()), 'height_bias_m': float(herr[m].mean()),
                         'height_rmse_m': float(np.sqrt(np.mean(herr[m] ** 2))),
                         'timing_mae_min': float(np.mean(np.abs(terr[m]))),
                         'timing_bias_min': float(np.mean(terr[m])),
                         'share_at_window_edge': float(edge[m].mean())})
    return rows


def run(gauge, variant, split, with_ar=False, max_anchors=None, smoke_suffix=''):
    t0 = time.time()
    out_dir = paths.ensure_dir(pathlib.Path(str(paths.dense_output_dir(gauge, variant)) + smoke_suffix)
                               / ('postprocess_subset' if max_anchors else 'postprocess'))
    pred_file = pathlib.Path(str(paths.dense_output_dir(gauge, variant)) + smoke_suffix) / \
        ('dense_val_predictions_stage2.npz' if split == 'val' else 'dense_test_predictions_stage2.npz')
    z = np.load(pred_file)
    s = B.load_series(gauge, verbose=False)
    times = z['anchors'].astype('datetime64[ns]')
    anchors = np.searchsorted(s.time.astype('datetime64[ns]'), times)
    assert np.array_equal(s.time[anchors].astype('datetime64[ns]'), times), 'anchors not on the grid'
    assert list(z['lead_names']) == list(C.DENSE_HORIZONS)
    preds = z['preds'].astype(np.float64)
    if max_anchors:
        keep = np.linspace(0, len(anchors) - 1, max_anchors).astype(int)
        anchors, preds = anchors[keep], preds[keep]
    print(f"{gauge} {variant}{smoke_suffix} {split}: {len(anchors):,} anchors")

    # interpolation is exact at the native leads
    tr = trajectories(s, anchors[:200], preds[:200], 'linear')
    assert np.allclose(tr[:, B.DENSE_STEPS], preds[:200], atol=1e-9)
    assert np.allclose(trajectories(s, anchors[:200], preds[:200], 'cubic')[:, B.DENSE_STEPS], preds[:200], atol=1e-9)

    # ---- RMSE at all 433 leads
    kinds = ['utide', 'linear', 'cubic', 'native']
    full = score_full(s, anchors, preds, kinds)
    full['model'] = full['model'].map({'utide': 'UTide alone', 'linear': 'dense, linear interp',
                                       'cubic': 'dense, cubic interp', 'native': 'dense, native leads only'})
    full.to_csv(out_dir / f'{split}_rmse_433_leads.csv', index=False)

    # ---- high and low waters
    hw_anchors = np.arange(0, len(anchors), HW_ANCHOR_STRIDE)
    a_hw, p_hw = anchors[hw_anchors], preds[hw_anchors]
    fns = {'UTide alone': lambda a, lo: trajectories(s, a, None, 'utide'),
           'dense, native leads only': lambda a, lo: trajectories(s, a, p_hw[lo:lo + CHUNK], 'native'),
           'dense, linear interp': lambda a, lo: trajectories(s, a, p_hw[lo:lo + CHUNK], 'linear'),
           'dense, cubic interp': lambda a, lo: trajectories(s, a, p_hw[lo:lo + CHUNK], 'cubic')}
    if with_ar:
        ar_cache = {}

        def ar_fn(a, lo):
            if lo not in ar_cache:
                ar_cache.clear(); ar_cache[lo] = ar_full_trajectories(gauge, s, a)
            return ar_cache[lo]
        fns['AR residual baseline (10-min)'] = ar_fn
    highs = extremes(s, 'high')
    lows = extremes(s, 'low')
    span = (s.time >= s.time[anchors[0]]) & (s.time <= s.time[anchors[-1]] + np.timedelta64(72, 'h'))
    n_days = span.sum() / C.STEPS_PER_DAY
    check, lw_heights = lw_floor_check(s, lows[span[lows]], highs[span[highs]])
    check.update({'gauge': gauge, 'split': split,
                  'high_waters_per_day': float(span[highs].sum() / n_days),
                  'low_waters_per_day': float(span[lows].sum() / n_days)})
    rows = match_events(s, a_hw, fns, highs, 'high')
    lw_scored = LW_SCORED[gauge]
    check['lw_scored'] = lw_scored
    if lw_scored:
        rows += match_events(s, a_hw, fns, lows, 'low')
    pd.DataFrame(rows).to_csv(out_dir / f'{split}_high_low_water.csv', index=False)
    (out_dir / f'{split}_extremes_check.json').write_text(json.dumps(check, indent=2), encoding='utf-8')

    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    ax.hist(lw_heights, bins=120, color='#52514e')
    ax.set_xlabel('Observed low-water height (m ODN)'); ax.set_ylabel('Count')
    ax.set_title(f"{C.gauge_info(gauge)['label']}, low waters ({split} span)", fontsize=10, loc='left')
    fig.tight_layout(); fig.savefig(out_dir / f'{split}_low_water_histogram.png', dpi=200); plt.close(fig)

    summary = full[full.lead_min.isin([10, 30, 60, 360, 1440, 2880, 4320])].pivot_table(
        index='model', columns='lead_min', values='rmse_m')
    print(summary.round(4).to_string())
    print(pd.DataFrame(rows).round(4).to_string())
    print(json.dumps(check, indent=1))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")
    return out_dir


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gauge', required=True, choices=list(C.GAUGES))
    ap.add_argument('--variant', default='dense')
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--with-ar', action='store_true')
    ap.add_argument('--max-anchors', type=int, default=None)
    ap.add_argument('--smoke-suffix', default='')
    a = ap.parse_args()
    C.configure_threads()
    run(a.gauge, C.dense_variant(a.variant), a.split, a.with_ar, a.max_anchors, a.smoke_suffix)
