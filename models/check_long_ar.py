"""Independent checks of the long AR building blocks against statsmodels and brute force.

    python models/check_long_ar.py

1. Yule-Walker (fit_ar, estimator='yw', no jitter) on a gap-free stretch of the Southend train
   residual against statsmodels' yule_walker(method='adjusted'), which also divides lag k by n - k.
2. The gap-aware autocovariance (gappy_acov) against a pairwise brute-force sum on a gappy stretch.
3. The multi-step weight table (pi_to_weights) against an explicit recursive rollout.
Each prints the largest absolute difference; all should be at rounding level (< 1e-9).
"""
import pathlib
import sys

_start = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / '_')).resolve()
REPO_ROOT = next(p for p in _start.parents if (p / 'utils' / 'paths.py').exists())
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from statsmodels.regression.linear_model import yule_walker

from utils.models import baselines as B

s = B.load_series('southend', verbose=False)
y = s.track('residual')[:s.n_train]

starts, stops = B.segments(y, min_len=20000)
i = int(np.argmax(stops - starts))
z = y[starts[i]:stops[i]][:50000]
ours = B.fit_ar(z, 200, 'yw', jitter=0.0)
ref, _ = yule_walker(z, order=200, method='adjusted', demean=True)
print(f'1. Yule-Walker AR(200), {len(z):,} gap-free values: max |difference| = {np.abs(ours - ref).max():.1e}')

w = y[:20000]
d = w - np.nanmean(w)
brute = [np.nansum(d[k:] * d[:len(d) - k]) / np.sum(~np.isnan(d[k:]) & ~np.isnan(d[:len(d) - k])) for k in range(31)]
print(f'2. gappy_acov, {np.isnan(w).sum()} missing values, lags 0 to 30: '
      f'max |difference| = {np.abs(B.gappy_acov(w, 30) - np.array(brute)).max():.1e}')

rng = np.random.default_rng(0)
pi, hist = rng.normal(0, 0.05, 40), rng.normal(0, 1, 40)     # hist: oldest to newest
steps = np.arange(1, 25)
x, rollout = list(hist), []
for _ in steps:
    x.append(sum(pi[k] * x[-1 - k] for k in range(len(pi))))
    rollout.append(x[-1])
direct = B.pi_to_weights(pi, steps) @ hist[::-1]             # lag 1 = newest value
print(f'3. pi_to_weights against a recursive rollout, 24 steps: max |difference| = '
      f'{np.abs(direct - np.array(rollout)).max():.1e}')
