"""Validation RMSE of every dense candidate against the AR residual baseline, on one scorer. No training.

Answers "how much of the Southend gap to AR does each variant close?" on VALIDATION, so no test
numbers are needed for development decisions. The AR residual baseline uses its frozen config,
refit on train only (as eval_baselines does), forecast at the 73 dense leads from the same
validation anchors as the dense models. All models are scored with the same mask and targets.

    python models/diag_val_vs_ar.py
Outputs: outputs/diagnostics_2026-11/val_vs_ar_<gauge>.csv
"""
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

VARIANTS = ['dense', 'dense_v2_skip', 'dense_v2_skip_wloss', 'dense_v2_skip_scaled', 'dense_v2_pad',
            'dense_v2_skip_longmem']
CHUNK = 20000


def main():
    out_dir = paths.ensure_dir(paths.OUTPUTS_DIR / 'diagnostics_2026-11')
    for gauge in C.GAUGES:
        s = B.load_series(gauge, verbose=False)
        preds, times = {}, None
        for v in VARIANTS:
            f = paths.dense_output_dir(gauge, v) / 'dense_val_predictions_stage2.npz'
            if not f.exists():
                continue
            z = np.load(f)
            t = z['anchors'].astype('datetime64[ns]')
            if times is None:
                times = t
            assert np.array_equal(times, t), f'{gauge} {v}: anchors differ'
            preds[v] = z['preds'].astype(np.float64)
        anchors = np.searchsorted(s.time.astype('datetime64[ns]'), times)
        assert np.array_equal(s.time[anchors].astype('datetime64[ns]'), times)
        best = json.loads(paths.baseline_config_path(gauge).read_text(encoding='utf-8'))['ar_residual']['config']
        y_train = s.track('residual')[:s.n_train]
        phi = B.fit_ar(y_train, best['order'], best['estimator'], best.get('jitter', 1e-5))
        ar = B.linear_from_pi(phi, y_train, steps=B.DENSE_STEPS)
        names = list(preds) + ['ar_residual', 'utide']
        sse = {k: np.zeros(len(B.DENSE_STEPS)) for k in names}
        n = np.zeros(len(B.DENSE_STEPS))
        for lo in range(0, len(anchors), CHUNK):
            a = anchors[lo:lo + CHUNK]
            Y, F = B.targets(s.level, s.flagged, a, steps=B.DENSE_STEPS)
            u = s.utide[a[:, None] + B.DENSE_STEPS[None, :]]
            P = {k: p[lo:lo + CHUNK] for k, p in preds.items()}
            P['ar_residual'] = ar.forecast(s, 'residual', a, steps=B.DENSE_STEPS) + u
            P['utide'] = u
            ok = ~F & ~np.isnan(Y)
            for k in P:
                assert not np.isnan(P[k][ok]).any(), k
            for k, p in P.items():
                sse[k] += (np.where(ok, p - Y, 0.0) ** 2).sum(0)
            n += ok.sum(0)
        rmse = pd.DataFrame({k: np.sqrt(v / n) for k, v in sse.items()}, index=list(C.DENSE_HORIZONS))
        rmse.index.name = 'horizon'
        rmse.to_csv(out_dir / f'val_vs_ar_{gauge}.csv')
        summ = pd.DataFrame({'mean_73': rmse.mean(), 'mean_24_72h': rmse.iloc[24:].mean(),
                             'gap_to_ar_73_mm': (rmse.mean() - rmse['ar_residual'].mean()) * 1000,
                             'gap_to_ar_24_72h_mm': (rmse.iloc[24:].mean() - rmse['ar_residual'].iloc[24:].mean()) * 1000})
        print(f'\n== {gauge} (validation, {len(anchors):,} anchors)')
        print(rmse.loc[['10min', '1h', '6h', '24h', '48h', '72h']].round(4).to_string())
        print(summ.round(4).to_string())


if __name__ == '__main__':
    main()
