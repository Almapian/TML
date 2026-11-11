"""Apply the pre-registered (amended, 2026-11-10) selection rule to the dense candidates. Validation only.

Score per candidate and gauge: for each of the 73 dense leads h,
    RMSE_h(candidate) / min(RMSE_h(residual persistence), RMSE_h(UTide alone)),
averaged over the 73 leads. Lowest wins. RMSEs come from each candidate's dense_val_metrics.csv
('Dense stage 2' rows); the reference rows come from the v2 files, which share the validation
anchors (asserted). Candidates whose metrics file is missing (did not finish) are listed as such.

    python models/select_dense_winner.py --gauge southend [--suffix _smoke] [--out DIR]
"""
import pathlib
import sys

_start = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / '_')).resolve()
REPO_ROOT = next(p for p in _start.parents if (p / 'utils' / 'paths.py').exists())
sys.path.insert(0, str(REPO_ROOT))

import argparse
import json

import numpy as np
import pandas as pd

from utils import paths
from utils.models import config as C

CANDIDATES = ['dense', 'dense_v2_skip', 'dense_v2_skip_wloss', 'dense_v2_skip_scaled', 'dense_v2_pad']
LEADS = list(C.DENSE_HORIZONS)


def metrics_file(gauge, variant, suffix):
    # the legacy model is never retrained, so its real file is used even in a dry run
    d = paths.dense_output_dir(gauge, variant)
    return (pathlib.Path(str(d) + ('' if variant == C.DENSE_LEGACY_VARIANT else suffix))) / 'dense_val_metrics.csv'


def select(gauge, suffix='', exclude_legacy=False):
    rmse, refs, missing = {}, None, []
    for v in CANDIDATES:
        if exclude_legacy and v == C.DENSE_LEGACY_VARIANT:
            continue
        f = metrics_file(gauge, v, suffix)
        if not f.exists():
            missing.append(v)
            continue
        m = pd.read_csv(f)
        rmse[v] = m[m.model == 'Dense stage 2'].set_index('horizon').loc[LEADS, 'rmse_m'].to_numpy()
        if v != C.DENSE_LEGACY_VARIANT:
            r = m[m.model.isin(['Residual persistence', 'UTide standalone'])].pivot_table(
                index='horizon', columns='model', values='rmse_m').loc[LEADS]
            if refs is None:
                refs = r
            else:
                assert np.allclose(refs.to_numpy(), r.to_numpy(), atol=1e-9), f'{v}: reference rows differ'
    if refs is None:
        raise SystemExit(f'{gauge}: no v2 candidate finished, so there are no reference rows')
    denom = np.minimum(refs['Residual persistence'].to_numpy(), refs['UTide standalone'].to_numpy())
    scores = {v: float(np.mean(r / denom)) for v, r in rmse.items()}
    winner = min(scores, key=scores.get)
    return {'gauge': gauge, 'rule': 'mean over 73 leads of val RMSE / min(residual persistence, UTide) RMSE; lowest wins',
            'scores': scores, 'winner': winner, 'missing': missing,
            'per_lead_ratio': {v: (r / denom).tolist() for v, r in rmse.items()}, 'leads': LEADS}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gauge', required=True, choices=list(C.GAUGES))
    ap.add_argument('--suffix', default='')
    ap.add_argument('--out', default=str(paths.OUTPUTS_DIR / 'overnight_2026-11-10'))
    ap.add_argument('--exclude-legacy', action='store_true', help='dry run only: force a v2 winner')
    a = ap.parse_args()
    res = select(a.gauge, a.suffix, a.exclude_legacy)
    out = paths.ensure_dir(pathlib.Path(a.out))
    (out / f'selection_{a.gauge}{a.suffix}.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
    print(json.dumps({k: res[k] for k in ('gauge', 'scores', 'winner', 'missing')}, indent=2))
