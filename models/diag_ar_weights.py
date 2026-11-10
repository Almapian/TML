"""Where the AR residual baseline's forecast weight sits, by lag window. No training, no scoring.

The long AR forecast at lead h is a fixed weighted sum of past residuals (observed level minus
the harmonic prediction). Summing those weights over lag windows shows what the model relies on:
the last hour, the last day, the same tidal phase a lunar day earlier (lag 149 = 2 x M2), or the
weeks before. The total is the share of a sustained offset (for example a raised mean level) that
carries into the forecast. Frozen config, refit on train only, exactly as eval_baselines does.

    python models/diag_ar_weights.py
Output: outputs/diagnostics_2026-11/ar_weight_windows.csv
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

STEPS = np.array([1, 6, 36, 144, 288, 432])   # 10 min, 1h, 6h, 24h, 48h, 72h


def windows(lag):
    return {'last 1h': lag <= 6, '1h to 12h': (lag > 6) & (lag <= 72), '12h to 22.7h': (lag > 72) & (lag <= 136),
            'one lunar day earlier (lag 149 +-2h)': np.abs(lag - 149) <= 12,
            '1 to 30 days, other': (lag > 161) & (lag <= 4320) & (np.abs(lag - 4174) > 12),
            'about 29 days earlier (lag 4174 +-2h)': np.abs(lag - 4174) <= 12,
            'over 30 days': lag > 4320}


rows = []
for g in C.GAUGES:
    s = B.load_series(g, verbose=False)
    cfg = json.loads(paths.baseline_config_path(g).read_text(encoding='utf-8'))['ar_residual']['config']
    y = s.track('residual')[:s.n_train]
    phi = B.fit_ar(y, cfg['order'], cfg['estimator'], cfg.get('jitter', 1e-5))
    m = B.linear_from_pi(phi, y, steps=STEPS)
    for i, st in enumerate(STEPS):
        r = {'gauge': g, 'order': cfg['order'], 'lead_h': st / 6}
        r.update({k: float(m.weights[i, v].sum()) for k, v in windows(m.lags).items()})
        r['total'] = float(m.weights[i].sum())
        rows.append(r)
df = pd.DataFrame(rows)
out = paths.ensure_dir(paths.OUTPUTS_DIR / 'diagnostics_2026-11') / 'ar_weight_windows.csv'
df.to_csv(out, index=False)
with pd.option_context('display.width', 250):
    print(df.round(3).to_string(index=False))
