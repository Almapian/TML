"""Checks for the dense v2 variants (residual-persistence skip).

    python tests/test_dense_v2.py
"""
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from utils.models import config as C
from utils.models import datasets, windowing
from utils.models.architectures import TidalSeq2Seq, residual_persistence

torch.manual_seed(0)
N, L, H = 16, C.LOOKBACK, len(C.DENSE_HORIZONS)
STEPS = list(C.DENSE_HORIZONS.values())


def _inputs(h=H):
    return torch.randn(N, L), torch.randn(N, h), torch.randn(N)


def _zero_network(model):
    with torch.no_grad():
        model.decoder.out.weight.zero_()
        model.decoder.out.bias.zero_()


def test_skip_with_zero_network_is_residual_persistence():
    x, u, ua = _inputs()
    m = TidalSeq2Seq(residual_skip=True)
    _zero_network(m)
    pred, _ = m(x, u, ua)
    expected = u + (x[:, -1] - ua)[:, None]          # UTide(t+k) + obs(t) - UTide(t)
    assert torch.equal(pred, expected), 'zeroed network should give residual persistence exactly'


def test_skip_needs_anchor_utide():
    x, u, _ = _inputs()
    try:
        TidalSeq2Seq(residual_skip=True)(x, u)
    except ValueError:
        return
    raise AssertionError('residual_skip without utide_anchor must raise')


def test_default_model_unchanged():
    """Flags off: the output is the plain decoder readout, and legacy weights load."""
    x, u, ua = _inputs()
    torch.manual_seed(1); a = TidalSeq2Seq()
    torch.manual_seed(1); b = TidalSeq2Seq(residual_skip=False)
    assert torch.equal(a(x, u)[0], b(x, u, ua)[0])
    ckpt = REPO_ROOT / 'models' / 'checkpoints' / 'dense' / 'stage2_dense_seq2seq.pt'
    if ckpt.exists():
        sd = torch.load(ckpt, map_location='cpu', weights_only=False)['model_state_dict']
        for flags in ({}, {'residual_skip': True}):
            TidalSeq2Seq(**flags).load_state_dict(sd)      # no new parameters in any variant


def test_skip_uses_no_data_after_anchor():
    """The base forecast depends only on x_hist[:, -1] (the anchor) and UTide."""
    x, u, ua = _inputs()
    base = residual_persistence(x, u, ua)
    x2 = x.clone(); x2[:, :-1] += 5.0                     # change history before the anchor
    assert torch.equal(residual_persistence(x2, u, ua), base)
    x3 = x.clone(); x3[:, -1] += 1.0                      # change the anchor value itself
    assert torch.allclose(residual_persistence(x3, u, ua), base + 1.0, atol=1e-6)


def _toy_df(n_days=8, gap_at=None):
    t = pd.date_range('2015-01-01', periods=n_days * 144, freq='10min')
    df = pd.DataFrame({'DateTime': t, 'Observed_ODN': np.sin(np.arange(len(t)) / 7.0).astype(np.float32),
                       'is_flagged': np.zeros(len(t), bool)})
    df['utide_recon'] = (np.arange(len(t)) * 1e-3).astype(np.float32)      # marker = row index / 1000
    if gap_at is not None:
        df = df.drop(index=range(gap_at, gap_at + 3)).reset_index(drop=True)
    return df


def test_anchor_utide_is_at_the_anchor():
    df = _toy_df(gap_at=600)
    X, Yd, Y, F, T, Ua = windowing.build_windows_multi(df, L, STEPS, return_anchor_utide=True)
    anchor = T[:, 0] - np.timedelta64(10, 'm')
    rows = ((anchor - df.DateTime.iloc[0].to_datetime64()) / np.timedelta64(10, 'm')).astype(int)
    assert np.allclose(Ua, rows * 1e-3), 'Uanc is not UTide at the anchor'
    assert np.array_equal(X[:, -1], df.set_index('DateTime').loc[anchor, 'Observed_ODN'].to_numpy())
    X0, _, _, _, T0 = windowing.build_windows_multi(df, L, STEPS)
    assert np.array_equal(X0, X) and np.array_equal(T0, T), 'return_anchor_utide changed the windows'


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f'ok    {name}')
        except Exception as exc:
            failed += 1
            print(f'FAIL  {name}: {type(exc).__name__}: {exc}')
    print(f'{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
