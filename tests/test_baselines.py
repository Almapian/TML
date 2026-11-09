"""Checks for utils/models/baselines.py against reference libraries and synthetic signals.

    python tests/test_baselines.py
"""
import pathlib
import sys
import warnings

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_toeplitz
from sklearn.linear_model import Ridge
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf

from utils.models import baselines as B

STEPS = np.array([1, 2, 6, 24])
RNG = np.random.default_rng(0)


def fake_series(y, utide=None):
    n = len(y)
    time = np.datetime64('2000-01-01T00:00') + np.arange(n) * np.timedelta64(10, 'm')
    return B.Series('test', time, y.copy(), np.zeros(n) if utide is None else utide,
                    np.zeros(n, dtype=bool), n)


def test_acf_pacf_match_statsmodels():
    y = arma_generate_sample([1, -0.7, 0.2], [1, 0.4], 20000, distrvs=RNG.standard_normal)
    cov = B.gappy_acov(y, 30)
    biased = cov * (len(y) - np.arange(31)) / len(y)
    assert np.allclose(biased / biased[0], acf(y, nlags=30, fft=True), atol=1e-8)
    assert np.allclose(B.durbin_levinson(biased, 30)[0], pacf(y, nlags=30, method='ldb')[1:], atol=1e-6)


def test_css_and_forecast_match_sarimax():
    y = arma_generate_sample([1, -0.8], [1, 0.4], 6000, distrvs=RNG.standard_normal)
    order = B.SarimaOrder(p=1, q=1, season=(12, 40))
    params, _, _ = B.fit_sarima_css(y, order)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = SARIMAX(y, order=(1, 0, 1), trend='c').fit(disp=False)
    assert abs(params[0] - ref.params[1]) < 0.02 and abs(params[1] - ref.params[2]) < 0.02, (params, ref.params)
    model = B.linear_from_pi(B.sarima_pi(order, ref.params[1:3], 400), y, STEPS)
    series = fake_series(y)
    mine = model.forecast(series, 'level', np.array([len(y) - 25]), STEPS)[0]
    fixed = SARIMAX(y[:-24], order=(1, 0, 1), trend='c').smooth(ref.params)
    theirs = fixed.forecast(24)[STEPS - 1]
    assert np.max(np.abs(mine - theirs)) < 0.02, (mine, theirs)


def test_seasonal_differencing_polys():
    order = B.SarimaOrder(p=1, D=1, season=(10, 40))
    ar, ma = B.sarima_polys(order, np.array([0.5]))
    assert ar == {0: 1.0, 1: -0.5, 10: -1.0, 11: 0.5} and ma == {0: 1.0}


def test_ar_recovers_coefficients_with_gaps():
    y = arma_generate_sample([1, -1.2, 0.5], [1], 200000, distrvs=RNG.standard_normal)
    y[5000:5100] = np.nan
    for estimator in ('yw', 'burg'):
        assert np.allclose(B.fit_ar(y, 2, estimator), [1.2, -0.5], atol=0.02), estimator


def test_gp_matches_dense_cholesky():
    kernel = B.SpectralMixture(np.array([1.0, 0.3]), np.array([0.05, 0.11]), np.array([1e-5, 1e-5]),
                               0.2, 0.05, 0.01)
    y = RNG.normal(size=500)
    n_lags = 60
    model = B.gp_forecaster(kernel, y, n_lags, steps=STEPS)
    r = kernel.acov(n_lags + 30)
    K = r[np.abs(np.subtract.outer(np.arange(n_lags), np.arange(n_lags)))]
    k_star = np.stack([r[h:h + n_lags] for h in STEPS], axis=1)
    assert np.allclose(model.weights.T, cho_solve(cho_factor(K), k_star), atol=1e-6)


def test_ridge_matches_sklearn():
    y = arma_generate_sample([1, -0.9], [1], 5000, distrvs=RNG.standard_normal)
    series = fake_series(y)
    lags = np.arange(1, 9)
    gram = B.ridge_gram(series, 'level', lags, steps=STEPS)
    model = B.ridge_forecaster(gram, y, 0.1)
    z = y - np.nanmean(y)
    anchors = np.arange(7, len(y) - 24, B.C.BASELINE_RIDGE_TRAIN_STRIDE)
    X = B._gather(z, anchors, lags)
    target = z[anchors + 6]
    ref = Ridge(alpha=0.1 * float(np.mean(gram.values)), fit_intercept=False).fit(X, target)
    assert np.allclose(model.weights[2], ref.coef_, atol=1e-5), (model.weights[2], ref.coef_)


def test_dynamic_harmonic_tracks_a_cycle():
    n, f = 30000, 1 / 74.5
    t = np.arange(n)
    y = (2 * np.cos(2 * np.pi * f * t + 0.4) + 0.5 * np.cos(2 * np.pi * f * 2.03 * t)
         + RNG.normal(scale=0.05, size=n))
    model = B.fit_dynamic_harmonic(y[:20000], np.array([f, f * 2.03]), 2, 1e-4, 1.0, 0.5)
    series = fake_series(y)
    anchors = np.arange(21000, 29000, 100)
    pred = model.forecast(series, 'level', anchors, STEPS)
    assert np.sqrt(np.mean((pred - y[anchors[:, None] + STEPS]) ** 2)) < 0.1


def test_dynamic_harmonic_matches_statsmodels_filter():
    """After burn-in the steady-state forecast must equal an exact Kalman filter's."""
    n = 4000
    y = np.cos(2 * np.pi * 0.03 * np.arange(n)) + 0.3 * arma_generate_sample([1, -0.6], [1], n, distrvs=RNG.standard_normal)
    model = B.fit_dynamic_harmonic(y[:3000], np.array([0.03]), 2, 1e-3, 0.999, 0.2)
    series = fake_series(y)
    anchor = np.array([3900])
    mine = model.forecast(series, 'level', anchor, STEPS)[0] - model.mean

    m = len(model.Z)
    remainder, _ = B.harmonic_remainder(y[:3000], np.array([0.03]))
    acov = B.gappy_acov(remainder, 2)
    acov[0] *= 1 + 1e-6
    phi = solve_toeplitz(acov[:2], acov[1:3])
    innovation = acov[0] - phi @ acov[1:3]
    Q = np.diag([1e-3 * innovation, 1e-3 * innovation, innovation, 0.0])
    ss = MLEModel(y[:3901] - model.mean, k_states=m, k_posdef=m, initialization='approximate_diffuse')
    ss.ssm['design'] = model.Z[None, :]
    ss.ssm['transition'] = model.T
    ss.ssm['selection'] = np.eye(m)
    ss.ssm['state_cov'] = Q
    ss.ssm['obs_cov'] = np.array([[0.2 * innovation]])
    state = ss.ssm.filter().filtered_state[:, 3900]
    theirs = [model.Z @ np.linalg.matrix_power(model.T, int(h)) @ state for h in STEPS]
    assert np.allclose(mine, theirs, atol=1e-6), (mine, theirs)


def test_spectral_peaks_recover_frequencies():
    n, freqs = 2 ** 17, np.array([1 / 74.52, 1 / 72.0, 1 / 143.6])
    t = np.arange(n)
    y = sum(a * np.cos(2 * np.pi * f * t) for a, f in zip([3, 1, 0.5], freqs)) + RNG.normal(scale=0.1, size=n)
    found = B.spectral_peaks(*B.periodogram(y), 3)['freq_cycles_per_step'].to_numpy()
    assert np.allclose(np.sort(found), np.sort(freqs), atol=2e-6), (found, freqs)


def test_forecast_ignores_the_future():
    y = arma_generate_sample([1, -0.8], [1], 3000, distrvs=RNG.standard_normal)
    anchors = np.array([1500, 2000])
    model = B.linear_from_pi(np.array([0.5, 0.2]), y, STEPS)
    series = fake_series(y)
    before = model.forecast(series, 'level', anchors, STEPS)
    series.level[2001:] = 99.0
    assert np.array_equal(before, model.forecast(series, 'level', anchors, STEPS))
    harmonic = B.fit_dynamic_harmonic(y[:1000], np.array([0.01]), 2, 1e-3, 0.999, 0.1)
    series = fake_series(y)
    before = harmonic.forecast(series, 'level', anchors, STEPS)
    series.level[2001:] = 99.0
    assert np.array_equal(before, harmonic.forecast(series, 'level', anchors, STEPS))


def test_gap_fill_uses_only_the_past():
    z = np.array([1.0, 2.0, np.nan, np.nan, 50.0])
    out = B.fill_gaps(z, np.array([0.5]), np.array([1]), -10.0, 10.0)
    assert out[2] == 1.0 and out[3] == 0.5 and out[4] == 50.0


def test_valid_anchors_match_window_rule():
    y = np.ones(1000)
    y[400:410] = np.nan
    series = fake_series(y)
    anchors = B.valid_anchors(series, '2000-01-01', '2001-01-01', lookback=96, max_step=100)
    assert anchors[0] == 95 and anchors[-1] == 999 - 100
    assert not np.any((anchors > 299) & (anchors < 505))


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
