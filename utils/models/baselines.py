"""Classical forecasters scored on the dense pipeline's anchors, lead times and masks.

Everything here is fitted on the training split with frozen parameters, then conditions on
observations up to each anchor, as the neural encoders do. Two tracks share the code:

    level     forecast Observed_ODN
    residual  forecast Observed_ODN - UTide, then add UTide back

The error of a residual forecast equals the error of its level forecast, so both tracks are
scored in their own units.

AR, SARIMA, the GP and ridge are all fixed linear filters, so they share one engine: a weight
matrix W (leads x lags), with forecast = mean + W @ history (`LinearForecaster`). Dynamic
harmonic regression has a recursive state instead (`DynamicHarmonic`).

`load_series(upto=...)` cuts the arrays before anything sees them, so the identification and
search stages cannot reach test data by accident.
"""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.linalg import eigh, solve_discrete_are, solve_toeplitz
from scipy.optimize import least_squares
from scipy.signal import find_peaks, get_window

from .. import paths
from . import config as C
from .datasets import fit_utide, load_gauge_df, reconstruct_utide

DENSE_STEPS = np.array(list(C.DENSE_HORIZONS.values()), dtype=np.int64)
MIN_CYCLE_FREQ = 1.0 / (45 * C.STEPS_PER_DAY)   # cycles/step; slower "peaks" are not cycles here
HANN_LINE_SD = 0.72                              # a pure line's sd in the Hann periodogram, in bins
OU_RATE_RANGE = (1.0 / (20 * C.STEPS_PER_DAY), 1.0)   # per step: broadband memory of 20 days to 10 min


# ======================================================================================
# Data on a regular 10-minute grid
# ======================================================================================
@dataclass
class Series:
    gauge: str
    time: np.ndarray        # datetime64 grid, every 10 minutes
    level: np.ndarray       # Observed_ODN, NaN where the gauge has no row
    utide: np.ndarray       # train-only UTide reconstruction, gap-free
    flagged: np.ndarray     # chatter/stuck/imputed rows, as in datasets.load_gauge_df
    n_train: int            # grid index of the first row at or after TRAIN_END

    def track(self, track):
        return self.level.copy() if track == 'level' else self.level - self.utide


def _utide_on_grid(gauge, df, grid):
    """UTide reconstruction across the grid, using the neural models' exact train-only fit."""
    cache = paths.ensure_dir(paths.baseline_output_dir(gauge)) / 'utide_recon_grid.npy'
    if cache.exists():
        recon = np.load(cache)
        if len(recon) == len(grid):
            return recon
    info = C.gauge_info(gauge)
    coef = fit_utide(df, end=C.TRAIN_END, lat=info['lat'], stride=info['utide_fit_stride'])
    recon = reconstruct_utide(grid, coef)
    np.save(cache, recon)
    return recon


def load_series(gauge, upto=None, verbose=True):
    """The gauge on a regular grid, cut to rows before `upto` when given."""
    df = load_gauge_df(C.gauge_info(gauge)['csv'], verbose=verbose)
    grid = pd.date_range(df['DateTime'].min(), df['DateTime'].max(), freq=C.STEP)
    indexed = df.set_index('DateTime')
    level = indexed['Observed_ODN'].reindex(grid).to_numpy(dtype=np.float64)
    flagged = indexed['is_flagged'].astype(bool).reindex(grid, fill_value=False).to_numpy()
    utide_recon = _utide_on_grid(gauge, df, grid).astype(np.float64)

    keep = len(grid) if upto is None else int(np.searchsorted(grid.values, np.datetime64(upto)))
    time = grid.values[:keep]
    n_train = int(np.searchsorted(time, np.datetime64(C.TRAIN_END)))
    return Series(gauge, time, level[:keep], utide_recon[:keep], flagged[:keep], n_train)


def valid_anchors(series, start, end, lookback=C.LOOKBACK, max_step=int(DENSE_STEPS.max())):
    """Grid indices of the anchors `windowing.build_windows_multi` makes in [start, end):
    inside one gap-free run, with `lookback` rows behind and `max_step` rows ahead."""
    lo = int(np.searchsorted(series.time, np.datetime64(start)))
    hi = int(np.searchsorted(series.time, np.datetime64(end)))
    present = ~np.isnan(series.level[lo:hi])
    edges = np.flatnonzero(np.diff(np.concatenate([[0], present.astype(np.int8), [0]])))
    out = []
    for a, b in zip(edges[::2], edges[1::2]):
        first, last = a + lookback - 1, b - max_step - 1
        if last >= first:
            out.append(np.arange(first, last + 1))
    return lo + np.concatenate(out)


def split_anchors(series, split):
    start, end = {'val': (C.TRAIN_END, C.VAL_END), 'test': (C.VAL_END, '2100-01-01')}[split]
    return valid_anchors(series, start, end)


def targets(y, flagged, anchors, steps=DENSE_STEPS):
    """(N, H) target values and flags at `steps` after each anchor."""
    positions = anchors[:, None] + steps[None, :]
    return y[positions], flagged[positions]


def mean_lead_mse(pred, true, flag):
    """Validation criterion: masked MSE per lead, averaged with equal weight per lead."""
    err = np.where(flag, np.nan, pred - true)
    return float(np.mean(np.nanmean(err ** 2, axis=0)))


# ======================================================================================
# Second-order statistics
# ======================================================================================
def gappy_acov(y, maxlag):
    """Autocovariance of a NaN-holed series about its mean, each lag normalised by the number
    of pairs actually present. O(n log n) through the FFT."""
    mask = (~np.isnan(y)).astype(np.float64)
    z = np.nan_to_num(y - np.nanmean(y))
    nfft = 2 ** int(np.ceil(np.log2(2 * len(z))))
    fz, fm = np.fft.rfft(z, nfft), np.fft.rfft(mask, nfft)
    pairs = np.maximum(np.round(np.fft.irfft(np.abs(fm) ** 2)[:maxlag + 1]), 1)
    return np.fft.irfft(np.abs(fz) ** 2)[:maxlag + 1] / pairs


@njit(cache=True)
def durbin_levinson(acov, maxlag):
    """(pacf[1..maxlag], innovation variance[0..maxlag]) from an autocovariance."""
    phi = np.zeros(maxlag + 1)
    prev = np.zeros(maxlag + 1)
    pacf = np.zeros(maxlag + 1)
    sigma2 = np.zeros(maxlag + 1)
    sigma2[0] = acov[0]
    for k in range(1, maxlag + 1):
        num = acov[k]
        for j in range(1, k):
            num -= prev[j] * acov[k - j]
        ref = num / sigma2[k - 1]
        pacf[k] = ref
        phi[k] = ref
        for j in range(1, k):
            phi[j] = prev[j] - ref * prev[k - j]
        sigma2[k] = sigma2[k - 1] * (1.0 - ref * ref)
        prev[1:k + 1] = phi[1:k + 1]
    return pacf[1:], sigma2


@njit(cache=True)
def _burg(z, starts, stops, order):
    """Burg AR coefficients over several gap-free segments of one array."""
    forward = z.copy()
    backward = z.copy()
    phi = np.zeros(order + 1)
    prev = np.zeros(order + 1)
    for m in range(1, order + 1):
        num = 0.0
        den = 0.0
        for s in range(len(starts)):
            for t in range(starts[s] + m, stops[s]):
                num += forward[t] * backward[t - 1]
                den += forward[t] ** 2 + backward[t - 1] ** 2
        ref = 2.0 * num / den if den > 0 else 0.0
        phi[m] = ref
        for j in range(1, m):
            phi[j] = prev[j] - ref * prev[m - j]
        prev[1:m + 1] = phi[1:m + 1]
        for s in range(len(starts)):
            for t in range(stops[s] - 1, starts[s] + m - 1, -1):
                f_old = forward[t]
                forward[t] = f_old - ref * backward[t - 1]
                backward[t] = backward[t - 1] - ref * f_old
    return phi[1:]


def segments(y, min_len=1):
    """(starts, stops) of the non-NaN runs of `y` at least `min_len` long."""
    present = (~np.isnan(y)).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], present, [0]])))
    starts, stops = edges[::2], edges[1::2]
    keep = stops - starts >= min_len
    return starts[keep].astype(np.int64), stops[keep].astype(np.int64)


def fill_for_spectrum(y, order=449):
    """A gap-free copy of `y` for spectra and ridge design matrices only, with each gap
    filled by an AR forecast. Never used on the series being forecast."""
    mu = np.nanmean(y)
    acov = gappy_acov(y, order)
    acov[0] *= 1 + 1e-5
    phi = solve_toeplitz(acov[:order], acov[1:order + 1])
    z = fill_gaps(y - mu, phi, np.arange(1, order + 1), float(np.nanmin(y) - mu), float(np.nanmax(y) - mu))
    return z + mu


def periodogram(y):
    """(frequency in cycles per step, one-sided power) of the gap-filled series, Hann window."""
    z = fill_for_spectrum(y)
    z = (z - z.mean()) * get_window('hann', len(z))
    power = np.abs(np.fft.rfft(z)) ** 2 / np.sum(get_window('hann', len(z)) ** 2)
    return np.fft.rfftfreq(len(z)), power


def spectral_peaks(freq, power, count):
    """The `count` largest spectral lines, refined by parabolic interpolation of log power.
    Returns a table of frequency, period (h) and relative power."""
    log_power = np.log(np.maximum(power, 1e-30))
    idx, _ = find_peaks(log_power, distance=4)
    idx = idx[(freq[idx] > MIN_CYCLE_FREQ) & (idx > 0) & (idx < len(freq) - 1)]
    idx = idx[np.argsort(power[idx])[::-1][:count]]
    a, b, c = log_power[idx - 1], log_power[idx], log_power[idx + 1]
    shift = 0.5 * (a - c) / (a - 2 * b + c)
    f = freq[idx] + shift * (freq[1] - freq[0])
    return pd.DataFrame({'freq_cycles_per_step': f, 'period_h': 1 / (6 * f),
                         'relative_power': power[idx] / power[1:].sum()})


# ======================================================================================
# Linear engine
# ======================================================================================
@njit(cache=True)
def fill_gaps(z, one_step, lags, low, high, start=0):
    """Replace NaNs left to right with the model's one-step forecast, so a filled value only
    ever uses earlier data. Missing history before the series starts counts as the mean."""
    out = z.copy()
    for t in range(start, len(out)):
        if np.isnan(out[t]):
            value = 0.0
            for k in range(len(lags)):
                if t - lags[k] >= 0:
                    previous = out[t - lags[k]]
                    if not np.isnan(previous):
                        value += one_step[k] * previous
            out[t] = min(max(value, low), high)
    return out


@njit(parallel=True, cache=True)
def _apply_weights(z, anchors, weights, lags):
    n, n_leads, n_lags = len(anchors), weights.shape[0], weights.shape[1]
    out = np.zeros((n, n_leads))
    for i in prange(n):
        a = anchors[i]
        for h in range(n_leads):
            total = 0.0
            row = weights[h]
            for k in range(n_lags):
                index = a + 1 - lags[k]
                if index >= 0:
                    total += row[k] * z[index]
            out[i, h] = total
    return out


@dataclass
class LinearForecaster:
    weights: np.ndarray     # (leads, lags)
    lags: np.ndarray        # 1 = the value at the anchor, 2 = one step before it, ...
    mean: float
    low: float              # train range about the mean, bounds the gap filling
    high: float
    beta: np.ndarray = None           # ridge only: coefficient per lead on UTide at the target
    utide_mean: float = 0.0

    def forecast(self, series, track, anchors, steps=DENSE_STEPS):
        """(N, H) forecasts in track units, conditioning only on data up to each anchor."""
        z = fill_gaps(series.track(track) - self.mean, self.weights[0], self.lags, self.low, self.high,
                      max(0, int(anchors.min()) - int(self.lags.max()) - 1))
        out = _apply_weights(z, anchors, self.weights, self.lags) + self.mean
        if self.beta is not None:
            out += self.beta[None, :] * (series.utide[anchors[:, None] + steps[None, :]] - self.utide_mean)
        return out


@njit(cache=True)
def pi_to_weights(pi, steps):
    """h-step weights of an AR(inf) truncated to len(pi) lags: iterate the one-step forecast,
    replacing unknown future values by their own forecasts."""
    n_lags = len(pi)
    max_h = int(steps.max())
    full = np.zeros((max_h + 1, n_lags))
    for h in range(1, max_h + 1):
        row = full[h]
        for j in range(n_lags):
            if h - 1 + j < n_lags:
                row[j] = pi[h - 1 + j]
        for k in range(1, min(h, n_lags + 1)):
            coef = pi[k - 1]
            if coef != 0.0:
                earlier = full[h - k]
                for j in range(n_lags):
                    row[j] += coef * earlier[j]
    return full[steps]


def _train_bounds(y, mean):
    return float(np.nanmin(y) - mean), float(np.nanmax(y) - mean)


def linear_from_pi(pi, y_train, steps=DENSE_STEPS):
    mean = float(np.nanmean(y_train))
    low, high = _train_bounds(y_train, mean)
    return LinearForecaster(pi_to_weights(np.asarray(pi, dtype=np.float64), steps),
                            np.arange(1, len(pi) + 1), mean, low, high)


# ======================================================================================
# Method 1a: long AR
# ======================================================================================
def fit_ar(y_train, order, estimator='yw', jitter=1e-5):
    """AR(order) coefficients. 'yw' solves Yule-Walker on the gap-aware autocovariance,
    'burg' runs Burg across gap-free segments (always a stable model)."""
    mean = np.nanmean(y_train)
    if estimator == 'burg':
        starts, stops = segments(y_train, min_len=order + 1)
        return _burg(np.nan_to_num(y_train - mean), starts, stops, order)
    acov = gappy_acov(y_train, order)
    acov[0] *= 1 + jitter
    return solve_toeplitz(acov[:order], acov[1:order + 1])


# ======================================================================================
# Method 1b: SARIMA by conditional sum of squares
# ======================================================================================
@dataclass(frozen=True)
class SarimaOrder:
    p: int = 1
    d: int = 0
    q: int = 0
    P: int = 0       # seasonal AR at season[0]
    D: int = 0
    Q: int = 0
    P2: int = 0      # seasonal AR at season[1]
    D2: int = 0
    season: tuple = C.BASELINE_SEASONS

    @property
    def n_params(self):
        return self.p + self.q + self.P + self.Q + self.P2


def _sparse_mul(a, b):
    out = {}
    for la, ca in a.items():
        for lb, cb in b.items():
            out[la + lb] = out.get(la + lb, 0.0) + ca * cb
    return out


def _lag_poly(coefs, spacing, sign):
    """{0: 1, spacing: sign*c1, 2*spacing: sign*c2, ...}."""
    poly = {0: 1.0}
    for i, c in enumerate(coefs, start=1):
        poly[i * spacing] = sign * float(c)
    return poly


def sarima_polys(order, params):
    """(ar, ma) lag polynomials of the fully expanded model, as sparse {lag: coefficient}."""
    i = 0
    take = {}
    for name, size in [('p', order.p), ('q', order.q), ('P', order.P), ('Q', order.Q), ('P2', order.P2)]:
        take[name] = params[i:i + size]
        i += size
    s1, s2 = order.season
    ar = _lag_poly(take['p'], 1, -1.0)
    ar = _sparse_mul(ar, _lag_poly(take['P'], s1, -1.0))
    ar = _sparse_mul(ar, _lag_poly(take['P2'], s2, -1.0))
    for lag, power in [(1, order.d), (s1, order.D), (s2, order.D2)]:
        for _ in range(power):
            ar = _sparse_mul(ar, {0: 1.0, lag: -1.0})
    ma = _sparse_mul(_lag_poly(take['q'], 1, 1.0), _lag_poly(take['Q'], s1, 1.0))
    return ar, ma


@njit(cache=True)
def _css_residuals(z, starts, stops, ar_lags, ar_coefs, ma_lags, ma_coefs, burn):
    """One-step errors, with pre-sample errors at zero (the CSS assumption) in each segment.
    e[t] = sum(ar[j] z[t-j]) - sum(ma[j] e[t-j]), for j > 0 in ma."""
    e = np.zeros(len(z))
    for s in range(len(starts)):
        for t in range(starts[s] + burn, stops[s]):
            value = 0.0
            for k in range(len(ar_lags)):
                value += ar_coefs[k] * z[t - ar_lags[k]]
            for k in range(len(ma_lags)):
                lag_index = t - ma_lags[k]
                if lag_index >= starts[s] + burn:
                    value -= ma_coefs[k] * e[lag_index]
            e[t] = max(min(value, 1e3), -1e3)
    return e


def _as_arrays(poly, skip_zero_lag):
    items = sorted((lag, coef) for lag, coef in poly.items() if coef != 0.0 and not (skip_zero_lag and lag == 0))
    return (np.array([lag for lag, _ in items], dtype=np.int64),
            np.array([coef for _, coef in items], dtype=np.float64))


def fit_sarima_css(y_train, order, max_nfev=60):
    """Conditional-sum-of-squares fit. Returns (params, sigma2, n_used). Non-stationary or
    non-invertible regions are not excluded by constraint; they just score badly."""
    z = np.nan_to_num(y_train - np.nanmean(y_train))
    burn = max(sarima_polys(order, np.zeros(order.n_params))[0]) + order.q + order.Q * order.season[0]
    starts, stops = segments(y_train, min_len=burn + 1)
    used = np.zeros(len(z), dtype=bool)
    for start, stop in zip(starts, stops):
        used[start + burn:stop] = True

    def residual(params):
        ar, ma = sarima_polys(order, params)
        return _css_residuals(z, starts, stops, *_as_arrays(ar, False), *_as_arrays(ma, True), burn)[used]

    params = np.zeros(order.n_params)
    if order.n_params:
        params[0] = 0.5 if order.p else 0.0
        params = least_squares(residual, params, bounds=(-2.5, 2.5), method='trf', x_scale=0.2,
                               max_nfev=max_nfev, ftol=1e-7, xtol=1e-7, gtol=1e-7).x
    e = residual(params)
    return params, float(np.mean(e ** 2)), len(e)


def sarima_pi(order, params, n_lags):
    """pi[1..n_lags] of the expanded model written as z[t] = sum(pi[k] z[t-k]) + e[t]."""
    ar, ma = sarima_polys(order, params)
    size = max(max(ar), max(ma), n_lags) + 1
    ar_dense = np.zeros(size)
    ma_dense = np.zeros(size)
    for lag, coef in ar.items():
        ar_dense[lag] = coef
    for lag, coef in ma.items():
        ma_dense[lag] = coef
    return -_power_series_ratio(ar_dense, ma_dense, n_lags)[1:]


@njit(cache=True)
def _power_series_ratio(numer, denom, n_lags):
    """Coefficients of numer(B) / denom(B) up to B^n_lags, with denom[0] = 1."""
    out = np.zeros(n_lags + 1)
    nonzero = np.flatnonzero(denom[1:]) + 1
    for k in range(n_lags + 1):
        value = numer[k]
        for j in nonzero:
            if j > k:
                break
            value -= denom[j] * out[k - j]
        out[k] = max(min(value, 1e6), -1e6)
    return out


# ======================================================================================
# Method 2: Gaussian process (spectral mixture + OU + nugget)
# ======================================================================================
@dataclass
class SpectralMixture:
    """k(tau) = sum_q w_q exp(-2 pi^2 v_q tau^2) cos(2 pi mu_q tau) + a exp(-theta |tau|)
    + nugget delta(tau), with tau in 10-minute steps and mu in cycles per step."""
    weights: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    ou_var: float
    ou_rate: float
    nugget: float

    def acov(self, n_lags):
        tau = np.arange(n_lags, dtype=np.float64)
        k = np.sum(self.weights[:, None] * np.exp(-2 * np.pi ** 2 * self.variances[:, None] * tau ** 2)
                   * np.cos(2 * np.pi * self.means[:, None] * tau), axis=0)
        k = k + self.ou_var * np.exp(-self.ou_rate * tau)
        k[0] += self.nugget
        return k


def welch_spectrum(y, nperseg=2 ** 16):
    """Averaged Hann periodogram of the gap-filled series, as a density (cycles per step)."""
    z = fill_for_spectrum(y)
    z = z - z.mean()
    window = get_window('hann', nperseg)
    hop = nperseg // 2
    n_blocks = (len(z) - nperseg) // hop + 1
    power = np.zeros(nperseg // 2 + 1)
    for b in range(n_blocks):
        power += np.abs(np.fft.rfft(z[b * hop:b * hop + nperseg] * window)) ** 2
    density = 2 * power / (n_blocks * np.sum(window ** 2))
    return np.fft.rfftfreq(nperseg), density


def fit_spectral_mixture(y_train, n_components, centres='record', restarts=3, iterations=250, seed=C.SEED):
    """Hyperparameters by the Whittle likelihood of the averaged train periodogram, so the
    whole 14-year record informs the kernel at O(n log n), which exact GP likelihood cannot.

    Line centres are fixed, so widths cannot grow to make up for a mis-centred line. 'record'
    takes them from the full-record periodogram, sharpest for pure astronomical lines; 'welch'
    from the averaged spectrum, better where river flow smears a line into a bump. The Hann
    window's own width is added to each line in the model of the spectrum and left out
    of the kernel, so the kernel keeps the line's true width and hence its long memory."""
    import torch
    freq, density = welch_spectrum(y_train)
    keep = freq > 0
    f = torch.tensor(freq[keep], dtype=torch.float64)
    target = torch.tensor(np.maximum(density[keep], 1e-14), dtype=torch.float64)
    df = freq[1] - freq[0]
    window_sd = HANN_LINE_SD * df
    peaks = spectral_peaks(*(periodogram(y_train) if centres == 'record' else (freq, density)), n_components)
    mu = torch.tensor(peaks['freq_cycles_per_step'].to_numpy(), dtype=torch.float64)
    n = len(peaks)
    total_var = float(np.nanvar(y_train))
    rate_low, rate_high = OU_RATE_RANGE
    best, best_loss = None, np.inf

    def spectrum(raw):
        w = torch.exp(raw['log_w'])
        sd = torch.sqrt(torch.exp(2 * raw['log_sd']) + window_sd ** 2)
        lines = (w[:, None] / (sd[:, None] * np.sqrt(2 * np.pi))
                 * torch.exp(-0.5 * ((f[None, :] - mu[:, None]) / sd[:, None]) ** 2)).sum(dim=0)
        rate = rate_low + (rate_high - rate_low) * torch.sigmoid(raw['raw_rate'])
        ou = torch.exp(raw['log_ou']) * 4 * rate / (rate ** 2 + (2 * np.pi * f) ** 2)   # one-sided
        return lines + ou + torch.exp(raw['log_nugget']) * 2

    for restart in range(restarts):
        spread = 0.0 if restart == 0 else 0.5
        noise = np.random.default_rng(seed + restart).normal(size=n)
        raw = {
            'log_w': torch.tensor(np.log(np.maximum(peaks['relative_power'].to_numpy(), 1e-6) * total_var) + spread * noise),
            'log_sd': torch.full((n,), float(np.log(window_sd * 0.1 * (1 + restart))), dtype=torch.float64),
            'log_ou': torch.tensor(float(np.log(0.1 * total_var)), dtype=torch.float64),
            'raw_rate': torch.tensor(-2.0 + restart, dtype=torch.float64),
            'log_nugget': torch.tensor(float(np.log(1e-5 * total_var)), dtype=torch.float64),
        }
        for value in raw.values():
            value.requires_grad_(True)
        optimiser = torch.optim.LBFGS(list(raw.values()), lr=0.5, max_iter=iterations, history_size=30,
                                      line_search_fn='strong_wolfe')

        def closure():
            optimiser.zero_grad()
            s = spectrum(raw)
            loss = torch.mean(torch.log(s) + target / s)
            loss.backward()
            return loss

        optimiser.step(closure)
        with torch.no_grad():
            s = spectrum(raw)
            loss = float(torch.mean(torch.log(s) + target / s))
        if np.isfinite(loss) and loss < best_loss:
            best_loss = loss
            best = {k: v.detach().numpy().copy() for k, v in raw.items()}

    if best is None:
        raise RuntimeError('every Whittle restart diverged')
    rate = rate_low + (rate_high - rate_low) / (1 + np.exp(-best['raw_rate']))
    return SpectralMixture(np.exp(best['log_w']), mu.numpy(), np.exp(2 * best['log_sd']),
                           float(np.exp(best['log_ou'])), float(rate), float(np.exp(best['log_nugget'])))


def gp_forecaster(kernel, y_train, n_lags, nugget_scale=1.0, width_scale=1.0, steps=DENSE_STEPS):
    """A stationary GP conditioned on the last `n_lags` regular points is a fixed linear
    filter, and its covariance matrix is Toeplitz, so all leads share one Levinson solve.
    `width_scale` shrinks the fitted line widths: the Welch window broadens every spectral
    line, which otherwise reads as memory decaying faster than it really does."""
    narrowed = replace(kernel, variances=kernel.variances * width_scale)
    r = narrowed.acov(n_lags + int(steps.max()) + 1)
    r[0] += kernel.nugget * (nugget_scale - 1)
    column = r[:n_lags]
    # Cov(z[t+h], z[t+1-j]) = r[h + j - 1], for j = 1..n_lags
    rhs = np.stack([r[h:h + n_lags] for h in steps], axis=1)
    weights = solve_toeplitz(column, rhs).T
    mean = float(np.nanmean(y_train))
    low, high = _train_bounds(y_train, mean)
    return LinearForecaster(weights, np.arange(1, n_lags + 1), mean, low, high)


# ======================================================================================
# Method 3: dynamic harmonic regression
# ======================================================================================
@njit(cache=True)
def _steady_filter(y, T, Z, filt_gain, keep):
    """Steady-state Kalman recursion. Returns the filtered state at each index in `keep`
    (sorted). A NaN observation is a pure prediction step."""
    m = T.shape[0]
    a = np.zeros(m)                  # a[t | t-1]
    out = np.zeros((len(keep), m))
    pointer = 0
    for t in range(len(y)):
        pred = 0.0
        for i in range(m):
            pred += Z[i] * a[i]
        v = 0.0 if np.isnan(y[t]) else y[t] - pred
        filtered = a + filt_gain * v
        while pointer < len(keep) and keep[pointer] == t:
            out[pointer] = filtered
            pointer += 1
        a = T @ filtered
    return out


@dataclass
class DynamicHarmonic:
    """Stochastic trigonometric cycles at fixed frequencies plus an AR(p) remainder, the
    TBATS idea in state-space form. The steady-state gain keeps it a fixed linear system."""
    T: np.ndarray
    Z: np.ndarray
    filt_gain: np.ndarray
    mean: float

    def forecast(self, series, track, anchors, steps=DENSE_STEPS):
        order = np.argsort(anchors)
        y = series.track(track) - self.mean
        y = y[:int(anchors.max()) + 1]
        states = _steady_filter(y, self.T, self.Z, self.filt_gain, anchors[order])
        leads = np.zeros((len(steps), len(self.Z)))
        power = np.eye(len(self.Z))
        current = 0
        for i, h in enumerate(steps):
            power = power @ np.linalg.matrix_power(self.T, int(h) - current)
            current = int(h)
            leads[i] = self.Z @ power
        out = np.zeros((len(anchors), len(steps)))
        out[order] = states @ leads.T
        return out + self.mean


def harmonic_design(n, freqs):
    t = np.arange(n, dtype=np.float64)
    return np.column_stack([g(2 * np.pi * f * t) for f in freqs for g in (np.cos, np.sin)])


def harmonic_remainder(y, freqs):
    """(remainder, variance explained) after a least-squares fit of the cycles at `freqs`."""
    z = y - np.nanmean(y)
    present = ~np.isnan(z)
    design = harmonic_design(len(z), freqs)
    coef, *_ = np.linalg.lstsq(design[present], z[present], rcond=None)
    remainder = z - design @ coef
    return remainder, 1 - np.nanvar(remainder) / np.nanvar(z)


def fit_dynamic_harmonic(y_train, freqs, ar_order, cycle_noise, damping, obs_noise):
    """`cycle_noise` and `obs_noise` are variances relative to the AR innovation variance."""
    remainder, _ = harmonic_remainder(y_train, freqs)
    acov = gappy_acov(remainder, ar_order)
    acov[0] *= 1 + 1e-6
    phi = solve_toeplitz(acov[:ar_order], acov[1:ar_order + 1])
    innovation = float(max(acov[0] - phi @ acov[1:ar_order + 1], 1e-10))

    n_cycles = len(freqs)
    m = 2 * n_cycles + ar_order
    T = np.zeros((m, m))
    Q = np.zeros((m, m))
    Z = np.zeros(m)
    for i, f in enumerate(freqs):
        c, s = np.cos(2 * np.pi * f), np.sin(2 * np.pi * f)
        T[2 * i:2 * i + 2, 2 * i:2 * i + 2] = damping * np.array([[c, s], [-s, c]])
        Q[2 * i, 2 * i] = Q[2 * i + 1, 2 * i + 1] = cycle_noise * innovation
        Z[2 * i] = 1.0
    start = 2 * n_cycles
    T[start, start:] = phi
    T[start + 1:, start:-1] += np.eye(ar_order - 1)
    Q[start, start] = innovation
    Z[start] = 1.0

    R = np.array([[obs_noise * innovation]])
    P = solve_discrete_are(T.T, Z[:, None], Q, R)
    F = float(Z @ P @ Z + R[0, 0])
    filt_gain = P @ Z / F
    return DynamicHarmonic(T, Z, filt_gain, float(np.nanmean(y_train)))


# ======================================================================================
# Method 4: direct ridge autoregression
# ======================================================================================
def ridge_lags(max_lag, spacing):
    """Every lag up to the 16h lookback, then every `spacing`-th lag to `max_lag`."""
    dense = np.arange(1, min(max_lag, C.LOOKBACK) + 1)
    sparse = np.arange(C.LOOKBACK + spacing, max_lag + 1, spacing)
    return np.concatenate([dense, sparse]).astype(np.int64)


@njit(parallel=True, cache=True)
def _gather(z, anchors, lags):
    out = np.zeros((len(anchors), len(lags)))
    for i in prange(len(anchors)):
        for k in range(len(lags)):
            out[i, k] = z[anchors[i] + 1 - lags[k]]
    return out


@dataclass
class RidgeGram:
    """Cross-products for one lag set. All leads and every lambda share one
    eigendecomposition, so a lambda sweep costs almost nothing."""
    lags: np.ndarray
    values: np.ndarray
    vectors: np.ndarray
    xty: np.ndarray
    xtu: np.ndarray
    u_stats: tuple
    n: int


def ridge_gram(series, track, lags, chunk=8192, steps=DENSE_STEPS):
    """Train cross-products. Gaps in the design matrix are filled once (`fill_for_spectrum`);
    a row is dropped when any of its 73 targets is missing or flagged."""
    y_train = series.track(track)[:series.n_train]
    mean = np.nanmean(y_train)
    z = fill_for_spectrum(y_train) - mean
    raw = y_train - mean
    u = series.utide[:series.n_train]
    u_mean = float(u.mean())

    last = series.n_train - int(steps.max()) - 1
    anchors = np.arange(int(lags.max()) - 1, last + 1, C.BASELINE_RIDGE_TRAIN_STRIDE)
    tgt = raw[anchors[:, None] + steps[None, :]]
    bad = np.isnan(tgt) | series.flagged[anchors[:, None] + steps[None, :]]
    anchors = anchors[~bad.any(axis=1)]

    xtx = np.zeros((len(lags), len(lags)))
    xty = np.zeros((len(lags), len(steps)))
    xtu = np.zeros((len(lags), len(steps)))
    uu = np.zeros(len(steps))
    uy = np.zeros(len(steps))
    for start in range(0, len(anchors), chunk):
        batch = anchors[start:start + chunk]
        X = _gather(z, batch, lags)
        positions = batch[:, None] + steps[None, :]
        Y = raw[positions]
        U = u[positions] - u_mean
        xtx += X.T @ X
        xty += X.T @ Y
        xtu += X.T @ U
        uu += np.sum(U ** 2, axis=0)
        uy += np.sum(U * Y, axis=0)
    values, vectors = eigh(xtx)
    return RidgeGram(lags, np.maximum(values, 0.0), vectors, xty, xtu, (uu, uy, u_mean), len(anchors))


def ridge_forecaster(gram, y_train, ridge_lambda, use_utide=False):
    """Closed-form ridge. `ridge_lambda` is relative to the mean eigenvalue. With `use_utide`
    the target-time UTide value enters as one extra regressor per lead, by Frisch-Waugh:
    beta = (u residual'y residual) / (u residual'u residual), against the lag regressors."""
    penalty = ridge_lambda * float(np.mean(gram.values))
    solve = lambda b: gram.vectors @ ((gram.vectors.T @ b) / (gram.values[:, None] + penalty))
    coef_y = solve(gram.xty)
    beta = None
    uu, uy, u_mean = gram.u_stats
    if use_utide:
        coef_u = solve(gram.xtu)
        denom = uu - np.sum(coef_u * gram.xtu, axis=0)
        numer = uy - np.sum(coef_u * gram.xty, axis=0)
        beta = numer / np.maximum(denom, 1e-9)
        coef_y = coef_y - coef_u * beta
    mean = float(np.nanmean(y_train))
    low, high = _train_bounds(y_train, mean)
    return LinearForecaster(coef_y.T, gram.lags, mean, low, high, beta, u_mean if use_utide else 0.0)


# ======================================================================================
# Unconditional references
# ======================================================================================
def persistence(series, track, anchors, steps=DENSE_STEPS):
    return np.repeat(series.track(track)[anchors][:, None], len(steps), axis=1)


def seasonal_naive(series, track, anchors, steps=DENSE_STEPS, season=C.BASELINE_SEASONS[0]):
    """The value a whole number of seasons before each target, using only data up to the
    anchor. Where that value is missing, the anchor's own value stands in."""
    y = series.track(track)
    back = np.ceil(steps / season).astype(np.int64) * season
    positions = anchors[:, None] + steps[None, :] - back[None, :]
    out = y[positions]
    fallback = np.repeat(y[anchors][:, None], len(steps), axis=1)
    return np.where(np.isnan(out), fallback, out)


# ======================================================================================
# One way to build a model from a config, shared by the search and the final evaluation
# ======================================================================================
SARIMA_MIN_LAGS = 8640


class ModelCache:
    """Expensive pieces reused between configs: kernels, ridge cross-products, remainders."""
    def __init__(self):
        self.kernels, self.grams = {}, {}

    def kernel(self, y_train, track, q, centres):
        if (track, q, centres) not in self.kernels:
            self.kernels[(track, q, centres)] = fit_spectral_mixture(y_train, q, centres)
        return self.kernels[(track, q, centres)]

    def gram(self, series, track, max_lag, spacing):
        key = (track, max_lag, spacing)
        if key not in self.grams:
            self.grams[key] = ridge_gram(series, track, ridge_lags(max_lag, spacing))
        return self.grams[key]


def build_model(method, cfg, series, track, frequencies, cache):
    """A fitted forecaster from a config dict. `frequencies` are the identified spectral
    peaks, largest first, for the dynamic harmonic model."""
    y_train = series.track(track)[:series.n_train]
    if method == 'ar':
        phi = fit_ar(y_train, cfg['order'], cfg['estimator'], cfg.get('jitter', 1e-5))
        return linear_from_pi(phi, y_train)
    if method == 'sarima':
        order = SarimaOrder(**{k: cfg[k] for k in ('p', 'd', 'q', 'P', 'D', 'Q', 'P2', 'D2')},
                            season=tuple(cfg['season']))
        params = fit_sarima_css(y_train, order)[0]
        degree = max(max(sarima_polys(order, params)[0]), max(sarima_polys(order, params)[1]))
        return linear_from_pi(sarima_pi(order, params, max(SARIMA_MIN_LAGS, degree + 1)), y_train)
    if method == 'gp':
        return gp_forecaster(cache.kernel(y_train, track, cfg['Q'], cfg.get('centres', 'record')), y_train, cfg['history'],
                             cfg['nugget_scale'], cfg['width_scale'])
    if method == 'dhr':
        return fit_dynamic_harmonic(y_train, np.asarray(frequencies[:cfg['K']]), cfg['ar_order'],
                                    cfg['cycle_noise'], 1 - cfg['damping_gap'], cfg['obs_noise'])
    if method == 'ridge':
        gram = cache.gram(series, track, cfg['history'], cfg['spacing'])
        return ridge_forecaster(gram, y_train, cfg['ridge_lambda'], cfg.get('use_utide', False))
    raise ValueError(f'unknown method {method!r}')
