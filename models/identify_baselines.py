"""Model identification for the classical baselines: the evidence behind each search range.

Runs on the training split only, for the water level and the UTide residual:

    SARIMA / AR     ADF and KPSS tests; ACF and PACF of the raw and differenced series;
                    AIC and BIC against AR order
    Gaussian process  spectrum, spectral-mixture fits against the empirical ACF
    Dynamic harmonic  ranked spectral peaks, variance explained against K, drift of the
                    largest cycles' amplitude and phase
    Direct ridge    correlation of each lag with the target at four leads, eigenvalues of X'X

    python models/identify_baselines.py
    # or: python models/run_gauge.py --pipeline baselines --gauge southend --gauge richmond

Outputs: outputs[/<gauge>]/baselines/identification/ (CSVs, identified_ranges.json read by
search_baselines.py, identification_summary.md stating each decision), and 300-dpi PNGs in
report_images/.

ACF/PACF of a tidal series never show the clean cut-offs of textbook ARMA: strong cycles at
non-integer periods give slowly decaying oscillation. So this stage sets candidate ranges and
validation forecast error makes the final choice.
"""
# %% Bootstrap
import pathlib
import sys

_start = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / '_')).resolve()
REPO_ROOT = next(p for p in _start.parents if (p / 'utils' / 'paths.py').exists())
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
if not hasattr(sys, 'ps1') and 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')

# %% Imports
import json
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from utils import paths
from utils.models import baselines as B
from utils.models import config as C
from utils.plot_config import ANNOTATION_INK, figure_suffix, save_fig, set_name_suffix, set_report_dir

# %% ------------------------------------------------------------------ CONFIG
GAUGE = C.from_env('TIDAL_GAUGE', 'southend')    # run_gauge.py sets it
SMOKE = C.from_env('TIDAL_SMOKE', '0').strip().lower() not in ('', '0', 'false')

MAX_LAG = 2000 if SMOKE else 20000           # ACF, PACF and AR order (20000 lags ~ 139 days)
SHORT_LAG = 432                              # the 3-day panels
SEASON, LONG_SEASON = C.BASELINE_SEASONS
PEAK_TABLE = 128
K_GRID = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
GP_COMPONENTS = [4, 8] if SMOKE else [4, 8, 16, 32]
ROLLING_DAYS = 30
RIDGE_LEADS_H = [1, 6, 24, 72]
TEST_DECIMATION = 6                          # ADF/KPSS on the hourly series
TEST_LAGS = 72

# period (h) -> name; labels the peak table only, nothing is fitted from these
CONSTITUENTS = {'M2': 12.4206, 'S2': 12.0, 'N2': 12.6583, 'K2': 11.9672, 'K1': 23.9345, 'O1': 25.8193,
                'P1': 24.0659, 'Q1': 26.8684, 'M4': 6.2103, 'MS4': 6.1033, 'MN4': 6.2692, 'M6': 4.1402,
                'MK3': 8.1771, 'MU2': 12.8718, 'NU2': 12.6260, 'L2': 12.1916, 'M3': 8.2804}

FAMILY = {'level': '#2a78d6', 'residual': '#eb6834', 'fit': '#1baf7a'}
INK = '#0b0b0b'

OUT = paths.ensure_dir(str(paths.baseline_output_dir(GAUGE)) + ('_smoke' if SMOKE else '')) / 'identification'
OUT.mkdir(exist_ok=True)
set_report_dir(str(OUT if SMOKE else paths.ensure_dir(paths.REPORT_IMAGES_DIR)))
set_name_suffix(figure_suffix(GAUGE))
LABEL = C.gauge_info(GAUGE)['label']

series = B.load_series(GAUGE, upto=C.TRAIN_END)
assert series.time.max() < np.datetime64(C.TRAIN_END), 'identification must not see validation or test data'
print(f"{LABEL}: {len(series.time):,} train rows to {series.time.max()}")


# %% Helpers ------------------------------------------------------------------
def difference(y, lag):
    out = np.full(len(y), np.nan)
    out[lag:] = y[lag:] - y[:-lag]
    return out


TRANSFORMS = {
    'raw': lambda y: y,
    'd1': lambda y: difference(y, 1),
    f's{SEASON}': lambda y: difference(y, SEASON),
    f'd1_s{SEASON}': lambda y: difference(difference(y, SEASON), 1),
}


def unit_root_tests(y):
    """ADF (null: unit root) and KPSS (null: stationary) on the gap-filled hourly series."""
    hourly = B.fill_for_spectrum(y[np.argmax(~np.isnan(y)):])[::TEST_DECIMATION]
    adf = adfuller(hourly, maxlag=TEST_LAGS, autolag=None)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')    # KPSS warns when p is outside its lookup table
        kp = kpss(hourly, regression='c', nlags=TEST_LAGS)
    return {'adf_stat': adf[0], 'adf_p': adf[1], 'kpss_stat': kp[0], 'kpss_p': kp[1]}


def constituent_name(period_h):
    name = min(CONSTITUENTS, key=lambda k: abs(CONSTITUENTS[k] - period_h))
    return name if abs(CONSTITUENTS[name] - period_h) < 0.004 * period_h else ''


def variance_explained(y, freqs, grid, chunk=100_000):
    """Share of variance explained by the first K cycles, K in `grid`, from one set of
    cross-products accumulated in chunks (the full design would not fit in memory)."""
    z = y - np.nanmean(y)
    n_col = 2 * len(freqs)
    xtx, xty = np.zeros((n_col, n_col)), np.zeros(n_col)
    t_all = np.arange(len(z), dtype=np.float64)
    angles = 2 * np.pi * freqs
    for start in range(0, len(z), chunk):
        t, zc = t_all[start:start + chunk], z[start:start + chunk]
        present = ~np.isnan(zc)
        phase = np.outer(t[present], angles)
        X = np.empty((present.sum(), n_col))
        X[:, ::2], X[:, 1::2] = np.cos(phase), np.sin(phase)
        xtx += X.T @ X
        xty += X.T @ zc[present]
    total = float(np.nansum(z ** 2))
    rows = []
    for k in grid:
        if k > len(freqs):
            continue
        idx = np.arange(2 * k)
        coef = np.linalg.solve(xtx[np.ix_(idx, idx)] + 1e-6 * np.eye(2 * k), xty[idx])
        rows.append({'K': k, 'variance_explained': float(coef @ xty[idx]) / total})
    return pd.DataFrame(rows)


def rolling_harmonics(y, freqs, days=ROLLING_DAYS):
    """Amplitude and phase of each cycle in consecutive blocks, fitted jointly."""
    z = y - np.nanmean(y)
    block = days * C.STEPS_PER_DAY
    rows = []
    for start in range(0, len(z) - block + 1, block):
        zc = z[start:start + block]
        present = ~np.isnan(zc)
        if present.mean() < 0.8:
            continue
        t = np.arange(start, start + block, dtype=np.float64)
        design = np.column_stack([g(2 * np.pi * f * t) for f in freqs for g in (np.cos, np.sin)])
        coef, *_ = np.linalg.lstsq(design[present], zc[present], rcond=None)
        for i, f in enumerate(freqs):
            rows.append({'block_start': series.time[start], 'period_h': 1 / (6 * f),
                         'amplitude_m': float(np.hypot(coef[2 * i], coef[2 * i + 1])),
                         'phase_rad': float(np.arctan2(coef[2 * i + 1], coef[2 * i]))})
    return pd.DataFrame(rows)


# %% Run ----------------------------------------------------------------------
tests, ranges, decisions = [], {}, []
acf_store, peak_store, k_store, kernel_store, kernel_fit = {}, {}, {}, {}, {}

for track in C.BASELINE_TRACKS:
    t0 = time.time()
    y = series.track(track)[:series.n_train]
    print(f"\n=== {track} ===")

    # --- SARIMA / AR: differencing, ACF, PACF ---
    acf_rows = {}
    for name, transform in TRANSFORMS.items():
        yt = transform(y)
        cov = B.gappy_acov(yt, MAX_LAG)
        cov[0] *= 1 + 1e-9
        pacf, sigma2 = B.durbin_levinson(cov, MAX_LAG)
        n = int(np.sum(~np.isnan(yt)))
        pd.DataFrame({'lag': np.arange(1, MAX_LAG + 1), 'acf': cov[1:] / cov[0], 'pacf': pacf}).to_csv(
            OUT / f'acf_pacf_{track}_{name}.csv', index=False)
        acf_rows[name] = (cov / cov[0], pacf, n)
        row = {'track': track, 'series': name, 'n': n, 'std': float(np.nanstd(yt))}
        row.update(unit_root_tests(yt))
        tests.append(row)
        print(f"{name:>10}: std {row['std']:.3f}  ADF p {row['adf_p']:.3g}  KPSS p {row['kpss_p']:.3g}")
    acf_store[track] = acf_rows

    # --- AR order ---
    acf_raw, pacf_raw, n_raw = acf_rows['raw']
    cov_raw = B.gappy_acov(y, MAX_LAG)
    cov_raw[0] *= 1 + 1e-9
    _, sigma2 = B.durbin_levinson(cov_raw, MAX_LAG)
    orders = np.arange(1, MAX_LAG + 1)
    log_s = np.log(np.maximum(sigma2[1:], 1e-300))
    order_table = pd.DataFrame({'p': orders, 'sigma2': sigma2[1:], 'aic': n_raw * log_s + 2 * orders,
                                'bic': n_raw * log_s + np.log(n_raw) * orders})
    order_table.iloc[np.unique(np.r_[0:500, 499:MAX_LAG:10])].to_csv(OUT / f'ar_order_{track}.csv', index=False)
    best_aic, best_bic = int(order_table.p[order_table.aic.idxmin()]), int(order_table.p[order_table.bic.idxmin()])
    band = 1.96 / np.sqrt(n_raw)
    above = np.flatnonzero(np.abs(pacf_raw[:SHORT_LAG]) > band) + 1
    last_short_pacf = int(above.max()) if len(above) else 0
    near = lambda c, width: int(c - width + np.argmax(acf_raw[c - width:c + width + 1]))
    s_peak = near(SEASON, 10) if MAX_LAG > SEASON + 10 else None
    long_peak = near(LONG_SEASON, 120) if MAX_LAG > LONG_SEASON + 120 else None
    print(f"AIC order {best_aic}, BIC order {best_bic}; ACF peaks at lags {s_peak}, {long_peak} "
          f"(r = {acf_raw[s_peak]:.3f}" + (f", {acf_raw[long_peak]:.3f})" if long_peak else ")"))

    # --- spectra, for the GP and dynamic harmonic methods ---
    freq, power = B.periodogram(y)
    peaks = B.spectral_peaks(freq, power, PEAK_TABLE)
    peaks.insert(0, 'rank', np.arange(1, len(peaks) + 1))
    peaks['nearest_constituent'] = [constituent_name(p) for p in peaks['period_h']]
    peaks.to_csv(OUT / f'spectral_peaks_{track}.csv', index=False)
    bins = np.unique(np.geomspace(1, len(freq) - 1, 6000).astype(int))
    pd.DataFrame({'freq_cycles_per_step': freq[bins], 'power': power[bins]}).to_csv(
        OUT / f'periodogram_{track}.csv', index=False)
    peak_store[track] = (freq[bins], power[bins], peaks)

    freqs = peaks['freq_cycles_per_step'].to_numpy()
    k_table = variance_explained(y, freqs, K_GRID)
    k_table.to_csv(OUT / f'harmonic_variance_vs_k_{track}.csv', index=False)
    k_store[track] = k_table
    drift = rolling_harmonics(y, freqs[:6])
    drift.to_csv(OUT / f'rolling_harmonics_{track}.csv', index=False)
    amp_cv = float((drift.groupby('period_h').amplitude_m.std() / drift.groupby('period_h').amplitude_m.mean()).median())
    print(f"variance explained by K: {dict(zip(k_table.K, k_table.variance_explained.round(4)))}; "
          f"median amplitude CV {amp_cv:.3f}")

    # --- GP: kernels against the empirical ACF ---
    kernel_rows = []
    lags = np.unique(np.r_[0:SHORT_LAG, np.arange(SHORT_LAG, MAX_LAG, 12)])
    kernel_frame = pd.DataFrame({'lag': lags, 'empirical_acf': acf_raw[lags]})
    for q in GP_COMPONENTS:
        for centres in ('record', 'welch'):
            kernel = B.fit_spectral_mixture(y, q, centres)
            model_cov = kernel.acov(MAX_LAG + 1)
            model_acf = model_cov / model_cov[0]
            kernel_frame[f'kernel_q{q}_{centres}'] = model_acf[lags]
            kernel_rows.append({'Q': q, 'centres': centres, 'acf_rmse': float(np.sqrt(np.mean((model_acf - acf_raw) ** 2))),
                                'ou_share': kernel.ou_var / model_cov[0], 'nugget_share': kernel.nugget / model_cov[0]})
            print(f"GP Q={q} {centres}: ACF RMSE {kernel_rows[-1]['acf_rmse']:.4f}, OU share {kernel_rows[-1]['ou_share']:.3f}")
    kernel_frame.to_csv(OUT / f'gp_kernel_vs_acf_{track}.csv', index=False)
    gp_table = pd.DataFrame(kernel_rows)
    gp_table.to_csv(OUT / f'gp_kernel_fit_{track}.csv', index=False)
    kernel_store[track], kernel_fit[track] = kernel_frame, gp_table
    acf_noise_floor = int(np.flatnonzero(np.abs(acf_raw) > band).max())

    # --- ridge: how much each lag says about the target, and the conditioning of X'X ---
    lead_rows = {'lag': np.arange(1, MAX_LAG - 432)}
    for h in RIDGE_LEADS_H:
        lead_rows[f'corr_lead_{h}h'] = acf_raw[h * 6 + lead_rows['lag'] - 1]
    pd.DataFrame(lead_rows).iloc[::6].to_csv(OUT / f'ridge_lag_correlation_{track}.csv', index=False)
    eigen = {}
    for name, lags_r in [('L96', B.ridge_lags(96, 1)), ('L4320_s6', B.ridge_lags(min(4320, MAX_LAG), 6))]:
        gram = B.ridge_gram(series, track, lags_r)
        eigen[name] = gram.values[::-1] / gram.values.mean()
        pd.DataFrame({'rank': np.arange(1, len(eigen[name]) + 1), 'eigenvalue_over_mean': eigen[name]}).to_csv(
            OUT / f'ridge_eigenvalues_{track}_{name}.csv', index=False)
    condition = float(eigen['L4320_s6'][0] / max(eigen['L4320_s6'][-1], 1e-300))

    # --- decisions, read by search_baselines.py ---
    test_row = {r['series']: r for r in tests if r['track'] == track}
    stationary = lambda r: r['adf_p'] < 0.05 and r['kpss_p'] > 0.05
    grid_top = max(C.BASELINE_HISTORY_GRID)
    extend_ar = best_bic >= grid_top and MAX_LAG >= best_bic
    ar_grid = sorted(set(C.BASELINE_HISTORY_GRID) | ({min(best_bic, MAX_LAG)} if extend_ar else set()))
    ranges[track] = {
        'd': [0, 1], 'D': [0, 1], 'D2': [0, 1],
        'p_max': int(np.clip(2 * last_short_pacf, 8, 96)), 'q_max': 16, 'P_max': 5, 'Q_max': 4, 'P2_max': 4,
        'season': [int(s_peak), int(long_peak or LONG_SEASON)],
        'ar_orders': ar_grid, 'gp_components': [4, 8, 16, 32, 48, 64, 96, 128], 'gp_history': list(C.BASELINE_HISTORY_GRID),
        'dhr_K': K_GRID, 'dhr_frequencies': freqs.tolist(),
        'ridge_history': list(C.BASELINE_HISTORY_GRID) + [C.BASELINE_RIDGE_LONG_HISTORY], 'ridge_spacing': [3, 6, 12],
    }
    decisions.append(f"""
## {track}

**Differencing.** Raw series: ADF p = {test_row['raw']['adf_p']:.3g}, KPSS p = {test_row['raw']['kpss_p']:.3g}, \
so it {'passes both' if stationary(test_row['raw']) else 'does not pass both'} stationarity checks on the hourly series. \
The tests detect unit roots, not periodicity, so seasonal differencing is judged by whether it removes the ACF \
oscillation (`acf_pacf_{track}_*.csv`) and stays a candidate: d, D and D2 are each searched over 0 and 1.

**Seasonal periods.** ACF peaks at lag {s_peak} (r = {acf_raw[s_peak]:.3f}) near the assumed {SEASON}, and at lag \
{long_peak} near the assumed {LONG_SEASON}{f" (r = {acf_raw[long_peak]:.3f})" if long_peak else ""}. The search uses \
these measured lags.

**Non-seasonal orders.** The PACF stays outside +/-{band:.5f} up to lag {last_short_pacf} within the first 3 days. \
With no clean cut-off, p is searched up to {ranges[track]['p_max']} and q up to {ranges[track]['q_max']}.

**AR order.** AIC is lowest at p = {best_aic} and BIC at p = {best_bic}, of orders up to {MAX_LAG}. \
Search grid: {ar_grid}.{' BIC minimum is at the edge of the range examined, so long orders are not ruled out.' if best_bic >= MAX_LAG else ''}

**Gaussian process.** The ACF stays outside the noise band to lag {acf_noise_floor}. Spectral-mixture ACF RMSE against \
the empirical ACF: {', '.join(f"Q={r.Q:.0f} {r.centres}: {r.acf_rmse:.4f}" for r in gp_table.itertuples())}. OU share at the \
largest Q: {gp_table.ou_share.iloc[-1]:.3f}, so the broadband term is {'needed' if gp_table.ou_share.iloc[-1] > 0.005 else 'small'}.

**Dynamic harmonic.** Variance explained: {', '.join(f"K={r.K:.0f}: {r.variance_explained:.4f}" for r in k_table.itertuples())}. \
K grid: {ranges[track]['dhr_K']}. The six largest cycles have a median amplitude coefficient of variation of \
{amp_cv:.3f} across {ROLLING_DAYS}-day blocks, which is why their amplitudes are stochastic rather than fixed.

**Direct ridge.** Target correlation by lag is in `ridge_lag_correlation_{track}.csv`. X'X with lags to 4320 has a \
condition number of {condition:.2e}, so regularisation is required and lambda spans 1e-8 to 1e0 of the mean eigenvalue.
""")
    print(f"track done in {(time.time() - t0) / 60:.1f} min")

# %% Tables -------------------------------------------------------------------
pd.DataFrame(tests).to_csv(OUT / 'stationarity_tests.csv', index=False)
(OUT / 'identified_ranges.json').write_text(json.dumps(ranges, indent=2), encoding='utf-8')
(OUT / 'identification_summary.md').write_text(
    f"# Identification summary: {LABEL}\n\nTrain split only ({series.time.min()} to {series.time.max()}). "
    "These decisions set candidate ranges; validation forecast error chooses the winner.\n"
    + ''.join(decisions), encoding='utf-8')
print(f"saved tables and identification_summary.md to {OUT}")

# %% Figures ------------------------------------------------------------------
for track in C.BASELINE_TRACKS:
    names = list(TRANSFORMS)[:3]
    fig, axes = plt.subplots(len(names), 2, sharex=True, figsize=(6.3, 5.4))
    x = np.arange(1, SHORT_LAG + 1) / 6
    for row, name in zip(axes, names):
        acf_v, pacf_v, n = acf_store[track][name]
        row[0].plot(x, acf_v[1:SHORT_LAG + 1], color=FAMILY[track], lw=1.1)
        row[1].plot(x, pacf_v[:SHORT_LAG], color=FAMILY[track], lw=0.9)
        for ax in row:
            ax.axhline(0, color=ANNOTATION_INK, lw=0.6)
            ax.axvline(SEASON / 6, color=ANNOTATION_INK, lw=0.6, ls=':')
        row[0].set_ylabel(f'ACF: {name}', fontsize=9)
        row[1].set_ylabel(f'PACF: {name}', fontsize=9)
    axes[-1, 0].set_xlabel('Lag (h)')
    axes[-1, 1].set_xlabel('Lag (h)')
    save_fig(fig, f'acf_pacf_{track}', width_in=6.3, height_in=5.4)
    plt.close(fig)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6.3, 4.6))
for ax, track in zip(axes, C.BASELINE_TRACKS):
    f, p, peaks = peak_store[track]
    period = 1 / (6 * f[1:])
    ax.loglog(period, p[1:], color=FAMILY[track], lw=0.7)
    ax.set_ylabel(f'Power: {track}', fontsize=9)
    lo, hi = np.percentile(p[1:], [1, 100])
    last = 0.0
    for row in peaks[peaks.nearest_constituent != ''].drop_duplicates('nearest_constituent').sort_values('period_h').itertuples():
        if row.period_h > 1.35 * last:     # only as many names as fit without overlapping
            ax.annotate(row.nearest_constituent, (row.period_h, hi), fontsize=7, ha='center', va='bottom',
                        color=ANNOTATION_INK, annotation_clip=False)
            last = row.period_h
axes[-1].set_xlabel('Period (h)')
save_fig(fig, 'periodogram_peaks', width_in=6.3, height_in=4.6)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.3, 3.4))
for track in C.BASELINE_TRACKS:
    t = k_store[track]
    ax.plot(t.K, t.variance_explained, color=FAMILY[track], marker='o', ms=5, lw=1.8, label=track)
ax.set_xlabel('Number of cycles K')
ax.set_ylabel('Variance explained')
ax.legend(frameon=False)
save_fig(fig, 'harmonic_variance_vs_k', width_in=6.3, height_in=3.4)
plt.close(fig)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6.3, 4.6))
for ax, track in zip(axes, C.BASELINE_TRACKS):
    frame = kernel_store[track]
    best_fit = kernel_fit[track].sort_values('acf_rmse').iloc[0]
    best_q = int(best_fit.Q)
    q_col = f'kernel_q{best_q}_{best_fit.centres}'
    keep = frame.lag <= 8640
    ax.plot(frame.lag[keep] / 144, frame.empirical_acf[keep], color=INK, lw=0.6, label='Empirical ACF')
    ax.plot(frame.lag[keep] / 144, frame[q_col][keep], color=FAMILY['fit'], lw=0.6, alpha=0.85,
            label=f'Fitted kernel, Q={best_q}')
    ax.set_ylabel(f'ACF: {track}', fontsize=9)
    ax.legend(frameon=False, ncol=2, loc='lower right', bbox_to_anchor=(1, 1), fontsize=8)
axes[-1].set_xlabel('Lag (days)')
save_fig(fig, 'gp_kernel_vs_empirical_acf', width_in=6.3, height_in=4.6)
plt.close(fig)

print("\nDone.")
