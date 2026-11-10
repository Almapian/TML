# Tidal Thames forecasting: progress since the dissertation (November 2026)

## What changed since the dissertation

- **Short-lead fix.** The dissertation's 73-lead neural model (10 min, then hourly to 72h, on top of the harmonic prediction) was poor at short leads: at 10 min it was worse than simply carrying the current surge forward ("persistence"). The cause was that every lead counted equally in training, so the 10-min lead was under 0.1% of what the model optimised. The new model starts from persistence and learns a correction, and each lead's error is weighted by how hard that lead is for persistence. Five versions were compared on validation (2018 to 2020) under a rule fixed in advance; this one won at both gauges and is the only one scored on test (2021 to 2024).
- **A data leak was found and fixed.** NRFA daily river flow and rainfall cover 09:00 to 09:00 and are dated by their start, so they are only known 33h after midnight on that date. The dissertation's stage 3 (weather and river) models used them up to 33h early. This flattered the stage 3 results at both gauges; it does not affect any result below, which uses no river data.
- **Continuous output and high-water scoring.** The hourly forecast is interpolated to every 10 min (cubic spline through the forecast correction to the harmonic prediction), and high and low waters are scored for height and timing.

## Headline results (test, 2021 to 2024; RMSE in cm, averaged over all forecasts)

**Southend Pier**

| method | 10 min | 1h | 6h | 24h | 48h | 72h |
|---|---|---|---|---|---|---|
| Harmonic analysis alone | 24.5 | 24.5 | 24.4 | 24.4 | 24.4 | 24.5 |
| Persistence of current surge | 2.0 | 10.3 | 27.1 | 30.8 | 32.5 | 34.6 |
| Best classical (autoregressive surge model) | **1.2** | **5.1** | **17.7** | **21.7** | **22.1** | **22.5** |
| Neural, dissertation | 4.9 | 6.5 | 18.2 | 21.8 | 22.3 | 23.8 |
| Neural, new | 1.3 | 5.5 | 18.3 | 22.2 | 22.5 | 24.1 |

**Richmond Lock**

| method | 10 min | 1h | 6h | 24h | 48h | 72h |
|---|---|---|---|---|---|---|
| Harmonic analysis alone | 56.0 | 56.0 | 56.0 | 56.0 | 56.0 | 56.0 |
| Persistence of current surge | 5.5 | 26.0 | 54.2 | 37.7 | 49.0 | 54.0 |
| Best classical (autoregressive surge model) | 3.3 | 14.3 | 24.9 | 27.8 | 32.1 | 35.2 |
| Neural, dissertation | 9.2 | 14.7 | **23.0** | **25.6** | 29.8 | **33.2** |
| Neural, new | **3.1** | **12.9** | 24.2 | 26.0 | **29.8** | 33.5 |

High water, new neural model with interpolation against harmonic analysis alone (test): Southend height error 19.9 cm RMSE (harmonic 25.1 cm), timing 5.6 min mean absolute error (harmonic 7.8 min); Richmond 29.0 cm (43.5 cm) and 9.8 min (15.0 min).

## What the classical baselines show, and how the comparison was made fair

The classical models (autoregressive, ridge regression, Gaussian process, SARIMA, dynamic harmonic regression) were each tuned on validation with the same effort, then frozen and scored once on test. All methods share the same forecast start times, leads, quality mask (flagged, stuck and patched values excluded), harmonic fit (2004 to 2017 only) and error code. Differences are tested with a 7-day block bootstrap, which respects the autocorrelation of tidal errors, averaged over all 73 leads and over 24h to 72h.

- **Richmond:** the neural model is better than the best classical model by 1.8 cm on average (2.2 cm over 24h to 72h), and the difference is statistically clear.
- **Southend:** the autoregressive model is better than the neural model by 0.7 cm, also statistically clear. At the estuary mouth the surge behaves close to linearly, which a linear model captures fully.

## Open problems

1. At Southend the neural model does not beat a well-tuned linear model.
2. The short-lead fix costs a few millimetres at 24h to 72h.
3. Every method under-predicts high waters, and the harmonic prediction does so most at Southend (by 14 cm on test against 8 cm on validation). Mean level at Southend has risen 5 to 7 cm above the 2004 to 2017 fit by 2022 to 2024. Giving the neural model the last 30 days of mean level did not help on validation.
4. Richmond low waters are held up by a control near -1.0 m ODN, so they are not scored.
5. The stage 3 weather and river model has not been retrained since the leak fix. At Richmond its saved version is effectively untrained: it stopped improving after the first pass through the data.

## Decisions I need from you

1. **Horizon:** 48h (the Port of London's need, and the model is weakest at its last lead) or 72h?
2. **What "continuous" should mean:** hourly forecasts interpolated to 10 min (current), a model that outputs every 10 min directly, or a forecast that can be issued and updated at any time?
3. **Lead story:** Richmond, where the neural model clearly wins, with Southend as the linear control case? Or both treated equally?
4. **Stage 3:** retrain it once, leak-free and tuned for Richmond, or move it to future work?
5. **Other gauges as inputs** (for example upstream or downstream Thames gauges): in scope for this paper, or later?

Sources: `reports/2026-11-nov-fixes/02_overnight_results.md`, `01_diagnostics.md`, and `CHANGELOG.md`.
