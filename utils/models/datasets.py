"""Loading, splitting, UTide fitting, windowing and scaling -> ready-to-train tensors.

Both the train and plot script for a stage call the same `prepare_stage*` function, so a
figure can never be drawn from a slightly different pipeline than the one that produced
the weights.

Everything here reads from `Data/` (the tracked copy of the data the pipeline consumes).
The chatter/stuck patching and the Chart-Datum -> ODN correction have already been applied
to `Data/southend_pier_data.csv`; see EDA/utide_test.ipynb for how.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import utide

from .. import paths
from .config import (DAILY_GRID_BUFFER_DAYS, GAUGE_USECOLS, LOOKBACK, METEO_LOOKBACK_HOURS,
                     RIVER_LOOKBACK_DAYS, SITE_LAT, STEP, TRAIN_END, TRAIN_STRIDE, UTIDE_EXT_DAYS,
                     VAL_END)
from .windowing import MeteoGrids, build_windows, build_windows_multi, build_windows_stage3


# ======================================================================================
# Shared loading / splitting / UTide
# ======================================================================================
def load_gauge_df(path=None, verbose=True):
    """Southend Pier 10-minute record with a single combined `is_flagged` column."""
    path = paths.SOUTHEND_CSV if path is None else path
    df = pd.read_csv(path, usecols=GAUGE_USECOLS, parse_dates=['DateTime'])
    df = df.sort_values('DateTime').reset_index(drop=True)
    df['is_flagged'] = df['is_chatter_flagged'] | df['is_stuck_flagged'] | df['is_imputed']

    if verbose:
        gap_mask = df['DateTime'].diff() > STEP
        print(f"{len(df):,} rows, {df.DateTime.min()} to {df.DateTime.max()}")
        print(f"{gap_mask.sum()} gaps > 10min ({gap_mask.sum() + 1} contiguous segments)")
        print(f"Flagged (chatter/stuck/imputed) points: {df['is_flagged'].sum():,} "
              f"({100 * df['is_flagged'].mean():.3f}%)")
    return df


def split_chronological(df, train_end=TRAIN_END, val_end=VAL_END, verbose=True):
    """Three non-overlapping blocks, split before windowing so no window crosses a boundary."""
    train_df = df[df['DateTime'] < train_end].reset_index(drop=True)
    val_df = df[(df['DateTime'] >= train_end) & (df['DateTime'] < val_end)].reset_index(drop=True)
    test_df = df[df['DateTime'] >= val_end].reset_index(drop=True)

    if verbose:
        for split_name, split in [('train', train_df), ('val', val_df), ('test', test_df)]:
            print(f"{split_name:5s}: {len(split):>8,} rows  "
                  f"[{split.DateTime.min()} -> {split.DateTime.max()}]  "
                  f"flagged={100 * split['is_flagged'].mean():.3f}%")
    return train_df, val_df, test_df


def fit_utide(df, start=None, end=TRAIN_END, lat=SITE_LAT, verbose=True):
    """Solve for harmonic constituents over [start, end), excluding flagged points so the
    fit never learns from UTide's own past reconstruction."""
    mask = df['DateTime'] < end
    if start is not None:
        mask &= df['DateTime'] >= start

    fit_series = df.loc[mask, 'Observed_ODN'].copy()
    fit_series[df.loc[mask, 'is_flagged']] = np.nan
    coef = utide.solve(df.loc[mask, 'DateTime'], fit_series, lat=lat, method='ols', conf_int='none')

    if verbose:
        print(f"UTide fit on {mask.sum():,} rows "
              f"({(~np.isnan(fit_series)).sum():,} after excluding flagged points).")
    return coef


def reconstruct_utide(index, coef, min_snr=0, min_pe=0):
    """Reconstruct the harmonic tide at `index`.

    Stages 2 and 3 keep every solved constituent (min_SNR=0, min_PE=0) because the
    reconstruction is a model *input* there, and is also the standalone comparator -- so
    the feature the model sees and the benchmark it is judged against can never diverge
    through constituent selection. Stage 1 uses UTide only as a baseline and kept the
    library default (min_SNR=2), so it passes that explicitly.
    """
    recon = utide.reconstruct(pd.DatetimeIndex(index), coef, min_SNR=min_snr, min_PE=min_pe, verbose=False)
    return np.asarray(recon['h'], dtype=np.float32)


def to_tensor(a, device, dtype=torch.float32):
    return torch.tensor(a, dtype=dtype, device=device)


# ======================================================================================
# Stage 1
# ======================================================================================
@dataclass
class Stage1Data:
    df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: np.ndarray
    y_train: np.ndarray
    flag_train: np.ndarray
    time_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    flag_val: np.ndarray
    time_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    flag_test: np.ndarray
    time_test: np.ndarray
    train_mean: float
    train_std: float
    train_tensors: dict
    val_tensors: dict
    test_tensors: dict


def prepare_stage1(device, lookback=LOOKBACK, train_stride=TRAIN_STRIDE, data_path=None, verbose=True):
    """Stage 1: single-step windows, train-only scaler, tensors on `device`."""
    df = load_gauge_df(data_path, verbose=verbose)
    train_df, val_df, test_df = split_chronological(df, verbose=verbose)

    X_train, y_train, flag_train, time_train = build_windows(train_df, lookback)
    X_val, y_val, flag_val, time_val = build_windows(val_df, lookback)
    X_test, y_test, flag_test, time_test = build_windows(test_df, lookback)

    X_train, y_train, flag_train, time_train = (
        X_train[::train_stride], y_train[::train_stride],
        flag_train[::train_stride], time_train[::train_stride],
    )
    if verbose:
        print(f"windows -> train: {X_train.shape} (stride {train_stride}), "
              f"val: {X_val.shape}, test: {X_test.shape}")

    train_mean = float(train_df['Observed_ODN'].mean())
    train_std = float(train_df['Observed_ODN'].std())
    if verbose:
        print(f"train mean={train_mean:.4f} m, std={train_std:.4f} m")

    def sc(a):
        return (a - train_mean) / train_std

    train_tensors = {'X': to_tensor(sc(X_train), device), 'Y': to_tensor(sc(y_train), device),
                     'Flag': to_tensor(flag_train, device, dtype=torch.bool)}
    val_tensors = {'X': to_tensor(sc(X_val), device), 'Y': to_tensor(sc(y_val), device),
                   'Flag': to_tensor(flag_val, device, dtype=torch.bool)}
    test_tensors = {'X': to_tensor(sc(X_test), device)}

    return Stage1Data(df, train_df, val_df, test_df,
                      X_train, y_train, flag_train, time_train,
                      X_val, y_val, flag_val, time_val,
                      X_test, y_test, flag_test, time_test,
                      train_mean, train_std, train_tensors, val_tensors, test_tensors)


# ======================================================================================
# Stage 2
# ======================================================================================
@dataclass
class Stage2Data:
    df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    coef: object
    X_test: np.ndarray
    Ydec_test: np.ndarray
    Y_test: np.ndarray
    Flag_test: np.ndarray
    Time_test: np.ndarray
    train_mean: float
    train_std: float
    train_tensors: dict
    val_tensors: dict
    test_tensors: dict
    train_tensors_flat: dict
    val_tensors_flat: dict
    test_tensors_flat: dict


def prepare_stage2(device, horizon_steps, lookback=LOOKBACK, train_stride=TRAIN_STRIDE,
                   data_path=None, verbose=True):
    """Stage 2: UTide fit on train only, reconstructed across the full span and used both
    as the decoder input feature and as the standalone UTide comparator."""
    df = load_gauge_df(data_path, verbose=verbose)

    coef = fit_utide(df, end=TRAIN_END, verbose=verbose)
    df['utide_recon'] = reconstruct_utide(df['DateTime'], coef)
    if verbose:
        print(f"Reconstructed {len(df):,} rows across the full span.")

    train_df, val_df, test_df = split_chronological(df, verbose=verbose)

    X_train, Ydec_train, Y_train, Flag_train, _ = build_windows_multi(train_df, lookback, horizon_steps)
    X_val, Ydec_val, Y_val, Flag_val, _ = build_windows_multi(val_df, lookback, horizon_steps)
    X_test, Ydec_test, Y_test, Flag_test, Time_test = build_windows_multi(test_df, lookback, horizon_steps)

    X_train, Ydec_train, Y_train, Flag_train = (
        X_train[::train_stride], Ydec_train[::train_stride],
        Y_train[::train_stride], Flag_train[::train_stride],
    )
    if verbose:
        print(f"windows -> train: {X_train.shape} (stride {train_stride}), "
              f"val: {X_val.shape}, test: {X_test.shape}")

    train_mean = float(train_df['Observed_ODN'].mean())
    train_std = float(train_df['Observed_ODN'].std())
    if verbose:
        print(f"train mean={train_mean:.4f} m, std={train_std:.4f} m")

    def sc(a):
        return (a - train_mean) / train_std

    train_tensors = {'X': to_tensor(sc(X_train), device), 'Ydec': to_tensor(sc(Ydec_train), device),
                     'Y': to_tensor(sc(Y_train), device),
                     'Flag': to_tensor(Flag_train, device, dtype=torch.bool)}
    val_tensors = {'X': to_tensor(sc(X_val), device), 'Ydec': to_tensor(sc(Ydec_val), device),
                   'Y': to_tensor(sc(Y_val), device),
                   'Flag': to_tensor(Flag_val, device, dtype=torch.bool)}
    test_tensors = {'X': to_tensor(sc(X_test), device), 'Ydec': to_tensor(sc(Ydec_test), device)}

    # the flat stage-1 comparators take no decoder input
    train_tensors_flat = {k: v for k, v in train_tensors.items() if k != 'Ydec'}
    val_tensors_flat = {k: v for k, v in val_tensors.items() if k != 'Ydec'}
    test_tensors_flat = {'X': test_tensors['X']}

    return Stage2Data(df, train_df, val_df, test_df, coef,
                      X_test, Ydec_test, Y_test, Flag_test, Time_test,
                      train_mean, train_std,
                      train_tensors, val_tensors, test_tensors,
                      train_tensors_flat, val_tensors_flat, test_tensors_flat)


def prepare_stage2_reduced(device, df, train_start, horizon_steps, lookback=LOOKBACK,
                           train_stride=TRAIN_STRIDE, name='reduced'):
    """One scenario of stage 2's training-data-volume ablation: UTide refit on the reduced
    window, and val/test windows rebuilt against that refit (only the observed targets are
    unaffected), with a scaler fit on the reduced training data alone."""
    scen_df = df.copy()
    scen_df['utide_recon'] = reconstruct_utide(
        scen_df['DateTime'], fit_utide(scen_df, start=train_start, end=TRAIN_END, verbose=False))

    tr_df = scen_df[(scen_df['DateTime'] >= train_start) & (scen_df['DateTime'] < TRAIN_END)].reset_index(drop=True)
    va_df = scen_df[(scen_df['DateTime'] >= TRAIN_END) & (scen_df['DateTime'] < VAL_END)].reset_index(drop=True)
    te_df = scen_df[scen_df['DateTime'] >= VAL_END].reset_index(drop=True)

    Xtr, Ydtr, Ytr, Ftr, _ = build_windows_multi(tr_df, lookback, horizon_steps)
    Xva, Ydva, Yva, Fva, _ = build_windows_multi(va_df, lookback, horizon_steps)
    Xte, Ydte, Yte, Fte, _ = build_windows_multi(te_df, lookback, horizon_steps)

    Xtr, Ydtr, Ytr, Ftr = Xtr[::train_stride], Ydtr[::train_stride], Ytr[::train_stride], Ftr[::train_stride]

    mean_ = float(tr_df['Observed_ODN'].mean())
    std_ = float(tr_df['Observed_ODN'].std())

    def sc(a):
        return (a - mean_) / std_

    print(f"[{name}] train {tr_df.DateTime.min().date()} -> {tr_df.DateTime.max().date()} "
          f"({Xtr.shape[0]:,} windows, stride {train_stride})  val {Xva.shape[0]:,}  test {Xte.shape[0]:,}")

    return {
        'name': name, 'mean': mean_, 'std': std_,
        'train_tensors': {'X': to_tensor(sc(Xtr), device), 'Ydec': to_tensor(sc(Ydtr), device),
                          'Y': to_tensor(sc(Ytr), device), 'Flag': to_tensor(Ftr, device, dtype=torch.bool)},
        'val_tensors': {'X': to_tensor(sc(Xva), device), 'Ydec': to_tensor(sc(Ydva), device),
                        'Y': to_tensor(sc(Yva), device), 'Flag': to_tensor(Fva, device, dtype=torch.bool)},
        'test_tensors': {'X': to_tensor(sc(Xte), device), 'Ydec': to_tensor(sc(Ydte), device)},
        'Y_test': Yte, 'Flag_test': Fte, 'Ydec_test': Ydte,
    }


# ======================================================================================
# Stage 3
# ======================================================================================
def load_meteo(meteo_dir=None, verbose=True):
    """The four consolidated meteo/river series, Southend Pier / Kingston catchment."""
    meteo_dir = paths.METEO_DIR if meteo_dir is None else meteo_dir

    if not (meteo_dir / 'pressure_all_gauges.csv').exists():
        raise FileNotFoundError(
            f"No meteo data in {meteo_dir}. Unlike the gauge record, the meteorological "
            f"covariates are not published with this repository (size); regenerate them with "
            f"src/download_meteo_data.py, or copy the four CSVs into {meteo_dir}. "
            f"Stages 1 and 2 do not need them -- only stage 3 does.")

    pressure = pd.read_csv(meteo_dir / 'pressure_all_gauges.csv', parse_dates=['valid_time'])
    pressure = pressure.rename(columns={'Southend Pier_msl': 'msl_pa'})
    wind = pd.read_csv(meteo_dir / 'wind_all_gauges.csv', parse_dates=['valid_time'])
    wind = wind.rename(columns={'Southend Pier_u10': 'u10', 'Southend Pier_v10': 'v10'})
    rainfall = pd.read_csv(meteo_dir / 'rainfall_kingston_catchment.csv', parse_dates=['date'])
    discharge = pd.read_csv(meteo_dir / 'river_discharge_kingston.csv', parse_dates=['date'])

    if verbose:
        print(f"pressure: {len(pressure):,} rows, {pressure.valid_time.min()} to {pressure.valid_time.max()}")
        print(f"wind:     {len(wind):,} rows, {wind.valid_time.min()} to {wind.valid_time.max()}")
        print(f"rainfall: {len(rainfall):,} rows, {rainfall.date.min()} to {rainfall.date.max()}")
        print(f"discharge:{len(discharge):,} rows, {discharge.date.min()} to {discharge.date.max()}")
    return pressure, wind, rainfall, discharge


def build_meteo_grids(df, pressure, wind, rainfall, discharge, train_end=TRAIN_END,
                      meteo_lookback_hours=METEO_LOOKBACK_HOURS,
                      river_lookback_days=RIVER_LOOKBACK_DAYS, verbose=True):
    """Hourly pressure/wind and daily rainfall/discharge on complete grids.

    Rainfall stops 2023-12-31 and discharge 2024-09-30 (both inside the test split); the
    missing days are filled with train-split month-of-year climatology and carry an
    availability flag the model sees as an extra channel.
    """
    pressure = pressure.copy()
    pressure['msl_hpa'] = pressure['msl_pa'] / 100.0

    hourly = pressure[['valid_time', 'msl_hpa']].merge(
        wind[['valid_time', 'u10', 'v10']], on='valid_time', how='inner')
    hourly = hourly.sort_values('valid_time').reset_index(drop=True)
    assert len(hourly) == len(pressure) == len(wind), "pressure/wind hourly grids don't match 1:1"
    hourly_diffs = hourly['valid_time'].diff().dropna()
    assert (hourly_diffs == pd.Timedelta('1h')).all(), "hourly grid has gaps -- unexpected per EDA"

    daily_start_ts = df['DateTime'].min().normalize()
    daily_end_ts = df['DateTime'].max().normalize() + pd.Timedelta(days=DAILY_GRID_BUFFER_DAYS)
    full_daily_index = pd.date_range(daily_start_ts, daily_end_ts, freq='D')

    rain_s = rainfall.set_index('date')['value'].reindex(full_daily_index)
    disc_s = discharge.set_index('date')['value'].reindex(full_daily_index)
    rain_avail_s = rain_s.notna().astype(np.float32)
    disc_avail_s = disc_s.notna().astype(np.float32)

    train_date_mask = full_daily_index < pd.Timestamp(train_end)
    rain_clim = rain_s[train_date_mask].groupby(full_daily_index[train_date_mask].month).mean()
    disc_clim = disc_s[train_date_mask].groupby(full_daily_index[train_date_mask].month).mean()
    assert rain_clim.isna().sum() == 0 and disc_clim.isna().sum() == 0, "train split has gaps?!"

    month_of_day = pd.Series(full_daily_index.month, index=full_daily_index)
    rain_filled_s = rain_s.fillna(month_of_day.map(rain_clim))
    disc_filled_s = disc_s.fillna(month_of_day.map(disc_clim))
    assert rain_filled_s.isna().sum() == 0 and disc_filled_s.isna().sum() == 0, "climatology fill left NaNs"

    if verbose:
        n_rain_filled, n_disc_filled = int((rain_avail_s == 0).sum()), int((disc_avail_s == 0).sum())
        print(f"hourly meteo grid verified complete: {len(hourly):,} rows, "
              f"{hourly['valid_time'].min()} to {hourly['valid_time'].max()}")
        print(f"rainfall:  {n_rain_filled} synthetic (climatology-filled) days")
        print(f"discharge: {n_disc_filled} synthetic (climatology-filled) days")
        print("(the tail of this range is padding beyond the real gauge record, for the decoder's "
              "trailing context points -- the genuine gaps are the 2024 rainfall/discharge cutoffs.)")

    return MeteoGrids(
        hourly_start=hourly['valid_time'].iloc[0].to_numpy(),
        pressure_hpa=hourly['msl_hpa'].to_numpy(dtype=np.float32),
        u10=hourly['u10'].to_numpy(dtype=np.float32),
        v10=hourly['v10'].to_numpy(dtype=np.float32),
        daily_start=full_daily_index[0].to_numpy(),
        rain=rain_filled_s.to_numpy(dtype=np.float32),
        discharge=disc_filled_s.to_numpy(dtype=np.float32),
        rain_avail=rain_avail_s.to_numpy(dtype=np.float32),
        discharge_avail=disc_avail_s.to_numpy(dtype=np.float32),
        meteo_lookback_hours=meteo_lookback_hours,
        river_lookback_days=river_lookback_days,
    )


@dataclass
class Stage3Data:
    df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    grids: MeteoGrids
    coef: object
    utide_full: np.ndarray
    utide_index: pd.DatetimeIndex
    utide_start: np.datetime64
    Y_test: np.ndarray
    Flag_test: np.ndarray
    Time_test: np.ndarray
    Anchor_test: np.ndarray
    Ydec_targets_test: np.ndarray
    train_mean: float
    train_std: float
    scalers: dict
    train_tensors: dict
    val_tensors: dict
    test_tensors: dict
    train_tensors_tideonly: dict = field(default_factory=dict)
    val_tensors_tideonly: dict = field(default_factory=dict)
    test_tensors_tideonly: dict = field(default_factory=dict)


def prepare_stage3(device, horizon_steps, decoder_seq_steps, target_slice, lookback=LOOKBACK,
                   train_stride=TRAIN_STRIDE, data_path=None, meteo_dir=None, verbose=True):
    """Stage 3: multi-resolution windows (tide 10-min, meteo hourly, river daily) plus the
    padded UTide decoder sequence, all scaled with train-split statistics."""
    df = load_gauge_df(data_path, verbose=verbose)
    pressure, wind, rainfall, discharge = load_meteo(meteo_dir, verbose=verbose)
    grids = build_meteo_grids(df, pressure, wind, rainfall, discharge, verbose=verbose)

    coef = fit_utide(df, end=TRAIN_END, verbose=verbose)
    utide_index = pd.date_range(df['DateTime'].min(),
                                df['DateTime'].max() + pd.Timedelta(days=UTIDE_EXT_DAYS), freq='10min')
    utide_full = reconstruct_utide(utide_index, coef)
    utide_start = utide_index[0].to_numpy()
    if verbose:
        print(f"reconstructed {len(utide_index):,} steps, {utide_index.min()} to {utide_index.max()} "
              f"({UTIDE_EXT_DAYS}d past the last gauge row)")

    train_df, val_df, test_df = split_chronological(df, verbose=verbose)

    def _windows(split_df):
        return build_windows_stage3(split_df, lookback, horizon_steps, grids,
                                    utide_full, utide_start, decoder_seq_steps)

    (X_train, Xm_train, Xr_train, Ydec_train,
     Y_train, Flag_train, _, _) = _windows(train_df)
    X_train, Xm_train, Xr_train, Ydec_train, Y_train, Flag_train = (
        X_train[::train_stride], Xm_train[::train_stride], Xr_train[::train_stride],
        Ydec_train[::train_stride], Y_train[::train_stride], Flag_train[::train_stride],
    )
    X_val, Xm_val, Xr_val, Ydec_val, Y_val, Flag_val, _, _ = _windows(val_df)
    (X_test, Xm_test, Xr_test, Ydec_test,
     Y_test, Flag_test, Time_test, Anchor_test) = _windows(test_df)

    if verbose:
        print(f"windows -> train: {X_train.shape} (stride {train_stride}), "
              f"val: {X_val.shape}, test: {X_test.shape}")
        print(f"per-branch shapes (train): tide={X_train.shape} meteo={Xm_train.shape} "
              f"river={Xr_train.shape} decoder-utide={Ydec_train.shape} targets={Y_train.shape}")

    train_mean = float(train_df['Observed_ODN'].mean())
    train_std = float(train_df['Observed_ODN'].std())
    scalers = {
        'tide_utide': {'mean': train_mean, 'std': train_std},
        'pressure': {'mean': float(Xm_train[..., 0].mean()), 'std': float(Xm_train[..., 0].std())},
        'u10': {'mean': float(Xm_train[..., 1].mean()), 'std': float(Xm_train[..., 1].std())},
        'v10': {'mean': float(Xm_train[..., 2].mean()), 'std': float(Xm_train[..., 2].std())},
        'rainfall': {'mean': float(Xr_train[..., 0].mean()), 'std': float(Xr_train[..., 0].std())},
        'discharge': {'mean': float(Xr_train[..., 1].mean()), 'std': float(Xr_train[..., 1].std())},
    }
    if verbose:
        print(f"tide/utide:  mean={train_mean:.4f} m,  std={train_std:.4f} m")
        for key in ('pressure', 'u10', 'v10', 'rainfall', 'discharge'):
            print(f"{key:12s} mean={scalers[key]['mean']:.3f}, std={scalers[key]['std']:.3f}")

    def scale_tide(a):
        return (a - train_mean) / train_std

    def scale_meteo(xm):
        out = xm.copy()
        out[..., 0] = (out[..., 0] - scalers['pressure']['mean']) / scalers['pressure']['std']
        out[..., 1] = (out[..., 1] - scalers['u10']['mean']) / scalers['u10']['std']
        out[..., 2] = (out[..., 2] - scalers['v10']['mean']) / scalers['v10']['std']
        return out

    def scale_river(xr):
        out = xr.copy()
        out[..., 0] = (out[..., 0] - scalers['rainfall']['mean']) / scalers['rainfall']['std']
        out[..., 1] = (out[..., 1] - scalers['discharge']['mean']) / scalers['discharge']['std']
        return out  # availability channels (2, 3) left as 0/1, unscaled

    train_tensors = {'X': to_tensor(scale_tide(X_train), device),
                     'Xm': to_tensor(scale_meteo(Xm_train), device),
                     'Xr': to_tensor(scale_river(Xr_train), device),
                     'Ydec': to_tensor(scale_tide(Ydec_train), device),
                     'Y': to_tensor(scale_tide(Y_train), device),
                     'Flag': to_tensor(Flag_train, device, dtype=torch.bool)}
    val_tensors = {'X': to_tensor(scale_tide(X_val), device),
                   'Xm': to_tensor(scale_meteo(Xm_val), device),
                   'Xr': to_tensor(scale_river(Xr_val), device),
                   'Ydec': to_tensor(scale_tide(Ydec_val), device),
                   'Y': to_tensor(scale_tide(Y_val), device),
                   'Flag': to_tensor(Flag_val, device, dtype=torch.bool)}
    test_tensors = {'X': to_tensor(scale_tide(X_test), device),
                    'Xm': to_tensor(scale_meteo(Xm_test), device),
                    'Xr': to_tensor(scale_river(Xr_test), device),
                    'Ydec': to_tensor(scale_tide(Ydec_test), device)}

    tideonly_keys = ('X', 'Ydec', 'Y', 'Flag')
    data = Stage3Data(
        df, train_df, val_df, test_df, grids, coef, utide_full, utide_index, utide_start,
        Y_test, Flag_test, Time_test, Anchor_test, Ydec_test[:, target_slice],
        train_mean, train_std, scalers, train_tensors, val_tensors, test_tensors,
        {k: v for k, v in train_tensors.items() if k in tideonly_keys},
        {k: v for k, v in val_tensors.items() if k in tideonly_keys},
        {k: v for k, v in test_tensors.items() if k in ('X', 'Ydec')},
    )
    return data


def prepare_stage3_reduced(device, data, train_start, horizon_steps, decoder_seq_steps,
                           target_slice, lookback=LOOKBACK, train_stride=TRAIN_STRIDE,
                           name='reduced'):
    """One scenario of stage 3's training-data-volume ablation, mirroring stage 2's: UTide
    refit on the reduced window, val/test windows rebuilt against that refit, and every
    scaler fit on only the data the scenario was allowed to see."""
    df = data.df
    coef_r = fit_utide(df, start=train_start, end=TRAIN_END, verbose=False)
    recon_r = reconstruct_utide(data.utide_index, coef_r)

    tr_df_r = df[(df['DateTime'] >= train_start) & (df['DateTime'] < TRAIN_END)].reset_index(drop=True)

    def _windows(split_df):
        return build_windows_stage3(split_df, lookback, horizon_steps, data.grids,
                                    recon_r, data.utide_start, decoder_seq_steps)

    Xtr, Xmtr, Xrtr, Ydtr, Ytr, Ftr, _, _ = _windows(tr_df_r)
    Xva, Xmva, Xrva, Ydva, Yva, Fva, _, _ = _windows(data.val_df)
    Xte, Xmte, Xrte, Ydte, Yte, Fte, _, _ = _windows(data.test_df)

    Xtr, Xmtr, Xrtr, Ydtr, Ytr, Ftr = (Xtr[::train_stride], Xmtr[::train_stride], Xrtr[::train_stride],
                                       Ydtr[::train_stride], Ytr[::train_stride], Ftr[::train_stride])

    mean_, std_ = float(tr_df_r['Observed_ODN'].mean()), float(tr_df_r['Observed_ODN'].std())
    p_ = (float(Xmtr[..., 0].mean()), float(Xmtr[..., 0].std()))
    u_ = (float(Xmtr[..., 1].mean()), float(Xmtr[..., 1].std()))
    v_ = (float(Xmtr[..., 2].mean()), float(Xmtr[..., 2].std()))
    r_ = (float(Xrtr[..., 0].mean()), float(Xrtr[..., 0].std()))
    d_ = (float(Xrtr[..., 1].mean()), float(Xrtr[..., 1].std()))

    def sc_tide(a):
        return (a - mean_) / std_

    def sc_meteo(xm):
        out = xm.copy()
        out[..., 0] = (out[..., 0] - p_[0]) / p_[1]
        out[..., 1] = (out[..., 1] - u_[0]) / u_[1]
        out[..., 2] = (out[..., 2] - v_[0]) / v_[1]
        return out

    def sc_river(xr):
        out = xr.copy()
        out[..., 0] = (out[..., 0] - r_[0]) / r_[1]
        out[..., 1] = (out[..., 1] - d_[0]) / d_[1]
        return out

    print(f"[{name}] train {tr_df_r.DateTime.min().date()} -> {tr_df_r.DateTime.max().date()} "
          f"({Xtr.shape[0]:,} windows, stride {train_stride})  val {Xva.shape[0]:,}  test {Xte.shape[0]:,}")

    return {
        'name': name, 'mean': mean_, 'std': std_,
        'train_tensors': {'X': to_tensor(sc_tide(Xtr), device), 'Xm': to_tensor(sc_meteo(Xmtr), device),
                          'Xr': to_tensor(sc_river(Xrtr), device), 'Ydec': to_tensor(sc_tide(Ydtr), device),
                          'Y': to_tensor(sc_tide(Ytr), device), 'Flag': to_tensor(Ftr, device, dtype=torch.bool)},
        'val_tensors': {'X': to_tensor(sc_tide(Xva), device), 'Xm': to_tensor(sc_meteo(Xmva), device),
                        'Xr': to_tensor(sc_river(Xrva), device), 'Ydec': to_tensor(sc_tide(Ydva), device),
                        'Y': to_tensor(sc_tide(Yva), device), 'Flag': to_tensor(Fva, device, dtype=torch.bool)},
        'test_tensors': {'X': to_tensor(sc_tide(Xte), device), 'Xm': to_tensor(sc_meteo(Xmte), device),
                         'Xr': to_tensor(sc_river(Xrte), device), 'Ydec': to_tensor(sc_tide(Ydte), device)},
        'Y_test': Yte, 'Flag_test': Fte, 'Ydec_test': Ydte[:, target_slice],
    }
