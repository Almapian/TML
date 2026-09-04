"""Sliding-window construction for each modelling stage.

All three builders share one rule: windows are only ever built *inside* a gap-free
segment of the gauge record, so no sample straddles a data gap. Splits are made before
windowing, so no window crosses a train/validation/test boundary either.

    build_windows        stage 1  -- lookback -> single next value
    build_windows_multi  stage 2  -- lookback -> one target per horizon, plus a UTide
                                     decoder feature per horizon
    build_windows_stage3 stage 3  -- as above, plus hourly meteo and daily river
                                     lookbacks and a densified/padded UTide sequence
"""
from dataclasses import dataclass

import numpy as np

from .config import STEP


def build_windows(sub_df, lookback):
    """Stage 1: (X, y, flag, time) of sliding `lookback` -> next-value windows."""
    sub_df = sub_df.reset_index(drop=True)
    is_gap = sub_df['DateTime'].diff() > STEP
    seg_id = is_gap.cumsum().to_numpy()

    values = sub_df['Observed_ODN'].to_numpy(dtype=np.float32)
    flagged = sub_df['is_flagged'].to_numpy()
    times = sub_df['DateTime'].to_numpy()

    X_list, y_list, flag_list, time_list = [], [], [], []
    for seg in np.unique(seg_id):
        idx = np.where(seg_id == seg)[0]
        if len(idx) < lookback + 1:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(values[idx], lookback + 1)
        X_list.append(windows[:, :lookback])
        y_list.append(windows[:, lookback])
        flag_list.append(flagged[idx][lookback:])
        time_list.append(times[idx][lookback:])

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    flag = np.concatenate(flag_list, axis=0)
    time = np.concatenate(time_list, axis=0)
    return X, y, flag, time


def build_windows_multi(sub_df, lookback, horizon_steps):
    """Stage 2: (X, Ydec, Y, Flag, Time), one target and one UTide decoder feature per
    horizon. Requires a `utide_recon` column alongside `Observed_ODN`.
    """
    sub_df = sub_df.reset_index(drop=True)
    is_gap = sub_df['DateTime'].diff() > STEP
    seg_id = is_gap.cumsum().to_numpy()

    values = sub_df['Observed_ODN'].to_numpy(dtype=np.float32)
    utide_vals = sub_df['utide_recon'].to_numpy(dtype=np.float32)
    flagged = sub_df['is_flagged'].to_numpy()
    times = sub_df['DateTime'].to_numpy()

    max_h = max(horizon_steps)
    horizon_arr = np.array(horizon_steps)

    X_list, Ydec_list, Y_list, Flag_list, Time_list = [], [], [], [], []
    for seg in np.unique(seg_id):
        idx = np.where(seg_id == seg)[0]
        n = len(idx)
        if n < lookback + max_h:
            continue

        seg_values = values[idx]
        seg_utide = utide_vals[idx]
        seg_flag = flagged[idx]
        seg_time = times[idx]

        n_windows = n - lookback - max_h + 1
        enc_windows = np.lib.stride_tricks.sliding_window_view(seg_values, lookback)[:n_windows]
        starts = np.arange(n_windows)
        # window `s` covers positions s .. s+lookback-1; horizon k's target sits at
        # s + (lookback - 1) + k, i.e. k steps after the last encoder value (k=1 -> next step)
        target_positions = starts[:, None] + (lookback - 1) + horizon_arr[None, :]

        X_list.append(enc_windows)
        Ydec_list.append(seg_utide[target_positions])
        Y_list.append(seg_values[target_positions])
        Flag_list.append(seg_flag[target_positions])
        Time_list.append(seg_time[target_positions])

    X = np.concatenate(X_list, axis=0)
    Ydec = np.concatenate(Ydec_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    Flag = np.concatenate(Flag_list, axis=0)
    Time = np.concatenate(Time_list, axis=0)
    return X, Ydec, Y, Flag, Time


@dataclass
class MeteoGrids:
    """Gap-free hourly (pressure/wind) and daily (rainfall/discharge) grids, indexed by
    absolute wall-clock offset from their own start rather than by the gauge's segment
    structure. Built once by `datasets.build_meteo_grids` and reused for every split.
    """
    hourly_start: np.datetime64
    pressure_hpa: np.ndarray
    u10: np.ndarray
    v10: np.ndarray
    daily_start: np.datetime64
    rain: np.ndarray
    discharge: np.ndarray
    rain_avail: np.ndarray
    discharge_avail: np.ndarray
    meteo_lookback_hours: int
    river_lookback_days: int


def gather_window(full_arr, end_idx, length):
    """Window of `length` ending at (inclusive) index `end_idx`, gathered for every row."""
    starts_ = end_idx - (length - 1)
    sw = np.lib.stride_tricks.sliding_window_view(full_arr, length)
    return sw[starts_]


def build_windows_stage3(sub_df, lookback, horizon_steps, grids, utide_full, utide_start,
                         decoder_seq_steps):
    """Stage 3: (X, X_meteo, X_river, Ydec_dense, Y, Flag, Time, anchor_time).

    Each anchor carries the tide lookback, the last `meteo_lookback_hours` of
    [pressure, u10, v10], the last `river_lookback_days` of
    [rainfall, discharge, rainfall_available, discharge_available], and the padded UTide
    decoder sequence at `decoder_seq_steps` offsets from the anchor.

    `utide_full`/`utide_start` are passed explicitly so the training-data-volume ablation
    can supply its own re-fit reconstruction -- each reduced-training scenario then sees
    the UTide input a model trained only on that window would actually have had.
    """
    sub_df = sub_df.reset_index(drop=True)
    is_gap = sub_df['DateTime'].diff() > STEP
    seg_id = is_gap.cumsum().to_numpy()

    values = sub_df['Observed_ODN'].to_numpy(dtype=np.float32)
    flagged = sub_df['is_flagged'].to_numpy()
    times = sub_df['DateTime'].to_numpy()

    max_h = max(horizon_steps)
    horizon_arr = np.array(horizon_steps)

    X_list, Y_list, Flag_list, Time_list = [], [], [], []
    for seg in np.unique(seg_id):
        idx = np.where(seg_id == seg)[0]
        n = len(idx)
        if n < lookback + max_h:
            continue

        seg_values = values[idx]
        seg_flag = flagged[idx]
        seg_time = times[idx]

        n_windows = n - lookback - max_h + 1
        enc_windows = np.lib.stride_tricks.sliding_window_view(seg_values, lookback)[:n_windows]
        starts = np.arange(n_windows)
        # unchanged from stage 2's build_windows_multi
        target_positions = starts[:, None] + (lookback - 1) + horizon_arr[None, :]

        X_list.append(enc_windows)
        Y_list.append(seg_values[target_positions])
        Flag_list.append(seg_flag[target_positions])
        Time_list.append(seg_time[target_positions])

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    Flag = np.concatenate(Flag_list, axis=0)
    Time = np.concatenate(Time_list, axis=0)          # (N, n_horizons) -- target time per horizon
    anchor_time = Time[:, 0] - np.timedelta64(horizon_steps[0] * 10, 'm')

    # --- meteo/river/UTide: gap-free grids, gathered by absolute wall-clock offset,
    # independent of the tide gauge's own segment structure ---
    anchor_utide_idx = ((anchor_time - utide_start) / np.timedelta64(10, 'm')).astype(np.int64)
    dec_offsets = np.array(decoder_seq_steps)
    assert anchor_utide_idx.max() + dec_offsets.max() < len(utide_full), "utide extension buffer too small!"
    Ydec_dense = utide_full[anchor_utide_idx[:, None] + dec_offsets[None, :]]

    anchor_hourly_idx = np.floor((anchor_time - grids.hourly_start) / np.timedelta64(1, 'h')).astype(np.int64)
    anchor_daily_idx = np.floor(
        (anchor_time.astype('datetime64[D]') - grids.daily_start.astype('datetime64[D]')) / np.timedelta64(1, 'D')
    ).astype(np.int64)

    valid = ((anchor_hourly_idx - (grids.meteo_lookback_hours - 1) >= 0)
             & (anchor_daily_idx - (grids.river_lookback_days - 1) >= 0))

    X_meteo = np.stack([
        gather_window(grids.pressure_hpa, anchor_hourly_idx, grids.meteo_lookback_hours),
        gather_window(grids.u10, anchor_hourly_idx, grids.meteo_lookback_hours),
        gather_window(grids.v10, anchor_hourly_idx, grids.meteo_lookback_hours),
    ], axis=-1)

    X_river = np.stack([
        gather_window(grids.rain, anchor_daily_idx, grids.river_lookback_days),
        gather_window(grids.discharge, anchor_daily_idx, grids.river_lookback_days),
        gather_window(grids.rain_avail, anchor_daily_idx, grids.river_lookback_days),
        gather_window(grids.discharge_avail, anchor_daily_idx, grids.river_lookback_days),
    ], axis=-1)

    return (X[valid], X_meteo[valid], X_river[valid], Ydec_dense[valid],
            Y[valid], Flag[valid], Time[valid], anchor_time[valid])
