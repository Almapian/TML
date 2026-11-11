"""No-leakage checks for the stage 3 river inputs (utils/models/windowing.py).

NRFA daily values are 09:00-09:00 GMT water-day means dated by their start day, so the value
dated D is complete only at 09:00 on D+1. With config.RIVER_AVAILABILITY_LAG_HOURS = 33 no
anchor may see a river value whose water day has not ended.

    python tests/test_stage3_windowing.py
"""
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from utils.models import config as C
from utils.models import windowing as W

DAY0 = np.datetime64('2010-01-01')


def _grids(n_days=80, lag=C.RIVER_AVAILABILITY_LAG_HOURS, river_days=5, meteo_hours=6):
    """Daily river series whose value equals its own day index, so a window says which days it saw."""
    n_hours = n_days * 24
    day_values = np.arange(n_days, dtype=np.float32)
    return W.MeteoGrids(
        hourly_start=DAY0.astype('datetime64[ns]'), pressure_hpa=np.zeros(n_hours, np.float32),
        u10=np.zeros(n_hours, np.float32), v10=np.zeros(n_hours, np.float32),
        daily_start=DAY0.astype('datetime64[ns]'), rain=day_values.copy(), discharge=day_values.copy(),
        rain_avail=np.ones(n_days, np.float32), discharge_avail=np.ones(n_days, np.float32),
        meteo_lookback_hours=meteo_hours, river_lookback_days=river_days, river_lag_hours=lag)


def _windows(lag, n_days=40):
    grids = _grids(lag=lag)
    times = pd.date_range(DAY0, periods=n_days * 144, freq='10min')
    df = pd.DataFrame({'DateTime': times, 'Observed_ODN': np.zeros(len(times), np.float32),
                       'is_flagged': np.zeros(len(times), bool)})
    utide = np.zeros(len(times) + 1000, np.float32)
    X, Xm, Xr, Yd, Y, F, T, A = W.build_windows_stage3(df, 6, [1, 6], grids, utide, times[0].to_datetime64(), [0, 1, 6])
    return Xr, A


def test_river_day_index_boundaries():
    d = np.datetime64('2010-01-10T00:00')
    cases = {  # anchor -> latest usable day index (day 9 is dated 2010-01-10)
        d + np.timedelta64(0, 'm'): 7,                          # 00:00 on D: day D-1 ends 09:00 D, not yet
        d + np.timedelta64(8 * 60 + 50, 'm'): 7,                # 08:50 on D: still not
        d + np.timedelta64(9 * 60, 'm'): 8,                     # 09:00 on D: day D-1 is now complete
        d + np.timedelta64(33 * 60 - 10, 'm'): 8,               # 08:50 on D+1: day D not complete
        d + np.timedelta64(33 * 60, 'm'): 9,                    # 09:00 on D+1: day D complete
    }
    for anchor, want in cases.items():
        got = int(W.river_day_index(np.array([anchor]), DAY0, 33)[0])
        assert got == want, f'{anchor}: got day {got}, want {want}'


def test_legacy_mapping_is_calendar_day():
    anchor = np.array([np.datetime64('2010-01-10T00:00'), np.datetime64('2010-01-10T23:50')])
    assert list(W.river_day_index(anchor, DAY0, 0)) == [9, 9]


def test_no_window_sees_an_unfinished_water_day():
    Xr, A = _windows(C.RIVER_AVAILABILITY_LAG_HOURS)
    latest_seen = Xr[:, -1, 1]                                            # discharge = day index
    day_end = DAY0.astype('datetime64[m]') + (latest_seen.astype(np.int64) + 1) * np.timedelta64(1, 'D') \
        + np.timedelta64(9, 'h')                                          # 09:00 on the next day
    assert np.all(day_end <= A.astype('datetime64[m]')), 'a window used a river value before it was complete'
    # and it is the most recent complete one, not an older value
    assert np.all(day_end + np.timedelta64(1, 'D') > A.astype('datetime64[m]'))


def test_legacy_mapping_leaks():
    """The original mapping does use unfinished water days -- the bug this fixes."""
    Xr, A = _windows(C.LEGACY_RIVER_LAG_HOURS)
    latest_seen = Xr[:, -1, 1]
    day_end = DAY0.astype('datetime64[m]') + (latest_seen.astype(np.int64) + 1) * np.timedelta64(1, 'D') \
        + np.timedelta64(9, 'h')
    assert np.any(day_end > A.astype('datetime64[m]'))


def test_window_is_contiguous_and_ends_at_latest_day():
    Xr, A = _windows(C.RIVER_AVAILABILITY_LAG_HOURS)
    days = Xr[:, :, 1]
    assert np.all(np.diff(days, axis=1) == 1), 'river lookback days are not consecutive'


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
