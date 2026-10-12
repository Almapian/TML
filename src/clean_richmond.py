"""Clean the Richmond Lock gauge record into the same shape as Data/southend_pier_data.csv.

Mirrors the Southend cleaning (EDA/southend_pier_data_cleaning_summary.md): identify faulty
readings -> null them -> refit UTide without them -> patch. Reads the intermediate cleaned
`3_Cleaned/01_RICHMOND.csv` (ODN-corrected, 10-minute) and writes
`Data/richmond_lock_data.csv`, which the modelling scripts read with GAUGE = 'richmond'.

    python src/clean_richmond.py
    # or run the `# %%` cells interactively in VS Code

Four fault types, each with a threshold chosen from the data (CONFIG below):

* spikes / chatter -- the UTide residual jumps away from its own 70-minute rolling median
  (isolated spikes up to ~15 m in 2009, falling-limb chatter in 2010-2012). 99.9% of all
  deviations are < 0.22 m, so 0.3 m only catches the tail. Two passes, so a spike hidden
  by a neighbouring spike is caught once the first is removed.
* stuck sensor -- >= 6 identical consecutive readings (1h) while the tide moves > 0.1 m.
  Most sit at -0.61 m ODN, i.e. exactly 0 on the gauge's chart datum.
* datum shift -- the whole trace offset by ~+3 m for five days (Aug 2009). Told apart from
  a genuine fluvial flood by the residual's 25-hour rolling *minimum*: a flood raises the
  low waters but the residual still returns towards zero around high water each tide; a
  datum shift keeps it high throughout.
* out of range -- outside [-2.5, 6.0] m ODN (the highest plausible reading is ~5.2 m).

Unlike Southend, a patched value is UTide *plus the residual interpolated across the
patch*, not UTide alone: river flow at Richmond moves low water by up to ~3 m, which UTide
cannot represent (its RMSE here is ~0.5 m vs ~0.25 m at Southend), so a pure-UTide patch
would put ~1 m steps into the record. The same fill is used for short data gaps. Any
flagged or missing run longer than MAX_FILL is left as a genuine gap instead (its rows are
dropped), so nothing longer than that is ever invented. Patched and filled values carry a
flag, which the modelling pipeline excludes from every loss and metric; they are only ever
seen as encoder *inputs*.

Before/after plots of the largest flagged episodes go to outputs/richmond/cleaning/ --
look at them before training.
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import paths
from utils.models import config as C
from utils.models import datasets

# %% ------------------------------------------------------------------ CONFIG
RAW_CSV = paths.CLEANED_DIR / '01_RICHMOND.csv'
OUT_CSV = paths.RICHMOND_CSV
PLOT_DIR = paths.OUTPUTS_DIR / 'richmond' / 'cleaning'
WRITE_OUTPUT = True

LAT = C.GAUGES['richmond']['lat']
CHART_DATUM_BELOW_ODN = 0.61          # PLA gauge table / 3_Cleaned/00_gauge_metadata.csv;
                                      # 3_Cleaned/01_RICHMOND.csv already carries Observed_ODN
UTIDE_FIT_STRIDE = 6                  # hourly is plenty for QC; keeps the 21-year fit to ~2 GB

SPIKE_DEV_M = 0.3                     # |residual - rolling median| (m)
SPIKE_WINDOW = 7                      # rolling-median window, 10-minute steps (70 min)
STUCK_MIN_RUN = 6                     # identical consecutive readings (1h)
STUCK_MIN_TIDE_RANGE_M = 0.10         # ...while UTide moves at least this much
SHIFT_WINDOW = 150                    # 25h, two semi-diurnal cycles
SHIFT_MIN_RESID_M = 1.0               # rolling-min residual above this -> datum shift
VALID_RANGE_M = (-2.5, 6.0)           # m ODN
MAX_FILL = pd.Timedelta('6h')         # longest flagged/missing run that is patched

N_EPISODE_PLOTS = 40                  # largest episodes plotted, before/after

STEPS = {'stage 1 (lookback + 1)': C.LOOKBACK + 1,
         'stage 2 (to 168h)': C.LOOKBACK + max(C.STAGE2_HORIZONS.values()),
         'stage 3 (to 30d)': C.LOOKBACK + max(C.STAGE3_HORIZONS.values())}

# %% Load onto the full 10-minute grid --------------------------------------
raw = pd.read_csv(RAW_CSV, parse_dates=['DateTime'])
raw = raw.drop_duplicates('DateTime').sort_values('DateTime')
grid = pd.date_range(raw['DateTime'].min(), raw['DateTime'].max(), freq='10min')
df = raw.set_index('DateTime').reindex(grid)
df.index.name = 'DateTime'
obs = df['Observed_ODN'].astype(float)
missing = obs.isna()
print(f"{len(raw):,} readings on a {len(grid):,}-step grid: {missing.sum():,} missing "
      f"({100 * missing.mean():.2f}%)")


def fit_tide(values, exclude):
    """UTide fit on the hourly grid with `exclude` nulled, reconstructed on every step."""
    fit_df = pd.DataFrame({'DateTime': grid, 'Observed_ODN': values.to_numpy(),
                           'is_flagged': (exclude | values.isna()).to_numpy()})
    coef = datasets.fit_utide(fit_df, end=grid[-1] + pd.Timedelta('10min'), lat=LAT,
                              stride=UTIDE_FIT_STRIDE)
    return pd.Series(datasets.reconstruct_utide(grid, coef), index=grid)


# %% 1. First-pass fit (only physically impossible values excluded) ----------
out_of_range = obs.notna() & ((obs < VALID_RANGE_M[0]) | (obs > VALID_RANGE_M[1]))
tide = fit_tide(obs, out_of_range)
resid = obs - tide
print(f"first-pass UTide RMSE: {np.sqrt(np.nanmean(resid[~out_of_range] ** 2)):.3f} m")

# %% 2. Identify faults ------------------------------------------------------
# datum shift: residual's 25h rolling minimum stays high; the core is then widened by half
# a window either side (the rolling minimum can't see an episode's first/last 12.5h) but
# only onto points that are themselves well above the tide
roll_min = resid.where(~out_of_range).rolling(SHIFT_WINDOW, center=True, min_periods=SHIFT_WINDOW // 2).min()
shift_core = (roll_min > SHIFT_MIN_RESID_M).astype(float)
shift_near = shift_core.rolling(SHIFT_WINDOW + 1, center=True, min_periods=1).max() > 0
datum_shift = shift_near & (resid > SHIFT_MIN_RESID_M)

# stuck sensor: runs of identical readings while the tide keeps moving
run_id = (obs.diff() != 0).cumsum()
run_len = obs.groupby(run_id).transform('size')
run_tide_range = tide.groupby(run_id).transform('max') - tide.groupby(run_id).transform('min')
stuck = obs.notna() & (run_len >= STUCK_MIN_RUN) & (run_tide_range > STUCK_MIN_TIDE_RANGE_M)

# spikes/chatter: residual far from its own rolling median; two passes
spike = pd.Series(False, index=grid)
excluded = out_of_range | datum_shift | stuck
for _ in range(2):
    r = resid.where(~(excluded | spike))
    med = r.rolling(SPIKE_WINDOW, center=True, min_periods=3).median()
    spike |= (r - med).abs() > SPIKE_DEV_M
spike &= ~excluded

flagged = out_of_range | datum_shift | stuck | spike
counts = {'spike / chatter': spike.sum(), 'stuck sensor': stuck.sum(),
          'datum shift': datum_shift.sum(), 'out of range': (out_of_range & ~datum_shift).sum()}
for k, v in counts.items():
    print(f"{k:16s} {v:>7,} readings ({100 * v / obs.notna().sum():.3f}% of readings)")

# %% 3. Refit without the faulty readings -----------------------------------
tide = fit_tide(obs, flagged)
resid_clean = (obs - tide).where(~flagged)
print(f"refit UTide RMSE (unflagged readings): {np.sqrt(np.nanmean(resid_clean ** 2)):.3f} m")

# %% 4. Patch short runs, drop long ones -------------------------------------
hole = flagged | missing
hole_id = (hole != hole.shift()).cumsum()
hole_len = hole.groupby(hole_id).transform('size')
fillable = hole & (hole_len * pd.Timedelta('10min') <= MAX_FILL)
dropped = hole & ~fillable

resid_interp = resid_clean.interpolate(method='time', limit_area='inside')
patched = obs.where(~hole)
patched[fillable] = (tide + resid_interp)[fillable]
fillable &= patched.notna()           # a run at the very start/end has nothing to interpolate from
dropped = hole & ~fillable

method = pd.Series('none', index=grid, dtype=object)
method[fillable & missing] = 'gap_fill'
method[fillable & out_of_range] = 'out_of_range'
method[fillable & datum_shift] = 'datum_shift'
method[fillable & stuck] = 'stuck'
method[fillable & spike] = 'spike'

print(f"\npatched: {fillable.sum():,} steps "
      f"({(fillable & missing).sum():,} gap fill, {(fillable & flagged).sum():,} faulty readings)")
print(f"left as gaps (runs > {MAX_FILL}): {dropped.sum():,} steps, of which "
      f"{(dropped & flagged).sum():,} were faulty readings rather than missing ones")

# %% 5. Assemble, in the Southend schema ------------------------------------
# ODN = chart-datum reading - 0.61 m (PLA gauge table); both columns rounded alike so the
# offset is exact on every row
keep = ~dropped
odn = patched[keep].round(4)
out = pd.DataFrame({
    'DateTime': grid[keep],
    'idSite': '01_RICHMOND',
    'SiteName': 'Richmond Lock',
    'Observed': (odn + CHART_DATUM_BELOW_ODN).round(4).to_numpy(),
    'is_imputed': (fillable & ~spike & ~stuck)[keep].to_numpy(),
    'Observed_ODN': odn.to_numpy(),
    'is_chatter_flagged': (fillable & spike)[keep].to_numpy(),
    'imputation_method': method[keep].to_numpy(),
    'is_stuck_flagged': (fillable & stuck)[keep].to_numpy(),
})
assert out['Observed_ODN'].notna().all()
flag_any = out['is_imputed'] | out['is_chatter_flagged'] | out['is_stuck_flagged']
print(f"\noutput: {len(out):,} rows, {flag_any.sum():,} flagged ({100 * flag_any.mean():.2f}%), "
      f"{(out['DateTime'].diff() > C.STEP).sum()} gaps remain")
print(out['imputation_method'].value_counts().to_string())


# %% 6. Window yield per stage (share of the grid that can anchor a window) --
def window_yield(times, need):
    seg = (times.diff() > C.STEP).cumsum()
    lengths = seg.groupby(seg).size()
    return (lengths[lengths >= need] - need + 1).sum() / len(grid)


raw_times = pd.Series(raw['DateTime'].to_numpy())
print("\nwindow yield (before -> after cleaning):")
for name, need in STEPS.items():
    print(f"  {name:24s} {100 * window_yield(raw_times, need):5.1f}% -> "
          f"{100 * window_yield(out['DateTime'], need):5.1f}%")

if WRITE_OUTPUT:
    out.to_csv(OUT_CSV, index=False)
    print(f"\nsaved {OUT_CSV}")

# %% 7. Before/after plots of the largest flagged episodes -------------------
# episodes = flagged readings merged when closer than 6h apart
PLOT_DIR.mkdir(parents=True, exist_ok=True)
for old in PLOT_DIR.glob('episode_*.png'):
    old.unlink()
flag_times = grid[flagged.to_numpy()]
if len(flag_times):
    ep_id = np.concatenate([[0], np.cumsum(np.diff(flag_times) > np.timedelta64(6, 'h'))])
    episodes = (pd.DataFrame({'t': flag_times, 'ep': ep_id}).groupby('ep')['t']
                .agg(['min', 'max', 'size']).sort_values('size', ascending=False))
    print(f"\n{len(episodes)} flagged episodes; plotting the {min(N_EPISODE_PLOTS, len(episodes))} largest "
          f"to {PLOT_DIR}")
    patched_full = patched.where(~dropped)
    for rank, (_, ep) in enumerate(episodes.head(N_EPISODE_PLOTS).iterrows(), start=1):
        a, b = ep['min'] - pd.Timedelta('1D'), ep['max'] + pd.Timedelta('1D')
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax0.plot(obs[a:b].clip(-3, 9), lw=0.8, color='tab:orange', label='raw reading')
        ax0.plot(tide[a:b], lw=0.8, color='tab:blue', alpha=0.7, label='UTide (refit)')
        ax0.scatter(obs[a:b][flagged[a:b]].index, obs[a:b][flagged[a:b]].clip(-3, 9), s=6,
                    color='red', zorder=3, label='flagged')
        ax0.set_ylabel('m ODN (raw, clipped)')
        ax0.legend(fontsize=7, loc='upper right')
        ax1.plot(patched_full[a:b], lw=0.8, color='black', label='cleaned')
        ax1.plot(tide[a:b], lw=0.8, color='tab:blue', alpha=0.7, label='UTide (refit)')
        ax1.set_ylabel('m ODN (cleaned)')
        ax1.legend(fontsize=7, loc='upper right')
        kinds = ', '.join(k for k, m in [('spike', spike), ('stuck', stuck), ('datum shift', datum_shift),
                                         ('out of range', out_of_range)] if m[ep['min']:ep['max']].any())
        ax0.set_title(f"#{rank}: {ep['min']} -> {ep['max']}  ({ep['size']} flagged; {kinds})", fontsize=9)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"episode_{rank:02d}_{ep['min']:%Y%m%d}.png", dpi=90)
        plt.close(fig)

# %% Done
print("\nReview the episode plots before training; tune the CONFIG thresholds if a genuine "
      "flood or surge was flagged.")
