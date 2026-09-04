"""Download and consolidate the meteorological covariates for the Thames Estuary.

Script form of `src/meteo_data_download_colab (1).ipynb` (and of the superseded
`src/meteo_data.ipynb` prototype). Fetches:

  * atmospheric pressure and 10m u/v wind (ERA5, via the Copernicus CDS)
  * river discharge and catchment rainfall (NRFA, Thames at Kingston, station 39001)

and writes the four consolidated CSVs the stage-3 model consumes. ERA5 is pulled in
quarterly chunks so no single request trips the CDS cost limit and an interrupted run only
costs the request in progress; each chunk is skipped if its file already exists, so the
script is safe to stop and resume.

    python src/download_meteo_data.py
    # or run the `# %%` cells interactively in VS Code

**Credentials.** ERA5 needs a free CDS account. `cdsapi.Client()` reads `~/.cdsapirc`
automatically; if the environment variable CDS_API_KEY is set instead, this script writes
that file for you (the same thing the Colab notebook did with Colab Secrets). You must
also accept the ERA5 licence once, on the dataset page, or every request is rejected:
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
NRFA needs no credentials at all -- it is a fully open API.

The consolidated output lands in `3_Cleaned/meteo/` (gitignored working data). The copies
the modelling actually reads live in `Data/meteo/` -- re-copy them if you re-download.
"""
# %% Bootstrap
import pathlib
import sys

_start = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / '_')).resolve()
REPO_ROOT = next(p for p in _start.parents if (p / 'utils' / 'paths.py').exists())
sys.path.insert(0, str(REPO_ROOT))

# %% Imports
import glob
import io
import os
import random
import time

import pandas as pd
import requests

from utils import paths

# %% ------------------------------------------------------------------ CONFIG
MET_DIR = paths.ensure_dir(paths.CLEANED_DIR / "meteo_download")
ERA5_RAW_DIR = paths.ensure_dir(MET_DIR / "era5_raw")        # one netCDF per (year, quarter)
NRFA_DIR = paths.ensure_dir(MET_DIR / "nrfa")                 # raw NRFA responses, as returned
CONSOLIDATED_DIR = paths.ensure_dir(paths.CLEANED_METEO_DIR)  # 3_Cleaned/meteo -- the model's input

STUDY_START, STUDY_END = '2004-01-01', '2024-12-31'
YEARS = range(2004, 2025)
MONTHS_PER_CHUNK = 3   # quarterly; drop to 1 (monthly) if a request trips the cost limit
VARIABLES = ["mean_sea_level_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind"]

NRFA_BASE = "https://nrfaapps.ceh.ac.uk/nrfa/ws/time-series"
NRFA_STATION = 39001   # Thames at Kingston, just upstream of Teddington Weir (the tidal limit)

RUN_ERA5_DOWNLOAD = True    # False to skip straight to consolidation of what is already on disk
RUN_NRFA_DOWNLOAD = True
RUN_CONSOLIDATION = True

# %% Gauge coordinates and download region ---------------------------------
# Transcribed from the PLA national grid reference table (WGS84 lat/long column). One
# correction: the table lists Southend Pier's longitude as 00 43.43'N, but N/S only
# applies to latitude. Southend sits east of Coryton (0.51E) and west of Shivering Sands
# (1.11E), so the value is almost certainly 00 43.43'E, and that is what is used here.
# Hammersmith, Westminster and Herne Bay are discontinued with no coordinates in the
# source table, so they are left out.
def dms_to_dd(deg, minutes, hemisphere):
    """Degrees + decimal minutes + hemisphere letter -> signed decimal degrees."""
    dd = deg + minutes / 60
    if hemisphere in ('S', 'W'):
        dd *= -1
    return dd


GAUGES_DMS = {
    'Richmond Lock': ((51, 27.78, 'N'), (0, 19.05, 'W')),
    'Chelsea Bridge': ((51, 29.04, 'N'), (0, 8.98, 'W')),
    'London Bridge': ((51, 30.45, 'N'), (0, 4.74, 'W')),
    'Charlton': ((51, 29.66, 'N'), (0, 1.58, 'E')),
    'North Woolwich': ((51, 29.92, 'N'), (0, 2.77, 'E')),
    'Erith': ((51, 28.91, 'N'), (0, 11.21, 'E')),
    'Tilbury': ((51, 27.40, 'N'), (0, 20.10, 'E')),
    'Gravesend Denton': ((51, 26.67, 'N'), (0, 23.69, 'E')),
    'Coryton Thameshaven': ((51, 30.28, 'N'), (0, 30.31, 'E')),
    'Coryton No5 Jetty': ((51, 30.42, 'N'), (0, 31.39, 'E')),
    'Coryton Holehaven': ((51, 30.71, 'N'), (0, 33.07, 'E')),
    'Southend Pier': ((51, 30.87, 'N'), (0, 43.43, 'E')),
    'Shivering Sands': ((51, 28.37, 'N'), (1, 6.39, 'E')),
    'Margate Pile': ((51, 23.68, 'N'), (1, 22.73, 'E')),
    'Margate Harbour': ((51, 23.52, 'N'), (1, 22.68, 'E')),
    'Walton-on-the-Naze': ((51, 50.60, 'N'), (1, 16.80, 'E')),
}
GAUGES = {name: (dms_to_dd(*lat), dms_to_dd(*lon)) for name, (lat, lon) in GAUGES_DMS.items()}


def bbox_from_gauges(gauges, buffer_deg=0.3):
    """[North, West, South, East] with a buffer in degrees, as the CDS API expects."""
    lats = [lat for lat, lon in gauges.values()]
    lons = [lon for lat, lon in gauges.values()]
    return [max(lats) + buffer_deg, min(lons) - buffer_deg,
            min(lats) - buffer_deg, max(lons) + buffer_deg]


GAUGE_BBOX = bbox_from_gauges(GAUGES, buffer_deg=0.3)   # all 16 gauges
SURGE_BBOX = [55, -3, 51, 4]                             # wider southern North Sea fetch region

# Only the gauges actually being modelled. A tight box has far fewer grid points than
# either alternative, which is what lets three variables and a whole quarter go in one
# request (21 years x 4 quarters = 84 requests, rather than 756). Adding a gauge outside
# ACTIVE_BBOX means widening the box and re-fetching.
ACTIVE_GAUGES = ['Southend Pier']
ACTIVE_GAUGES_DICT = {name: GAUGES[name] for name in ACTIVE_GAUGES}
ACTIVE_BBOX = bbox_from_gauges(ACTIVE_GAUGES_DICT, buffer_deg=0.1)

for name, (lat, lon) in GAUGES.items():
    print(f"{name:22s} {lat:8.4f}, {lon:8.4f}")
print(f"\nGAUGE_BBOX (all 16 gauges): {GAUGE_BBOX}")
print(f"SURGE_BBOX (fetch region):  {SURGE_BBOX}")
print(f"ACTIVE_GAUGES: {ACTIVE_GAUGES}")
print(f"Using ACTIVE_BBOX: {ACTIVE_BBOX}")


# %% CDS credentials --------------------------------------------------------
def ensure_cds_credentials():
    """cdsapi reads ~/.cdsapirc on its own; write it from $CDS_API_KEY when that is how
    the token is supplied (the environment-variable equivalent of Colab Secrets)."""
    cdsapirc_path = os.path.expanduser('~/.cdsapirc')
    key = os.environ.get('CDS_API_KEY')

    if key:
        with open(cdsapirc_path, 'w', encoding='utf-8') as fh:
            fh.write(f"url: https://cds.climate.copernicus.eu/api\nkey: {key}\n")
        print(f"wrote credentials from $CDS_API_KEY to {cdsapirc_path}")
    elif os.path.exists(cdsapirc_path):
        print(f"using existing {cdsapirc_path}")
    else:
        raise RuntimeError(
            "No CDS credentials found. Either set the CDS_API_KEY environment variable to "
            "your Personal Access Token from https://cds.climate.copernicus.eu/profile, or "
            f"create {cdsapirc_path} yourself with 'url:' and 'key:' lines. You must also "
            "accept the ERA5 licence once at "
            "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels")
    return cdsapirc_path


# %% ERA5 download ----------------------------------------------------------
def chunk_path(year, start_month, months_per_chunk=MONTHS_PER_CHUNK, out_dir=None):
    out_dir = ERA5_RAW_DIR if out_dir is None else out_dir
    end_month = start_month + months_per_chunk - 1
    return pathlib.Path(out_dir) / f"era5_{year}_{start_month:02d}-{end_month:02d}.nc"


def download_era5_chunk(year, start_month, n_months, variables, area, out_dir, client):
    """Download n_months of ERA5 data from start_month, all variables in one request."""
    end_month = start_month + n_months - 1
    out_path = chunk_path(year, start_month, n_months, out_dir)
    request = {
        "product_type": "reanalysis",
        "variable": variables,
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(start_month, end_month + 1)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area,
        "format": "netcdf",
    }
    client.retrieve("reanalysis-era5-single-levels", request, str(out_path))
    return out_path


if RUN_ERA5_DOWNLOAD:
    import cdsapi

    ensure_cds_credentials()

    chunk_starts = list(range(1, 13, MONTHS_PER_CHUNK))
    tasks = [(y, m) for y in YEARS for m in chunk_starts]
    already_done = sum(chunk_path(y, m).exists() for y, m in tasks)
    print(f"\n{len(tasks)} requests total, {already_done} already on disk, "
          f"{len(tasks) - already_done} left to fetch.")

    client = cdsapi.Client()
    failed = []
    for i, (year, start_month) in enumerate(tasks, start=1):
        if chunk_path(year, start_month).exists():
            continue
        try:
            download_era5_chunk(year, start_month, MONTHS_PER_CHUNK, VARIABLES,
                                ACTIVE_BBOX, ERA5_RAW_DIR, client)
            print(f"[{i}/{len(tasks)}] done: {year} chunk starting month {start_month:02d}")
            time.sleep(random.uniform(15, 40))   # be polite to the queue between requests
        except Exception as e:
            print(f"[{i}/{len(tasks)}] FAILED: {year} chunk starting month {start_month:02d}: {e}")
            failed.append((year, start_month))

    if failed:
        print(f"\n{len(failed)} requests failed -- re-run to retry just those.")
        print("If they keep failing, set MONTHS_PER_CHUNK = 1 above and re-run.")
    else:
        print("\nAll ERA5 requests completed.")


# %% NRFA river discharge and catchment rainfall ----------------------------
# Both series come from Thames at Kingston (station 39001): gdf = gauged daily flow
# (m3/s), cdr = catchment daily rainfall (mm/day), spatially averaged over the catchment
# upstream of the gauge.
def fetch_nrfa(data_type, station=NRFA_STATION):
    """Download an NRFA time series ('gdf' or 'cdr') as raw CSV text."""
    params = {"format": "nrfa-csv", "data-type": data_type, "station": station}
    r = requests.get(NRFA_BASE, params=params, timeout=60)
    r.raise_for_status()
    return r.text


def parse_nrfa_csv(raw_text, min_consecutive=5):
    """Best-effort parse of NRFA's own CSV format, which carries a metadata header of
    variable length before the data table. Locates the first run of lines whose first
    field is a real date and treats that as the start of the data block -- a heuristic,
    not a guaranteed column count, so check the preview before trusting it downstream."""
    lines = raw_text.splitlines()
    start_idx = None
    for i in range(len(lines) - min_consecutive):
        candidate_dates = [pd.to_datetime(lines[j].split(',')[0], errors='coerce')
                           for j in range(i, i + min_consecutive)]
        if all(d is not pd.NaT for d in candidate_dates):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("Couldn't find a run of date-like rows -- inspect the raw text manually.")

    data_lines = lines[start_idx:]
    n_cols = len(data_lines[0].split(','))
    col_names = ['date', 'value'] + [f'col_{k}' for k in range(2, n_cols)]

    try:
        df = pd.read_csv(io.StringIO("\n".join(data_lines)), header=None, names=col_names)
    except Exception:
        print("Automatic parsing failed; first 10 raw data lines for manual inspection:")
        for line in data_lines[:10]:
            print(line)
        raise

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df.dropna(subset=['date']).set_index('date')


flow_df = rain_df = None
if RUN_NRFA_DOWNLOAD:
    flow_raw = fetch_nrfa('gdf')
    (NRFA_DIR / 'kingston_flow_raw.csv').write_text(flow_raw, encoding='utf-8')
    flow_df = parse_nrfa_csv(flow_raw).loc[STUDY_START:STUDY_END]
    print("\nDischarge preview (study period):")
    print(flow_df.head())

    rain_raw = fetch_nrfa('cdr')
    (NRFA_DIR / 'kingston_catchment_rainfall_raw.csv').write_text(rain_raw, encoding='utf-8')
    rain_df = parse_nrfa_csv(rain_raw).loc[STUDY_START:STUDY_END]
    print("\nCatchment rainfall preview (study period):")
    print(rain_df.head())


# %% Consolidate to one file per variable group -----------------------------
def consolidate_to_gauges(nc_dir, gauges, variables, out_csv):
    """Open every ERA5 chunk lazily, extract the nearest grid cell to each gauge, and
    write one wide CSV.

    Only gauges in `gauges` are extracted, deliberately: xarray's nearest-neighbour lookup
    does not fail or warn when a point falls outside the downloaded box -- it silently
    returns whichever grid cell is closest, even if that is the edge of the box and
    nowhere near the gauge. Running this against the full 16-gauge list while ACTIVE_BBOX
    covers only Southend would quietly produce meaningless columns rather than an error.
    """
    import xarray as xr

    files = sorted(glob.glob(os.path.join(nc_dir, "era5_*.nc")))
    if not files:
        raise FileNotFoundError(f"No ERA5 files found in {nc_dir}")

    ds = xr.open_mfdataset(files, combine='by_coords')
    available = list(ds.data_vars)
    missing = [v for v in variables if v not in available]
    if missing:
        print(f"Warning: {missing} not found. Variables in file: {available}")

    frames = {}
    for name, (lat, lon) in gauges.items():
        point = ds.sel(latitude=lat, longitude=lon, method='nearest')
        for var in variables:
            if var in point:
                frames[f"{name}_{var}"] = point[var].to_pandas()

    wide = pd.DataFrame(frames)
    wide.to_csv(out_csv)
    ds.close()
    print(f"Saved {wide.shape[0]} rows x {wide.shape[1]} columns to {out_csv}")
    return wide


if RUN_CONSOLIDATION:
    pressure_csv = CONSOLIDATED_DIR / 'pressure_all_gauges.csv'
    wind_csv = CONSOLIDATED_DIR / 'wind_all_gauges.csv'
    flow_csv = CONSOLIDATED_DIR / 'river_discharge_kingston.csv'
    rain_csv = CONSOLIDATED_DIR / 'rainfall_kingston_catchment.csv'

    pressure_df = consolidate_to_gauges(ERA5_RAW_DIR, ACTIVE_GAUGES_DICT, ['msl'], pressure_csv)
    wind_df = consolidate_to_gauges(ERA5_RAW_DIR, ACTIVE_GAUGES_DICT, ['u10', 'v10'], wind_csv)

    if flow_df is not None:
        flow_df.to_csv(flow_csv)
    if rain_df is not None:
        rain_df.to_csv(rain_csv)

    print("\nConsolidated files:")
    for path in [pressure_csv, wind_csv, flow_csv, rain_csv]:
        if path.exists():
            print(f"  {path}  ({path.stat().st_size / 1024:.0f} KB)")

    print(f"\nPressure/wind date range: {pressure_df.index.min()} to {pressure_df.index.max()}")
    if flow_df is not None:
        print(f"Discharge/rainfall date range: {flow_df.index.min()} to {flow_df.index.max()}")

    print("\nNote: the modelling reads Data/meteo/, not this folder. Copy the four files "
          "across if you have just re-downloaded them.")
