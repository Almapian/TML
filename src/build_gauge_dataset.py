"""Build the cleaned per-gauge tide datasets from the raw PLA deliverables.

Script form of `src/data.ipynb`. Reads the raw per-gauge .xlsx/.xls/.csv files under
`2_Deliverables/`, resolves which physical gauge installation each file belongs to,
normalises column names and datetime formats, drops unparseable rows and duplicate
timestamps, and writes one tidy CSV per gauge installation to `3_Cleaned/`.

**idSite is the primary key.** Some of the 13 folders contain data from *multiple*
physical gauges (09_Coryton has Thameshaven, No5 Jetty and Holehaven; 12_Margate has Pile
and Harbour), so each file is assigned an idSite from its filename rather than being
concatenated blindly. `00_gauge_metadata.csv` holds the location info and joins back on
idSite.

    python src/build_gauge_dataset.py
    # or run the `# %%` cells interactively in VS Code

Note: `2_Deliverables/` and `3_Cleaned/` are gitignored working data (~800MB and ~1GB) --
this script is only needed to regenerate them from the raw deliverables. The modelling and
EDA read the tracked copy in `Data/` instead.

CAUTION: writing over `3_Cleaned/*.csv` discards the `Observed_ODN` column that
`EDA/gauge_ts.ipynb` adds in a later pass (Chart Datum -> ODN), and the Southend
chatter/stuck patching from `EDA/utide_test.ipynb` lives in separate files that this
script does not touch. Re-run those passes after re-running this one, or set
WRITE_OUTPUTS = False to inspect the coverage report without overwriting anything.

Downstream usage of what this writes:
    ts   = pd.read_csv('3_Cleaned/09a_CORYTON_THAMESHAVEN.csv', parse_dates=['DateTime'])
    meta = pd.read_csv('3_Cleaned/00_gauge_metadata.csv')
    full = ts.merge(meta, on='idSite', how='left', suffixes=('', '_meta'))
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
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import paths

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.max_columns", None)

# %% ------------------------------------------------------------------ CONFIG
BASE_DIR = paths.RAW_DELIVERABLES_DIR      # 2_Deliverables/ -- the raw per-gauge files
OUTPUT_DIR = paths.CLEANED_DIR             # 3_Cleaned/ -- one CSV per gauge installation
HEATMAP_PATH = paths.OUTPUTS_DIR / "gauge_coverage_heatmap.png"

YEAR_START = 2004
YEAR_END = 2024

WRITE_OUTPUTS = True    # False to inspect the coverage report without writing CSVs

GAUGE_FOLDERS = [
    "01_Richmond", "02_Chelsea", "03_LondonBridgeTowerPier",
    "04_Charlton", "05_NorthWoolwichSilvertown", "06_Erith",
    "07_Tilbury", "08_GravesendDenton", "09_Coryton",
    "10_Southend", "11_ShiveringSand", "12_Margate", "13_Walton",
]

# Column alias sets -- all lowercase, all variants encountered in the deliverables. Only
# DateTime and Observed are read from the source files; idSite and SiteName are assigned
# from the filename via GAUGE_METADATA, so any internal id/name columns are ignored.
COLUMN_ALIASES = {
    "DateTime": {"datetime", "datatime", "date_time", "date time",
                 "date", "time", "timestamp", "date/time"},
    "Observed": {"observed", "obs", "water level", "water_level",
                 "level", "sea level", "sea_level", "height",
                 "water height", "water_height", "wl",
                 "tide level", "tide_level", "tidelevel"},
}
REQUIRED = {"DateTime", "Observed"}
OUTPUT_COLS = ["idSite", "SiteName", "DateTime", "Observed"]

# %% Gauge metadata --------------------------------------------------------
# Master record of every physical gauge installation across the 13 folders. Coordinates
# are OSGB36 (Easting/Northing) and WGS84 decimal degrees, converted from the PLA "Thames
# Tide Gauge National Grid References" table's degrees + decimal minutes. Chart datum is
# metres below ODN. Southend's longitude is printed as '00 43.43\'N' in the PLA reference
# but is clearly meant to be E (it is east of Greenwich); treated as +0.7238 here.
GAUGE_METADATA = pd.DataFrame([
    {"idSite": "01_RICHMOND", "GaugeFolder": "01_Richmond", "SiteName": "Richmond Lock", "PLA_Chart": 304, "Easting": 516980, "Northing": 175100, "Latitude": 51.4630, "Longitude": -0.3175, "ChartDatum_m_below_ODN": 0.61, "Status": "active"},
    {"idSite": "02_CHELSEA", "GaugeFolder": "02_Chelsea", "SiteName": "Chelsea Bridge", "PLA_Chart": 315, "Easting": 528580, "Northing": 177740, "Latitude": 51.4840, "Longitude": -0.1497, "ChartDatum_m_below_ODN": 2.44, "Status": "active"},
    {"idSite": "03_LONDONBRIDGE", "GaugeFolder": "03_LondonBridgeTowerPier", "SiteName": "London Bridge (Tower Pier)", "PLA_Chart": 318, "Easting": 533410, "Northing": 180480, "Latitude": 51.5075, "Longitude": -0.0790, "ChartDatum_m_below_ODN": 3.20, "Status": "active"},
    {"idSite": "04_CHARLTON", "GaugeFolder": "04_Charlton", "SiteName": "Charlton (Cory's)", "PLA_Chart": 323, "Easting": 540760, "Northing": 179220, "Latitude": 51.4943, "Longitude": 0.0263, "ChartDatum_m_below_ODN": 3.35, "Status": "active"},
    {"idSite": "05_NORTHWOOLWICH", "GaugeFolder": "05_NorthWoolwichSilvertown", "SiteName": "North Woolwich (Silvertown)", "PLA_Chart": 323, "Easting": 542130, "Northing": 179740, "Latitude": 51.4987, "Longitude": 0.0462, "ChartDatum_m_below_ODN": 3.35, "Status": "active"},
    {"idSite": "06_ERITH", "GaugeFolder": "06_Erith", "SiteName": "Erith (Erith Pier)", "PLA_Chart": 329, "Easting": 551940, "Northing": 178140, "Latitude": 51.4818, "Longitude": 0.1868, "ChartDatum_m_below_ODN": 3.28, "Status": "active"},
    {"idSite": "07_TILBURY", "GaugeFolder": "07_Tilbury", "SiteName": "Tilbury (NHCT)", "PLA_Chart": 335, "Easting": 562330, "Northing": 175670, "Latitude": 51.4567, "Longitude": 0.3350, "ChartDatum_m_below_ODN": 3.12, "Status": "active"},
    {"idSite": "08_GRAVESEND", "GaugeFolder": "08_GravesendDenton", "SiteName": "Gravesend (Denton)", "PLA_Chart": 337, "Easting": 566527, "Northing": 174441, "Latitude": 51.4445, "Longitude": 0.3948, "ChartDatum_m_below_ODN": 3.12, "Status": "active"},
    {"idSite": "09a_CORYTON_THAMESHAVEN", "GaugeFolder": "09_Coryton", "SiteName": "Coryton (Thameshaven)", "PLA_Chart": 340, "Easting": 573960, "Northing": 181390, "Latitude": 51.5047, "Longitude": 0.5052, "ChartDatum_m_below_ODN": 3.05, "Status": "active"},
    {"idSite": "09b_CORYTON_NO5JETTY", "GaugeFolder": "09_Coryton", "SiteName": "Coryton (No5 Jetty)", "PLA_Chart": 341, "Easting": 575197, "Northing": 181683, "Latitude": 51.5070, "Longitude": 0.5232, "ChartDatum_m_below_ODN": 3.05, "Status": "discontinued 2014-06"},
    {"idSite": "09c_CORYTON_HOLEHAVEN", "GaugeFolder": "09_Coryton", "SiteName": "Coryton (Holehaven)", "PLA_Chart": 341, "Easting": 577128, "Northing": 182304, "Latitude": 51.5118, "Longitude": 0.5512, "ChartDatum_m_below_ODN": 3.05, "Status": "discontinued 2017-12"},
    {"idSite": "10_SOUTHEND", "GaugeFolder": "10_Southend", "SiteName": "Southend Pier", "PLA_Chart": 342, "Easting": 589090, "Northing": 183020, "Latitude": 51.5145, "Longitude": 0.7238, "ChartDatum_m_below_ODN": 2.90, "Status": "active"},
    {"idSite": "11_SHIVERINGSANDS", "GaugeFolder": "11_ShiveringSand", "SiteName": "Shivering Sands (F1 Pile)", "PLA_Chart": 200, "Easting": 615828, "Northing": 179455, "Latitude": 51.4728, "Longitude": 1.1065, "ChartDatum_m_below_ODN": 2.68, "Status": "active"},
    {"idSite": "12a_MARGATE_HARBOUR", "GaugeFolder": "12_Margate", "SiteName": "Margate Harbour", "PLA_Chart": 200, "Easting": 635100, "Northing": 171290, "Latitude": 51.3920, "Longitude": 1.3780, "ChartDatum_m_below_ODN": 2.50, "Status": "active"},
    {"idSite": "12b_MARGATE_PILE", "GaugeFolder": "12_Margate", "SiteName": "Margate Pile", "PLA_Chart": 200, "Easting": 635147, "Northing": 171604, "Latitude": 51.3947, "Longitude": 1.3788, "ChartDatum_m_below_ODN": 2.50, "Status": "active (records ~2015-2022)"},
    {"idSite": "13_WALTON", "GaugeFolder": "13_Walton", "SiteName": "Walton on the Naze", "PLA_Chart": 200, "Easting": 626026, "Northing": 221163, "Latitude": 51.8433, "Longitude": 1.2800, "ChartDatum_m_below_ODN": 2.16, "Status": "active"},
]).set_index("idSite", drop=False)

# File-pattern -> idSite rules. Only the two multi-gauge folders need explicit rules;
# every other folder uses its single default idSite (auto-derived below). Rules are
# checked in order, first match wins, patterns are case-insensitive.
FILE_TO_IDSITE_RULES = {
    "09_Coryton": [
        # No5 Jetty -- discontinued Jun 2014. Files use 'CORYTON_P4' naming; data in that
        # folder runs through 2014/2015, matching the cutoff.
        (re.compile(r"CORYTON_P4", re.IGNORECASE), "09b_CORYTON_NO5JETTY"),
        # Holehaven -- active ~2015 to Dec 2017. Files use 'H._Haven' or 'Holehaven'.
        (re.compile(r"H[._ ]*Haven|Holehaven", re.IGNORECASE), "09c_CORYTON_HOLEHAVEN"),
        # Thameshaven -- active 2018 onwards.
        (re.compile(r"Thameshaven", re.IGNORECASE), "09a_CORYTON_THAMESHAVEN"),
    ],
    "12_Margate": [
        # Margate Pile -- fills the Harbour gap years (~2015-2022).
        (re.compile(r"Margate[_ ]Pile", re.IGNORECASE), "12b_MARGATE_PILE"),
        # Margate Harbour -- explicit 'Harb' naming.
        (re.compile(r"Margate[_ ]Harb", re.IGNORECASE), "12a_MARGATE_HARBOUR"),
        # Legacy 'MARGATE 1998.xls' / 'MARGATE 1999.xls' -- predate Pile, assume Harbour.
        (re.compile(r"^MARGATE\s+\d{4}", re.IGNORECASE), "12a_MARGATE_HARBOUR"),
        # Generic 'Tides_Margate_Pressure_...' (no Pile/Harb qualifier), 2023+. Assumed
        # Harbour (Pile records seem to end in 2022); a warning is printed at load.
        (re.compile(r"Margate[_ ]Pressure", re.IGNORECASE), "12a_MARGATE_HARBOUR"),
    ],
}

_folder_counts = GAUGE_METADATA["GaugeFolder"].value_counts()
FOLDER_DEFAULT_IDSITE = {
    folder: GAUGE_METADATA.loc[GAUGE_METADATA.GaugeFolder == folder, "idSite"].iloc[0]
    for folder in _folder_counts[_folder_counts == 1].index
}

print(f"Metadata: {len(GAUGE_METADATA)} gauges across {GAUGE_METADATA.GaugeFolder.nunique()} folders")
print(f"Single-gauge folders: {len(FOLDER_DEFAULT_IDSITE)}")
print(f"Multi-gauge folders (rule-based): {list(FILE_TO_IDSITE_RULES.keys())}\n")
print(GAUGE_METADATA[["SiteName", "Latitude", "Longitude", "ChartDatum_m_below_ODN", "Status"]].to_string())

# Audit log -- populated by load_file (dropped DateTime rows) and load_folder (duplicate
# timestamps), inspected after the load.
LOAD_AUDIT = {"dropped_datetime": [], "duplicates": []}


# %% Helper functions -------------------------------------------------------
def extract_years(fname):
    """Pull the first two 20XX years from a filename."""
    yrs = re.findall(r"(20\d{2})\d{4}", fname)
    if len(yrs) >= 2:
        return int(yrs[0]), int(yrs[1])
    if len(yrs) == 1:
        return int(yrs[0]), int(yrs[0])
    return None, None


def resolve_idsite(fname, gauge_folder):
    """Map a filename to the correct idSite.

    For multi-gauge folders it applies the regex rules in FILE_TO_IDSITE_RULES; otherwise
    it returns the folder's single default idSite. Returns None (with a warning) if no
    rule matches.
    """
    if gauge_folder in FILE_TO_IDSITE_RULES:
        for pattern, idsite in FILE_TO_IDSITE_RULES[gauge_folder]:
            if pattern.search(fname):
                return idsite
        print(f"    [WARN] {fname}: no idSite rule matched in {gauge_folder}; file skipped")
        return None

    if gauge_folder in FOLDER_DEFAULT_IDSITE:
        return FOLDER_DEFAULT_IDSITE[gauge_folder]

    print(f"    [WARN] {gauge_folder}: not in metadata; {fname} skipped")
    return None


def parse_datetimes(series):
    """Parse a DateTime column with per-file format sniffing.

    Some files store DateTime as Excel datetime cells, which pandas converts to ISO
    strings ('2016-01-13 20:30:00') regardless of the dd/mm/yyyy format shown in Excel.
    Others store it as literal text ('01/01/2023 00:00'), which needs dayfirst=True.
    Applying dayfirst=True to an ISO string silently swaps day and month (2016-01-12
    becomes 1 Dec instead of 12 Jan) and fails outright on days > 12 -- so the first
    non-null value is sniffed and the right parser picked per file.
    """
    sample = series.dropna().astype(str)
    if sample.empty:
        return pd.to_datetime(series, errors="coerce")
    iso_like = bool(re.match(r"^\d{4}[-/]", sample.iloc[0].strip()))
    return pd.to_datetime(series, dayfirst=not iso_like, errors="coerce")


def match_columns(df_columns):
    """Map canonical column names to actual file column names (case-insensitive)."""
    lower_to_actual = {c.strip().lower(): c for c in df_columns}
    found, missing = {}, []
    for canonical, aliases in COLUMN_ALIASES.items():
        match = next((lower_to_actual[a] for a in aliases if a in lower_to_actual), None)
        if match:
            found[canonical] = match
        else:
            missing.append(canonical)
    return found, missing


def load_file(fp, idsite):
    """Load one xlsx/csv and stamp it with the filename-derived idSite (and matching
    SiteName). Any internal id/name columns inside the source file are ignored -- the
    filename is the source of truth for which gauge a file belongs to.

    Returns a DataFrame of idSite | SiteName | DateTime | Observed, or None if a required
    column is missing.
    """
    try:
        ext = fp.suffix.lower()
        df = pd.read_excel(fp, dtype=str) if ext in (".xlsx", ".xls") else pd.read_csv(fp, dtype=str)

        found, missing = match_columns(df.columns)
        if missing:
            print(f"    [SKIP] {fp.name}: required columns not found: {missing}  |  "
                  f"file has: {list(df.columns)}")
            return None

        out = pd.DataFrame({
            "idSite": idsite,
            "SiteName": GAUGE_METADATA.loc[idsite, "SiteName"],
            "DateTime": df[found["DateTime"]].values,
            "Observed": df[found["Observed"]].values,
        })

        # keep the raw DateTime strings so the audit can show what the unparseable rows
        # actually looked like
        raw_dt = out["DateTime"].copy()
        out["DateTime"] = parse_datetimes(out["DateTime"])
        out["Observed"] = pd.to_numeric(out["Observed"], errors="coerce")

        bad_mask = out["DateTime"].isna()
        n_bad = int(bad_mask.sum())
        if n_bad:
            print(f"    [INFO] {fp.name}: {n_bad:,} rows dropped (unparseable DateTime)")
            LOAD_AUDIT["dropped_datetime"].append({
                "file": fp.name, "idsite": idsite, "n_dropped": n_bad,
                "sample": pd.DataFrame({
                    "row_index": raw_dt[bad_mask].index[:10],
                    "raw_datetime": raw_dt[bad_mask].head(10).values,
                    "observed": out.loc[bad_mask, "Observed"].head(10).values,
                }),
            })
        out = out.dropna(subset=["DateTime"])
        return out[OUTPUT_COLS]

    except Exception as e:
        print(f"    [ERROR] {fp.name}: {e}")
        return None


def detect_resolution_minutes(df, n_sample=2000):
    """Infer the dominant time step in minutes (median consecutive diff), snapped to the
    nearest standard interval."""
    ts = df["DateTime"].sort_values().head(n_sample)
    diffs = ts.diff().dropna().dt.total_seconds() / 60
    med = diffs.median()
    for nice in [1, 5, 10, 15, 30, 60, 120]:
        if abs(med - nice) < nice * 0.3:
            return nice
    return max(1, round(med))


def expected_records_span(df, res_mins):
    """Expected record count from the actual time span -- no hardcoded annual count."""
    span = (df["DateTime"].max() - df["DateTime"].min()).total_seconds() / 60
    return max(1, round(span / res_mins) + 1)


def expected_records_full_year(year, res_mins):
    """Expected record count for a full calendar year (leap-aware)."""
    days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    return round(days * 24 * 60 / res_mins)


def load_folder(folder):
    """Load every usable file in a folder, returned as a dict keyed by idSite. Files from
    the same folder but different physical gauges are kept separate (09_Coryton yields up
    to three entries)."""
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in (".xlsx", ".xls", ".csv"))
    chunks_by_idsite = {}

    for f in files:
        idsite = resolve_idsite(f.name, folder.name)
        if idsite is None:
            continue

        y0, y1 = extract_years(f.name)
        if y0 is not None and (y1 < YEAR_START or y0 > YEAR_END):
            continue

        df = load_file(f, idsite)
        if df is None or df.empty:
            continue
        chunks_by_idsite.setdefault(idsite, []).append(df)

    out = {}
    for idsite, chunks in chunks_by_idsite.items():
        combined = pd.concat(chunks, ignore_index=True)
        combined = combined[(combined["DateTime"].dt.year >= YEAR_START)
                            & (combined["DateTime"].dt.year <= YEAR_END)]

        # keep=False marks ALL rows sharing a timestamp (kept and dropped alike), so the
        # audit can show which values were in conflict
        dup_all = combined.duplicated("DateTime", keep=False)
        n_dup = int(combined.duplicated("DateTime").sum())
        if n_dup:
            # conflict stats over EVERY duplicated timestamp, not just the sample:
            # n_dup_timestamps = unique timestamps appearing more than once,
            # n_conflicting = those whose rows disagree on Observed (kept row is arbitrary)
            dup_groups = combined[dup_all].groupby("DateTime")["Observed"]
            LOAD_AUDIT["duplicates"].append({
                "idsite": idsite, "n_removed": n_dup,
                "n_dup_timestamps": int(dup_groups.ngroups),
                "n_conflicting": int((dup_groups.nunique(dropna=False) > 1).sum()),
                "sample": combined[dup_all].sort_values("DateTime").head(20).copy(),
            })
            print(f"    [INFO] {idsite}: {n_dup:,} duplicate timestamps removed")
            combined = combined.drop_duplicates("DateTime")

        if combined.empty:
            continue
        out[idsite] = combined.sort_values("DateTime").reset_index(drop=True)
    return out


# %% Run the load -----------------------------------------------------------
assert BASE_DIR.is_dir(), (
    f"raw deliverables not found at {BASE_DIR}. This script needs 2_Deliverables/, which "
    f"is gitignored working data -- the modelling and EDA use Data/ instead.")

all_gauges = {}   # keyed by idSite, so Coryton/Margate sub-gauges stay separate
print(f"\nLoading Thames tidal data {YEAR_START}-{YEAR_END}\n")

for gf in GAUGE_FOLDERS:
    folder = BASE_DIR / gf
    if not folder.exists():
        print(f"  x NOT FOUND: {gf}")
        continue

    print(f"  -> {gf}")
    gauge_dfs = load_folder(folder)
    if not gauge_dfs:
        print("     no usable data")
        continue

    for idsite, df in gauge_dfs.items():
        res = detect_resolution_minutes(df)
        nan_pct = 100 * df["Observed"].isna().mean()
        print(f"     [{idsite}] {len(df):>9,} rows  |  {df.DateTime.min().date()} -> "
              f"{df.DateTime.max().date()}  |  res: {res} min  |  Observed NaN: {nan_pct:.1f}%")
        all_gauges[idsite] = df

print(f"\nDone. {len(all_gauges)} distinct gauge time-series loaded.")

# %% Inspect drops and duplicates ------------------------------------------
# Dropped DateTime rows should be unit/header rows (e.g. 'UTC') or other non-data junk; if
# real timestamps appear, the parser is missing a format. Duplicate timestamps should
# mostly be exact duplicates caused by file overlaps; where the Observed values differ,
# the kept row is arbitrary (first occurrence wins).
print("\n" + "=" * 72)
print("  ROWS DROPPED -- unparseable DateTime")
print("=" * 72)
if not LOAD_AUDIT["dropped_datetime"]:
    print("\n  (none)")
else:
    total_dropped = sum(e["n_dropped"] for e in LOAD_AUDIT["dropped_datetime"])
    print(f"\n  {len(LOAD_AUDIT['dropped_datetime'])} files affected, "
          f"{total_dropped:,} rows dropped in total.\n")
    for entry in LOAD_AUDIT["dropped_datetime"]:
        print(f"  -> {entry['file']}")
        print(f"     idSite: {entry['idsite']}  |  rows dropped: {entry['n_dropped']:,}")
        print(entry["sample"].to_string())

print("\n" + "=" * 72)
print("  DUPLICATE TIMESTAMPS REMOVED")
print("=" * 72)
if not LOAD_AUDIT["duplicates"]:
    print("\n  (none)")
else:
    for entry in LOAD_AUDIT["duplicates"]:
        n_conf, n_ts = entry["n_conflicting"], entry["n_dup_timestamps"]
        verdict = (f"all {n_ts:,} duplicated timestamps have identical Observed values"
                   if n_conf == 0 else
                   f"!! {n_conf:,} of {n_ts:,} duplicated timestamps have CONFLICTING "
                   f"Observed values !!")
        print(f"\n  -> {entry['idsite']}")
        print(f"     rows removed: {entry['n_removed']:,}")
        print(f"     verdict: {verdict}")
        print(entry["sample"].to_string())

# %% Coverage report --------------------------------------------------------
# One row per gauge installation, using span-based expected counts: how complete is the
# data we DO have, over the span it covers?
rows = []
for idsite in GAUGE_METADATA["idSite"]:
    df = all_gauges.get(idsite)
    meta = GAUGE_METADATA.loc[idsite]
    if df is None or df.empty:
        rows.append({"idSite": idsite, "SiteName": meta["SiteName"], "Start": "-", "End": "-",
                     "Records": 0, "Resolution": "-", "Expected": 0, "Missing%": 100.0,
                     "Observed NaN%": "-", "Status": meta["Status"]})
        continue

    res = detect_resolution_minutes(df)
    exp = expected_records_span(df, res)
    rows.append({
        "idSite": idsite, "SiteName": meta["SiteName"],
        "Start": str(df.DateTime.min().date()), "End": str(df.DateTime.max().date()),
        "Records": len(df), "Resolution": f"{res} min", "Expected": exp,
        "Missing%": max(round(100 * (1 - len(df) / exp), 1), 0.0),
        "Observed NaN%": f"{round(100 * df['Observed'].isna().mean(), 1)}%",
        "Status": meta["Status"],
    })

coverage = pd.DataFrame(rows)
print("\nCoverage report:")
print(coverage.to_string(index=False))

# %% Per-year completeness heatmap ------------------------------------------
# Full-year expected counts here, so a gauge with data only through July 2024 shows ~58%,
# not 100%. Gauge swaps (Coryton No5 Jetty -> Holehaven -> Thameshaven) appear as clean
# handoffs in the matrix.
all_idsites = GAUGE_METADATA["idSite"].tolist()
years = list(range(YEAR_START, YEAR_END + 1))
matrix = np.full((len(all_idsites), len(years)), np.nan)

for i, idsite in enumerate(all_idsites):
    df = all_gauges.get(idsite)
    if df is None or df.empty:
        continue
    res = detect_resolution_minutes(df)
    for j, yr in enumerate(years):
        yr_df = df[df.DateTime.dt.year == yr]
        if yr_df.empty:
            continue
        matrix[i, j] = min(len(yr_df) / expected_records_full_year(yr, res) * 100, 100)

fig, ax = plt.subplots(figsize=(17, 6.5))
cmap = plt.cm.RdYlGn.copy()
cmap.set_bad("lightgrey")   # grey = no data at all for that year
im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=0, vmax=100)
plt.colorbar(im, ax=ax, label="% of full calendar year covered", shrink=0.8)

ax.set_xticks(range(len(years)))
ax.set_xticklabels(years, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(all_idsites)))
ax.set_yticklabels([f"{ids}  --  {GAUGE_METADATA.loc[ids, 'SiteName']}" for ids in all_idsites],
                   fontsize=8)
for i in range(len(all_idsites)):
    for j in range(len(years)):
        val = matrix[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7,
                    color="black" if val > 35 else "white")
ax.set_title("Data completeness by gauge & year (% of full calendar year covered)",
             fontsize=11, pad=10)
plt.tight_layout()
paths.ensure_dir(HEATMAP_PATH.parent)
plt.savefig(HEATMAP_PATH, dpi=150, bbox_inches="tight")
print(f"\nsaved {HEATMAP_PATH}")
plt.show()

# %% Save outputs -----------------------------------------------------------
if WRITE_OUTPUTS:
    paths.ensure_dir(OUTPUT_DIR)
    print("\nSaving per-gauge time series...\n")
    for idsite, df in all_gauges.items():
        out_path = OUTPUT_DIR / f"{idsite}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path.name:40s} ({len(df):>10,} rows)")

    meta_path = OUTPUT_DIR / "00_gauge_metadata.csv"
    GAUGE_METADATA.to_csv(meta_path, index=False)
    print(f"\n  Saved metadata: {meta_path.name}  ({len(GAUGE_METADATA)} gauges)")
    print("\nAll clean files saved.")
else:
    print("\nWRITE_OUTPUTS is False -- nothing written.")
