"""Canonical filesystem locations for this repository.

Every script under `models/` and `src/` imports its paths from here rather than
re-deriving `../` chains, so moving a script between directories can't silently
repoint it at the wrong data.

    from utils import paths
    df = pd.read_csv(paths.SOUTHEND_CSV, parse_dates=['DateTime'])

The notebooks under `models/notebooks/` deliberately do NOT use this module -- they
keep their own Colab/Drive path logic, see the note in the repository plan.
"""
from pathlib import Path

# utils/paths.py -> utils/ -> repository root
REPO_ROOT = Path(__file__).resolve().parents[1]

# --- tracked input data (a self-contained copy of what the pipeline actually consumes) ---
DATA_DIR = REPO_ROOT / "Data"
SOUTHEND_CSV = DATA_DIR / "southend_pier_data.csv"
RICHMOND_CSV = DATA_DIR / "richmond_lock_data.csv"   # written by src/clean_richmond.py
METEO_DIR = DATA_DIR / "meteo"
PRESSURE_CSV = METEO_DIR / "pressure_all_gauges.csv"
WIND_CSV = METEO_DIR / "wind_all_gauges.csv"
RAINFALL_CSV = METEO_DIR / "rainfall_kingston_catchment.csv"
DISCHARGE_CSV = METEO_DIR / "river_discharge_kingston.csv"

# --- untracked working data (raw deliverables and the full multi-gauge cleaned set) ---
RAW_DELIVERABLES_DIR = REPO_ROOT / "2_Deliverables"
CLEANED_DIR = REPO_ROOT / "3_Cleaned"
CLEANED_METEO_DIR = CLEANED_DIR / "meteo"

# --- trained model weights: small (~1.5MB total) and tracked, so the repo is reproducible
# without re-running the training that produced them ---
CHECKPOINTS_DIR = REPO_ROOT / "models" / "checkpoints"

# --- outputs: `outputs/` is scratch (gitignored), `report_images/` is the tracked deliverable ---
OUTPUTS_DIR = REPO_ROOT / "outputs"
REPORT_IMAGES_DIR = REPO_ROOT / "report_images"

STAGE1_OUTPUT_DIR = OUTPUTS_DIR / "initial_outputs"
STAGE2_OUTPUT_DIR = OUTPUTS_DIR / "stage2_outputs"
STAGE3_OUTPUT_DIR = OUTPUTS_DIR / "stage3_outputs"

_STAGE_OUTPUT_NAMES = {1: "initial_outputs", 2: "stage2_outputs", 3: "stage3_outputs"}


def stage_output_dir(stage, gauge="southend"):
    """Tables for one stage and gauge. Southend keeps the original locations; any other
    gauge gets its own subtree, so a second gauge can never overwrite Southend's results."""
    if gauge == "southend":
        return OUTPUTS_DIR / _STAGE_OUTPUT_NAMES[stage]
    return OUTPUTS_DIR / gauge / _STAGE_OUTPUT_NAMES[stage]


def checkpoint_dir(gauge="southend"):
    """Model weights for one gauge (Southend's stay directly under models/checkpoints/)."""
    return CHECKPOINTS_DIR if gauge == "southend" else CHECKPOINTS_DIR / gauge


def dense_output_dir(gauge="southend"):
    """Tables of the dense (continuous 0-72h) pipeline, apart from the main stages'."""
    return (OUTPUTS_DIR if gauge == "southend" else OUTPUTS_DIR / gauge) / "dense"


def dense_checkpoint_dir(gauge="southend"):
    """Weights of the dense pipeline, so they never collide with the main checkpoints."""
    return checkpoint_dir(gauge) / "dense"


def ensure_dir(path):
    """Create `path` (and parents) if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
