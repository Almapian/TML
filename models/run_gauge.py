"""Run the whole modelling pipeline for one gauge, unattended, retrying on failure.

    python models/run_gauge.py

Edit the CONFIG block (GAUGE, MAX_HORIZON_H) and launch it once from a terminal. It runs,
each as its own process, with GAUGE / MAX_HORIZON_H passed to every script:

    data cleaning (only if the gauge's cleaned CSV doesn't exist yet)
    train_stage2 -> plot_stage2 -> train_stage3 -> plot_stage3 -> train_stage1 -> plot_stage1

Stage 2 goes first because it is the core result; each stage is plotted as soon as it
has trained, so usable figures appear before the whole run is over.

Retries. If a step exits with an error it is re-run, up to MAX_RETRIES times. A retry
never repeats finished work: models whose checkpoint exists are loaded, not retrained,
and a model that was mid-training resumes from its last completed epoch (every epoch is
saved to outputs/_resume/). A step that fails MAX_RETRIES + 1 times in a row stops the run,
since that is a bug rather than a crash, and its log says why.

Re-launching. Completed steps are recorded in logs/<gauge>/run_state.json and skipped on
the next launch, so after a reboot (or a stop) just run this again. Delete that file, or
set SKIP_COMPLETED = False, to redo every step.

Logs. Each step's output goes both to the console and to logs/<gauge>/<step>.log
(appended, with a header per attempt).

Sleep. While running, Windows is asked not to sleep (SetThreadExecutionState), which
covers the idle-sleep timer. Closing the laptop lid can still sleep it, depending on the
power settings, so leave it open; and keep it plugged in.
"""
# %% Bootstrap
import pathlib
import sys

_start = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / '_')).resolve()
REPO_ROOT = next(p for p in _start.parents if (p / 'utils' / 'paths.py').exists())
sys.path.insert(0, str(REPO_ROOT))

# %% Imports
import ctypes
import json
import os
import subprocess
import time
from datetime import datetime

from utils import paths
from utils.models import config as C

# %% ------------------------------------------------------------------ CONFIG
GAUGE = 'richmond'        # see config.GAUGES
MAX_HORIZON_H = 72        # plot horizons up to this many hours; None for all
MAX_RETRIES = 3           # re-runs of a failing step before the run stops
RETRY_WAIT_S = 60         # pause before a retry
SKIP_COMPLETED = True     # skip steps logs/<gauge>/run_state.json marks as done

# steps, in order: (name, script relative to the repo root)
STEPS = [
    ('train_stage2', 'models/train_stage2.py'),
    ('plot_stage2', 'models/plot_stage2.py'),
    ('train_stage3', 'models/train_stage3.py'),
    ('plot_stage3', 'models/plot_stage3.py'),
    ('train_stage1', 'models/train_stage1.py'),
    ('plot_stage1', 'models/plot_stage1.py'),
]
# per gauge: the script that writes its cleaned CSV, run first if that CSV is missing
CLEANING_SCRIPTS = {'richmond': 'src/clean_richmond.py'}

PYTHON = sys.executable   # the interpreter running this script (the geospatial env)
LOG_DIR = paths.ensure_dir(REPO_ROOT / 'logs' / GAUGE)
STATE_PATH = LOG_DIR / 'run_state.json'


# %% Helpers ------------------------------------------------------------------
def keep_awake(on):
    """Ask Windows not to idle-sleep while the run is going (no-op elsewhere)."""
    if os.name != 'nt':
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def load_state():
    if SKIP_COMPLETED and STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding='utf-8'))
    return {'completed': []}


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding='utf-8')


def run_step(name, script):
    """Run one script as a child process, streaming its output to console + log.
    Returns the exit code."""
    env = dict(os.environ, TIDAL_GAUGE=GAUGE,
               TIDAL_MAX_HORIZON_H='none' if MAX_HORIZON_H is None else str(MAX_HORIZON_H),
               PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
    log_path = LOG_DIR / f'{name}.log'
    with open(log_path, 'a', encoding='utf-8') as log:
        log.write(f"\n===== {name}: attempt started {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        proc = subprocess.Popen([PYTHON, str(REPO_ROOT / script)], cwd=REPO_ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace')
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        code = proc.wait()
        log.write(f"===== {name}: exit code {code} at {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
    return code


def run_with_retries(name, script):
    for attempt in range(1, MAX_RETRIES + 2):
        t0 = time.time()
        print(f"\n>>> [{GAUGE}] {name} (attempt {attempt}/{MAX_RETRIES + 1})", flush=True)
        code = run_step(name, script)
        minutes = (time.time() - t0) / 60
        if code == 0:
            print(f">>> [{GAUGE}] {name} finished in {minutes:.1f} min", flush=True)
            return True
        print(f">>> [{GAUGE}] {name} FAILED (exit {code}) after {minutes:.1f} min -- "
              f"see {LOG_DIR / (name + '.log')}", flush=True)
        if attempt <= MAX_RETRIES:
            print(f">>> retrying in {RETRY_WAIT_S}s; finished models load from their checkpoints and "
                  f"an interrupted one resumes from its last completed epoch", flush=True)
            time.sleep(RETRY_WAIT_S)
    return False


# %% Run ----------------------------------------------------------------------
if __name__ == '__main__':
    info = C.gauge_info(GAUGE)
    print(f"gauge: {info['label']}  |  plot cap: {MAX_HORIZON_H}h  |  python: {PYTHON}")
    print(f"logs: {LOG_DIR}")

    steps = list(STEPS)
    if GAUGE in CLEANING_SCRIPTS and not info['csv'].exists():
        steps.insert(0, ('clean_data', CLEANING_SCRIPTS[GAUGE]))

    state = load_state()
    keep_awake(True)
    run_start = time.time()
    try:
        for name, script in steps:
            if name in state['completed']:
                print(f">>> [{GAUGE}] {name}: already completed in an earlier launch -- skipping")
                continue
            if not run_with_retries(name, script):
                print(f"\n>>> run stopped at {name}: it failed {MAX_RETRIES + 1} times in a row. "
                      f"Fix the error in its log, then re-launch -- completed steps are skipped.")
                sys.exit(1)
            state['completed'].append(name)
            save_state(state)
    finally:
        keep_awake(False)

    print(f"\n>>> [{GAUGE}] all steps done in {(time.time() - run_start) / 3600:.1f} h. "
          f"Figures: {paths.REPORT_IMAGES_DIR}")
