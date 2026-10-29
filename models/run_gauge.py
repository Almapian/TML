"""Run a modelling pipeline for one or more gauges, unattended, retrying on failure.

    python models/run_gauge.py                                   # main pipeline, GAUGE below
    python models/run_gauge.py --pipeline dense --gauge richmond --gauge southend

Each step runs as its own process, with TIDAL_GAUGE / TIDAL_MAX_HORIZON_H passed to every
script. Two pipelines:

    main   data cleaning (only if the gauge's cleaned CSV doesn't exist yet), then
           train_stage2 -> plot_stage2 -> train_stage3 -> plot_stage3 -> train_stage1 -> plot_stage1
    dense  train_dense -> plot_dense (continuous 0-72h forecasts; see models/train_dense.py)

Gauges given with --gauge run one after another, in the order given; with none, the CONFIG
GAUGE is used. Stage 2 goes first in the main pipeline because it is the core result; each
stage is plotted as soon as it has trained, so usable figures appear before the run is over.

Retries. If a step exits with an error it is re-run, up to MAX_RETRIES times. A retry
never repeats finished work: models whose checkpoint exists are loaded, not retrained,
and a model that was mid-training resumes from its last completed epoch (every epoch is
saved to outputs/_resume/). A step that fails MAX_RETRIES + 1 times in a row stops the run,
since that is a bug rather than a crash, and its log says why.

Re-launching. Completed steps are recorded in logs/<gauge>/run_state.json (main) or
logs/<gauge>/dense/run_state.json (dense) and skipped on the next launch, so after a
reboot (or a stop) just run the same command again. Delete that file, or pass --redo, to
redo every step.

Logs. Each step's output goes both to the console and to that directory's <step>.log
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
import argparse
import ctypes
import json
import os
import subprocess
import time
from datetime import datetime

from utils import paths
from utils.models import config as C

# %% ------------------------------------------------------------------ CONFIG
GAUGE = 'richmond'        # default gauge when no --gauge is given; see config.GAUGES
MAX_HORIZON_H = 72        # plot horizons up to this many hours; None for all
MAX_RETRIES = 3           # re-runs of a failing step before the run stops
RETRY_WAIT_S = 60         # pause before a retry
SKIP_COMPLETED = True     # skip steps the run_state.json marks as done

# steps per pipeline, in order: (name, script relative to the repo root)
PIPELINES = {
    'main': [
        ('train_stage2', 'models/train_stage2.py'),
        ('plot_stage2', 'models/plot_stage2.py'),
        ('train_stage3', 'models/train_stage3.py'),
        ('plot_stage3', 'models/plot_stage3.py'),
        ('train_stage1', 'models/train_stage1.py'),
        ('plot_stage1', 'models/plot_stage1.py'),
    ],
    'dense': [
        ('train_dense', 'models/train_dense.py'),
        ('plot_dense', 'models/plot_dense.py'),
    ],
}
# per gauge: the script that writes its cleaned CSV, run first if that CSV is missing
CLEANING_SCRIPTS = {'richmond': 'src/clean_richmond.py'}

PYTHON = sys.executable   # the interpreter running this script (the geospatial env)


# %% Helpers ------------------------------------------------------------------
def keep_awake(on):
    """Ask Windows not to idle-sleep while the run is going (no-op elsewhere)."""
    if os.name != 'nt':
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def log_dir(gauge, pipeline):
    """logs/<gauge>/ for the main pipeline (as before), logs/<gauge>/<pipeline>/ otherwise."""
    base = REPO_ROOT / 'logs' / gauge
    return paths.ensure_dir(base if pipeline == 'main' else base / pipeline)


def load_state(state_path, skip_completed):
    if skip_completed and state_path.exists():
        return json.loads(state_path.read_text(encoding='utf-8'))
    return {'completed': []}


def save_state(state_path, state):
    state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')


def run_step(gauge, name, script, logs):
    """Run one script as a child process, streaming its output to console + log.
    Returns the exit code."""
    env = dict(os.environ, TIDAL_GAUGE=gauge,
               TIDAL_MAX_HORIZON_H='none' if MAX_HORIZON_H is None else str(MAX_HORIZON_H),
               PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
    log_path = logs / f'{name}.log'
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


def run_with_retries(gauge, name, script, logs):
    for attempt in range(1, MAX_RETRIES + 2):
        t0 = time.time()
        print(f"\n>>> [{gauge}] {name} (attempt {attempt}/{MAX_RETRIES + 1})", flush=True)
        code = run_step(gauge, name, script, logs)
        minutes = (time.time() - t0) / 60
        if code == 0:
            print(f">>> [{gauge}] {name} finished in {minutes:.1f} min", flush=True)
            return True
        print(f">>> [{gauge}] {name} FAILED (exit {code}) after {minutes:.1f} min -- "
              f"see {logs / (name + '.log')}", flush=True)
        if attempt <= MAX_RETRIES:
            print(f">>> retrying in {RETRY_WAIT_S}s; finished models load from their checkpoints and "
                  f"an interrupted one resumes from its last completed epoch", flush=True)
            time.sleep(RETRY_WAIT_S)
    return False


def run_gauge(gauge, pipeline, skip_completed):
    """Every step of `pipeline` for one gauge. Returns False if a step gave up."""
    info = C.gauge_info(gauge)
    logs = log_dir(gauge, pipeline)
    state_path = logs / 'run_state.json'
    print(f"\n===== gauge: {info['label']}  |  pipeline: {pipeline}  |  logs: {logs}")

    steps = list(PIPELINES[pipeline])
    if pipeline == 'main' and gauge in CLEANING_SCRIPTS and not info['csv'].exists():
        steps.insert(0, ('clean_data', CLEANING_SCRIPTS[gauge]))

    state = load_state(state_path, skip_completed)
    for name, script in steps:
        if name in state['completed']:
            print(f">>> [{gauge}] {name}: already completed in an earlier launch -- skipping")
            continue
        if not run_with_retries(gauge, name, script, logs):
            print(f"\n>>> run stopped at {gauge} / {name}: it failed {MAX_RETRIES + 1} times in a row. "
                  f"Fix the error in its log, then re-launch -- completed steps are skipped.")
            return False
        state['completed'].append(name)
        save_state(state_path, state)
    return True


# %% Run ----------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--pipeline', choices=list(PIPELINES), default='main')
    parser.add_argument('--gauge', action='append', choices=list(C.GAUGES),
                        help='repeatable; runs in the order given (default: CONFIG GAUGE)')
    parser.add_argument('--redo', action='store_true', help='ignore run_state.json and redo every step')
    args, _ = parser.parse_known_args()   # tolerate IPython/VS Code's own arguments
    gauges = args.gauge or [GAUGE]

    print(f"pipeline: {args.pipeline}  |  gauges: {', '.join(gauges)}  |  plot cap: {MAX_HORIZON_H}h  "
          f"|  python: {PYTHON}")
    keep_awake(True)
    run_start = time.time()
    try:
        for gauge in gauges:
            if not run_gauge(gauge, args.pipeline, SKIP_COMPLETED and not args.redo):
                sys.exit(1)
    finally:
        keep_awake(False)

    print(f"\n>>> all steps done for {', '.join(gauges)} in {(time.time() - run_start) / 3600:.1f} h. "
          f"Figures: {paths.REPORT_IMAGES_DIR}")
