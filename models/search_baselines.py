"""Hyperparameter search for the classical baselines, on the validation split.

One Optuna TPE study per method, track and gauge, minimising masked MSE averaged equally over
the 73 dense lead times, the criterion the dense neural models trained on. AR, GP and ridge
also get a "parity" study capped at the neural models' 16-hour history.

The series is cut at VAL_END before anything else, so test data cannot be reached from here.
Studies live in a SQLite file, so a restarted run carries on from its finished trials.

    python models/search_baselines.py       # after models/identify_baselines.py

Each study below is the extended round (suffix ROUND), over the ranges widened after the
first search's edge winners, and starts from that round's winner. The first round's studies
and CSVs are kept as they were.

Outputs (outputs[/<gauge>]/baselines/search/): search_<method>_<track>[_parity]<ROUND>.csv,
edge_winners.csv (winners on a range edge, which must be extended), search_shortfall.csv
(studies stopped by the time budget). Frozen winners: models/checkpoints[/<gauge>]/baselines/
best_configs.json.
"""
# %% Bootstrap
import pathlib
import sys

_start = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / '_')).resolve()
REPO_ROOT = next(p for p in _start.parents if (p / 'utils' / 'paths.py').exists())
sys.path.insert(0, str(REPO_ROOT))

# %% Imports
import json
import time
import warnings

import numpy as np
import optuna
import pandas as pd

from utils import paths
from utils.models import baselines as B
from utils.models import config as C

# %% ------------------------------------------------------------------ CONFIG
GAUGE = C.from_env('TIDAL_GAUGE', 'southend')    # run_gauge.py sets it
SMOKE = C.from_env('TIDAL_SMOKE', '0').strip().lower() not in ('', '0', 'false')
METHODS = C.from_env('TIDAL_BASELINE_METHODS', ','.join(C.BASELINE_METHODS)).split(',')

ANCHOR_STRIDE = 288 if SMOKE else C.BASELINE_SEARCH_ANCHOR_STRIDE
TRIALS = {m: (2 if SMOKE else n) for m, n in C.BASELINE_TRIALS.items()}
PARITY_TRIALS = 2 if SMOKE else C.BASELINE_PARITY_TRIALS
TIMEOUT_S = 300 if SMOKE else C.BASELINE_TIMEOUT_S
MAX_RIDGE_LAGS = 2400          # above this many lags X'X stops being cheap: spacing widens
PARITY_HISTORY = C.LOOKBACK
PARITY_ORDERS = [16, 32, 48, 64, 96]
ROUND = C.from_env('TIDAL_BASELINE_ROUND', '_x')

suffix = '_smoke' if SMOKE else ''
BASE = pathlib.Path(str(paths.baseline_output_dir(GAUGE)) + suffix)
OUT = paths.ensure_dir(BASE / 'search')
RANGES = json.loads((BASE / 'identification' / 'identified_ranges.json').read_text(encoding='utf-8'))
CONFIG_PATH = paths.baseline_config_path(GAUGE) if not SMOKE else OUT / 'best_configs.json'
paths.ensure_dir(CONFIG_PATH.parent)

series = B.load_series(GAUGE, upto=C.VAL_END)
assert series.time.max() < np.datetime64(C.VAL_END), 'the search must not see test data'
anchors = B.split_anchors(series, 'val')[::ANCHOR_STRIDE]
print(f"{C.gauge_info(GAUGE)['label']}: {len(anchors):,} validation anchors, methods {METHODS}")

storage = f"sqlite:///{(OUT / 'optuna.db').as_posix()}"
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
cache = B.ModelCache()


# %% Search spaces ------------------------------------------------------------
def suggest(trial, method, track, parity):
    r = RANGES[track]
    if method == 'ar':
        cfg = {'order': trial.suggest_categorical('order', PARITY_ORDERS if parity else r['ar_orders']),
               'estimator': trial.suggest_categorical('estimator', ['yw', 'burg'])}
        if cfg['estimator'] == 'yw':
            cfg['jitter'] = trial.suggest_float('jitter', 1e-13, 1e-1, log=True)
        return cfg
    if method == 'sarima':
        cfg = {'p': trial.suggest_int('p', 0, r['p_max']), 'd': trial.suggest_int('d', 0, 1),
               'q': trial.suggest_int('q', 0, r['q_max']), 'P': trial.suggest_int('P', 0, r['P_max']),
               'D': trial.suggest_int('D', 0, 1), 'Q': trial.suggest_int('Q', 0, r['Q_max']),
               'P2': trial.suggest_int('P2', 0, r['P2_max']), 'D2': trial.suggest_int('D2', 0, 1),
               'season': r['season']}
        if cfg['p'] + cfg['q'] + cfg['P'] + cfg['Q'] + cfg['P2'] == 0:
            raise optuna.TrialPruned('nothing to estimate')
        return cfg
    if method == 'gp':
        return {'Q': trial.suggest_categorical('Q', r['gp_components']),
                'centres': trial.suggest_categorical('centres', ['record', 'welch']),
                'history': PARITY_HISTORY if parity else trial.suggest_categorical('history', r['gp_history']),
                'nugget_scale': trial.suggest_float('nugget_scale', 1.0, 1e4, log=True),
                'width_scale': trial.suggest_float('width_scale', 1e-4, 100.0, log=True)}
    if method == 'dhr':
        return {'K': trial.suggest_categorical('K', r['dhr_K']),
                'ar_order': trial.suggest_categorical('ar_order', [1, 2, 4, 8, 16, 32, 64]),
                'cycle_noise': trial.suggest_float('cycle_noise', 1e-9, 100.0, log=True),
                'damping_gap': trial.suggest_float('damping_gap', 1e-10, 1e-1, log=True),
                'obs_noise': trial.suggest_float('obs_noise', 1e-3, 10.0, log=True)}
    if method == 'ridge':
        history = PARITY_HISTORY if parity else trial.suggest_categorical('history', r['ridge_history'])
        spacing = 1 if parity else trial.suggest_categorical('spacing', r['ridge_spacing'])
        spacing = max(spacing, int(np.ceil((history - C.LOOKBACK) / MAX_RIDGE_LAGS)), 1)
        cfg = {'history': history, 'spacing': spacing,
               'ridge_lambda': trial.suggest_float('ridge_lambda', 1e-14, 1.0, log=True)}
        if track == 'residual':
            cfg['use_utide'] = trial.suggest_categorical('use_utide', [False, True])
        return cfg
    raise ValueError(method)


def objective(trial, method, track, parity):
    cfg = suggest(trial, method, track, parity)
    t0 = time.time()
    try:
        model = B.build_model(method, cfg, series, track, RANGES[track]['dhr_frequencies'], cache)
        pred = model.forecast(series, track, anchors)
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
        raise optuna.TrialPruned(f'{type(exc).__name__}: {exc}')
    true, flag = B.targets(series.track(track), series.flagged, anchors)
    score = B.mean_lead_mse(pred, true, flag)
    if not np.isfinite(score):
        raise optuna.TrialPruned('non-finite score')
    trial.set_user_attr('config', cfg)
    trial.set_user_attr('seconds', time.time() - t0)
    return score


def edge_report(study, method, track, parity):
    """Parameters of the winner sitting on the boundary of their range."""
    rows = []
    best = study.best_trial
    for name, value in best.params.items():
        dist = best.distributions[name]
        if isinstance(dist, optuna.distributions.CategoricalDistribution):
            numeric = [c for c in dist.choices if isinstance(c, (int, float)) and not isinstance(c, bool)]
            on_edge = len(numeric) > 1 and value in (min(numeric), max(numeric))
        else:
            low, high = dist.low, dist.high
            scale = np.log if getattr(dist, 'log', False) else (lambda v: v)
            position = (scale(value) - scale(low)) / (scale(high) - scale(low)) if high > low else 0.5
            on_edge = position < 0.03 or position > 0.97
        if on_edge:
            rows.append({'method': method, 'track': track, 'parity': parity, 'parameter': name, 'value': value})
    return rows


# %% Run ----------------------------------------------------------------------
best_configs = json.loads(CONFIG_PATH.read_text(encoding='utf-8')) if CONFIG_PATH.exists() else {}
edges, shortfall = [], []
for track in C.BASELINE_TRACKS:
    for method in METHODS:
        variants = [False, True] if method in C.BASELINE_PARITY_METHODS else [False]
        for parity in variants:
            name = f"{method}_{track}{'_parity' if parity else ''}"
            target = PARITY_TRIALS if parity else TRIALS[method]
            study = optuna.create_study(study_name=name + ROUND, storage=storage, direction='minimize', load_if_exists=True,
                                        sampler=optuna.samplers.TPESampler(seed=C.SEED, multivariate=True, group=True,
                                                                           n_startup_trials=max(10, target // 4)))
            finished = lambda: sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
                                   for t in study.trials)
            if not study.trials and name in optuna.study.get_all_study_names(storage):
                study.enqueue_trial(optuna.load_study(study_name=name, storage=storage).best_params)
            remaining = target - finished()
            t0 = time.time()
            if remaining > 0:
                study.optimize(lambda tr: objective(tr, method, track, parity), n_trials=remaining, timeout=TIMEOUT_S,
                               gc_after_trial=True)
            complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if finished() < target:
                shortfall.append({'study': name + ROUND, 'finished': finished(), 'target': target, 'budget_s': TIMEOUT_S})
                print(f"!! {name}: stopped by the time budget at {finished()}/{target} trials")
            if not complete:
                print(f"!! {name}: no trial completed")
                continue
            study.trials_dataframe().to_csv(OUT / f'search_{name}{ROUND}.csv', index=False)
            best = study.best_trial
            if name in best_configs and best.value >= best_configs[name]['val_mean_lead_mse']:
                continue
            best_configs[name] = {'method': method, 'track': track, 'parity': parity,
                                  'config': best.user_attrs['config'], 'val_mean_lead_mse': best.value,
                                  'val_rmse': float(np.sqrt(best.value)), 'trials': len(study.trials)}
            edges += edge_report(study, method, track, parity)
            print(f"{name}: val RMSE {np.sqrt(best.value):.4f} m, {len(complete)}/{finished()} trials ok "
                  f"in {(time.time() - t0) / 60:.1f} min\n    {best.user_attrs['config']}", flush=True)
            CONFIG_PATH.write_text(json.dumps(best_configs, indent=2), encoding='utf-8')

pd.DataFrame(edges, columns=['method', 'track', 'parity', 'parameter', 'value']).to_csv(OUT / 'edge_winners.csv', index=False)
pd.DataFrame(shortfall, columns=['study', 'finished', 'target', 'budget_s']).to_csv(OUT / 'search_shortfall.csv', index=False)
print(f"\nedge winners: {len(edges)}; shortfalls: {len(shortfall)}\nfrozen configs: {CONFIG_PATH}")
