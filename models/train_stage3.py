"""Stage 3 -- multi-resolution meteorological fusion.

Script form of `models/notebooks/stage3_meteo_fusion.ipynb`. Runs the per-model random
hyperparameter search, trains the multi-branch model and its tide-only ablation at the
winning settings, evaluates both against UTide, and runs the training-data-volume
ablation. SHAP interpretability lives in `models/plot_stage3.py`, since it is an analysis
of a trained model rather than part of training.

    python models/train_stage3.py
    # or run the `# %%` cells interactively in VS Code

Outputs (all under outputs/stage3_outputs/):
    stage3_metrics.csv                              per-horizon RMSE/MAE, 30 horizons
    stage3_lr_search_{multi,tideonly}.csv            random-search trial logs
    stage3_reduced_training_data_metrics.csv         training-volume ablation
    stage3_reduced_training_utide_only_metrics.csv   UTide alone per training window
    stage3_observed_reference.csv                    observed mean/std per horizon
    stage3_history.json                              training curves, for plot_stage3.py
Model weights go to models/checkpoints/ (tracked): stage3_*.pt
"""
# %% Bootstrap
import pathlib
import sys

_start = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / '_')).resolve()
REPO_ROOT = next(p for p in _start.parents if (p / 'utils' / 'paths.py').exists())
sys.path.insert(0, str(REPO_ROOT))

# %% Imports
import numpy as np
import pandas as pd
import torch

from utils import paths
from utils.models import config as C
from utils.models import datasets, training
from utils.models.architectures import TidalSeq2SeqMulti, TidalSeq2SeqTideOnly

# %% ------------------------------------------------------------------ CONFIG
LOOKBACK = C.LOOKBACK
TRAIN_STRIDE = C.TRAIN_STRIDE
EPOCHS = 60        # full-budget cap, used for the final runs and the s.12 ablation
PATIENCE = 10

# Random hyperparameter search, run independently per model before its full-budget
# retrain -- forcing identical hyperparameters onto two different architectures would
# risk confounding the ablation.
LR_SEARCH_MIN, LR_SEARCH_MAX = 3e-4, 3e-3   # one order of magnitude either side of stage 2's 1e-3
WEIGHT_DECAY_CHOICES = [0.0, 1e-5, 1e-4]
BATCH_SIZE_CHOICES = [256, 512, 1024]
N_SEARCH_TRIALS = 6
SEARCH_EPOCHS = 12
SEARCH_PATIENCE = 4

USE_HORIZON_LOSS_WEIGHTING = False   # off by default; see the notebook's s.9 discussion

HORIZONS = C.STAGE3_HORIZONS
HORIZON_NAMES = list(HORIZONS)
HORIZON_STEPS = list(HORIZONS.values())
N_HORIZONS = len(HORIZON_STEPS)
HEADLINE_NAMES = list(C.HEADLINE_HORIZONS)   # the stage-2-comparable seven

LOAD_FROM_CHECKPOINT = True   # False forces search + retrain even if checkpoints exist
RUN_REDUCED_TRAINING = True   # s.12 training-data-volume ablation

OUTPUT_DIR = paths.ensure_dir(paths.STAGE3_OUTPUT_DIR)      # tables (gitignored scratch)
CHECKPOINT_DIR = paths.ensure_dir(paths.CHECKPOINTS_DIR)     # model weights (tracked)
DEVICE = C.describe_device()
C.set_seed()
torch.backends.cudnn.benchmark = True

print(f"{N_HORIZONS} trained horizons: {HORIZON_NAMES}")
print(f"decoder sequence length: {C.DECODER_SEQ_LEN} "
      f"(1 leading pad + {N_HORIZONS} horizons + {len(C.TRAIL_PAD_DAYS)} trailing pad)")

# %% Data (gauge + meteo, multi-resolution windows) -------------------------
data = datasets.prepare_stage3(DEVICE, HORIZON_STEPS, C.DECODER_SEQ_STEPS, C.TARGET_SLICE_IN_DECODER,
                              lookback=LOOKBACK, train_stride=TRAIN_STRIDE)

predict_stage3 = lambda model, batch: model(batch['X'], batch['Xm'], batch['Xr'], batch['Ydec'])[0]
predict_stage3_tideonly = lambda model, batch: model(batch['X'], batch['Ydec'])[0]

make_multi = lambda: TidalSeq2SeqMulti(C.DECODER_SEQ_LEN, C.TARGET_SLICE_IN_DECODER)
make_tideonly = lambda: TidalSeq2SeqTideOnly(C.DECODER_SEQ_LEN, C.TARGET_SLICE_IN_DECODER)
print(f"TidalSeq2SeqMulti: {sum(p.numel() for p in make_multi().parameters()):,} parameters")
print(f"TidalSeq2SeqTideOnly: {sum(p.numel() for p in make_tideonly().parameters()):,} parameters")

# Optional per-horizon loss weighting (inverse target variance, mean-normalised)
HORIZON_WEIGHTS = None
if USE_HORIZON_LOSS_WEIGHTING:
    with torch.no_grad():
        per_horizon_var = data.train_tensors['Y'].var(dim=0)
        HORIZON_WEIGHTS = (1.0 / per_horizon_var)
        HORIZON_WEIGHTS = (HORIZON_WEIGHTS / HORIZON_WEIGHTS.mean()).to(DEVICE)
    print("Horizon loss weighting ENABLED -- weights:", HORIZON_WEIGHTS.cpu().numpy().round(3))

# %% Random hyperparameter search ------------------------------------------
def sample_hparams(rng):
    log_lr = rng.uniform(np.log(LR_SEARCH_MIN), np.log(LR_SEARCH_MAX))
    return {'lr': float(np.exp(log_lr)),
            'weight_decay': float(rng.choice(WEIGHT_DECAY_CHOICES)),
            'batch_size': int(rng.choice(BATCH_SIZE_CHOICES))}


def random_search(model_ctor, predict_fn, train_tensors, val_tensors, label, rng,
                  n_trials=N_SEARCH_TRIALS, epochs=SEARCH_EPOCHS, patience=SEARCH_PATIENCE):
    """Short, cheap trials at a reduced epoch budget, as a proxy for the full run."""
    trial_records = []
    for t in range(n_trials):
        hp = sample_hparams(rng)
        C.set_seed()  # identical init/batch order across trials -- only the hyperparameters differ
        model = model_ctor().to(DEVICE)
        _, history = training.train_model(
            model, predict_fn, train_tensors, val_tensors,
            epochs=epochs, batch_size=hp['batch_size'], lr=hp['lr'], patience=patience,
            weight_decay=hp['weight_decay'], horizon_weights=HORIZON_WEIGHTS,
            label=f'{label}-search{t + 1}', device=DEVICE)
        best_val = min(history['val_loss'])
        trial_records.append({**hp, 'best_val_loss': best_val})
        print(f"[{label}] trial {t + 1}/{n_trials}: lr={hp['lr']:.2e}  "
              f"weight_decay={hp['weight_decay']:.0e}  batch_size={hp['batch_size']}  "
              f"-> best_val_loss={best_val:.5f}")

    trials_df = pd.DataFrame(trial_records).sort_values('best_val_loss').reset_index(drop=True)
    best = trials_df.iloc[0].to_dict()
    print(f"[{label}] winner: lr={best['lr']:.2e}  weight_decay={best['weight_decay']:.0e}  "
          f"batch_size={int(best['batch_size'])}  (val_loss={best['best_val_loss']:.5f})")
    return best, trials_df


def load_or_search(checkpoint_path, search_fn, label):
    """Reuse the winning hyperparameters already saved in a checkpoint's config, if one
    exists -- avoids re-running search trials just to recover values that are only needed
    for a from-scratch retrain we are about to skip anyway."""
    if LOAD_FROM_CHECKPOINT and checkpoint_path.exists():
        cfg = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)['config']
        best = {'lr': cfg['lr'], 'weight_decay': cfg['weight_decay'], 'batch_size': cfg['batch_size']}
        print(f"[{label}] using search-winning hyperparameters saved in {checkpoint_path}: {best}")
        return best, None
    return search_fn()


search_rng = np.random.default_rng(C.SEED)

ckpt_multi = CHECKPOINT_DIR / 'stage3_tidal_seq2seq_multi.pt'
best_multi, search_trials_multi = load_or_search(
    ckpt_multi,
    lambda: random_search(make_multi, predict_stage3, data.train_tensors, data.val_tensors,
                          label='stage3-multi', rng=search_rng),
    label='stage3-multi')
LR_MULTI, WD_MULTI, BATCH_MULTI = best_multi['lr'], best_multi['weight_decay'], int(best_multi['batch_size'])
if search_trials_multi is not None:
    search_trials_multi.to_csv(OUTPUT_DIR / 'stage3_lr_search_multi.csv', index=False)
    print(f"saved {OUTPUT_DIR / 'stage3_lr_search_multi.csv'}")

ckpt_tideonly = CHECKPOINT_DIR / 'stage3_tidal_seq2seq_tideonly.pt'
best_tideonly, search_trials_tideonly = load_or_search(
    ckpt_tideonly,
    lambda: random_search(make_tideonly, predict_stage3_tideonly, data.train_tensors_tideonly,
                          data.val_tensors_tideonly, label='stage3-tideonly', rng=search_rng),
    label='stage3-tideonly')
LR_TIDEONLY, WD_TIDEONLY, BATCH_TIDEONLY = (
    best_tideonly['lr'], best_tideonly['weight_decay'], int(best_tideonly['batch_size']))
if search_trials_tideonly is not None:
    search_trials_tideonly.to_csv(OUTPUT_DIR / 'stage3_lr_search_tideonly.csv', index=False)
    print(f"saved {OUTPUT_DIR / 'stage3_lr_search_tideonly.csv'}")

# %% Final training at the winning hyperparameters, full budget -------------
histories = {}

C.set_seed()
stage3_model = make_multi().to(DEVICE)
stage3_model, histories['Stage-3 (meteo fusion)'] = training.load_or_train(
    stage3_model, ckpt_multi,
    lambda m: training.train_model(m, predict_stage3, data.train_tensors, data.val_tensors,
                                   epochs=EPOCHS, batch_size=BATCH_MULTI, lr=LR_MULTI,
                                   patience=PATIENCE, weight_decay=WD_MULTI,
                                   horizon_weights=HORIZON_WEIGHTS, label='stage3-multi', device=DEVICE),
    label='stage3-multi', load_from_checkpoint=LOAD_FROM_CHECKPOINT, device=DEVICE)

C.set_seed()
stage3_tideonly_model = make_tideonly().to(DEVICE)
stage3_tideonly_model, histories['Stage-3 (tide-only, no meteo)'] = training.load_or_train(
    stage3_tideonly_model, ckpt_tideonly,
    lambda m: training.train_model(m, predict_stage3_tideonly, data.train_tensors_tideonly,
                                   data.val_tensors_tideonly, epochs=EPOCHS, batch_size=BATCH_TIDEONLY,
                                   lr=LR_TIDEONLY, patience=PATIENCE, weight_decay=WD_TIDEONLY,
                                   horizon_weights=HORIZON_WEIGHTS, label='stage3-tideonly', device=DEVICE),
    label='stage3-tideonly', load_from_checkpoint=LOAD_FROM_CHECKPOINT, device=DEVICE)

training.save_history(histories, str(OUTPUT_DIR / 'stage3_history.json'))

# %% Evaluation (test set, touched once) ------------------------------------
def observed_stats(y_raw, flagged):
    """Same masking discipline as masked_rmse_mae, but reporting the observed value's own
    scale (mean, std) per horizon rather than an error -- the context these tables need."""
    means, stds = [], []
    for j in range(y_raw.shape[1]):
        mask = ~flagged[:, j]
        means.append(float(np.mean(y_raw[mask, j])))
        stds.append(float(np.std(y_raw[mask, j])))
    return np.array(means), np.array(stds)


scaler = (data.train_mean, data.train_std)
rmse_utide, mae_utide = training.masked_rmse_mae(data.Ydec_targets_test, data.Y_test, data.Flag_test)
rmse_s3, mae_s3, _ = training.compute_metrics(
    stage3_model, predict_stage3, data.test_tensors, data.Y_test, data.Flag_test, *scaler)
rmse_s3_tideonly, mae_s3_tideonly, _ = training.compute_metrics(
    stage3_tideonly_model, predict_stage3_tideonly, data.test_tensors_tideonly,
    data.Y_test, data.Flag_test, *scaler)
obs_mean_test, obs_std_test = observed_stats(data.Y_test, data.Flag_test)

results = []
for j, hname in enumerate(HORIZON_NAMES):
    results.append({'model': 'UTide standalone', 'horizon': hname,
                    'rmse_m': rmse_utide[j], 'mae_m': mae_utide[j]})
    results.append({'model': 'Stage-3 (meteo fusion)', 'horizon': hname,
                    'rmse_m': rmse_s3[j], 'mae_m': mae_s3[j]})
    results.append({'model': 'Stage-3 (tide-only, no meteo)', 'horizon': hname,
                    'rmse_m': rmse_s3_tideonly[j], 'mae_m': mae_s3_tideonly[j]})

metrics_df = pd.DataFrame(results)
metrics_path = OUTPUT_DIR / 'stage3_metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
print(f"saved {metrics_path}")

# observed-level reference row, so the plotting script can put RMSE in context
obs_path = OUTPUT_DIR / 'stage3_observed_reference.csv'
pd.DataFrame({'horizon': HORIZON_NAMES, 'observed_mean_m': obs_mean_test,
              'observed_std_m': obs_std_test}).to_csv(obs_path, index=False)
print(f"saved {obs_path}")

print("\nRMSE (m), headline 7 horizons -- directly comparable to stage 2's own table:")
print(metrics_df[metrics_df.horizon.isin(HEADLINE_NAMES)].pivot(
    index='model', columns='horizon', values='rmse_m')[HEADLINE_NAMES].round(4))

# positive = the full model's RMSE is lower than tide-only's -> meteo/river helped
meteo_improvement_pct = 100 * (rmse_s3_tideonly - rmse_s3) / rmse_s3_tideonly
for h, pct in zip(HEADLINE_NAMES, meteo_improvement_pct[:len(HEADLINE_NAMES)]):
    print(f"  {h:>6s}: full model vs. tide-only RMSE: {pct:+.1f}% (positive = adding meteo/river helped)")

# %% Save checkpoints -------------------------------------------------------
torch.save({
    'model_state_dict': stage3_model.state_dict(),
    'config': {
        'enc_hidden_size': 64, 'enc_num_layers': 1,
        'meteo_hidden_size': 64, 'meteo_lookback_hours': C.METEO_LOOKBACK_HOURS,
        'river_hidden_size': 32, 'river_lookback_days': C.RIVER_LOOKBACK_DAYS,
        'dec_hidden_size': 64, 'dec_num_layers': 1, 'attn_dims': (64, 64, 32),
        'lookback': LOOKBACK, 'horizons': HORIZONS, 'decoder_seq_steps': C.DECODER_SEQ_STEPS,
        'target_slice_in_decoder': [C.TARGET_SLICE_IN_DECODER.start, C.TARGET_SLICE_IN_DECODER.stop],
        'lr': LR_MULTI, 'weight_decay': WD_MULTI, 'batch_size': BATCH_MULTI,  # search winner
    },
    'scalers': data.scalers,
}, ckpt_multi)
print(f"saved {ckpt_multi}")

torch.save({
    'model_state_dict': stage3_tideonly_model.state_dict(),
    'config': {
        'enc_hidden_size': 64, 'enc_num_layers': 1,
        'dec_hidden_size': 64, 'dec_num_layers': 1, 'attn_dim': 64,
        'lookback': LOOKBACK, 'horizons': HORIZONS, 'decoder_seq_steps': C.DECODER_SEQ_STEPS,
        'target_slice_in_decoder': [C.TARGET_SLICE_IN_DECODER.start, C.TARGET_SLICE_IN_DECODER.stop],
        'lr': LR_TIDEONLY, 'weight_decay': WD_TIDEONLY, 'batch_size': BATCH_TIDEONLY,
    },
    'scalers': {'tide_utide': data.scalers['tide_utide']},
}, ckpt_tideonly)
print(f"saved {ckpt_tideonly}")

# %% Training-data-volume ablation ------------------------------------------
# Mirrors stage 2 s.13.3: train end fixed, only the window start moves, UTide refit per
# window. Hyperparameters are held at the tide-only model's winning config, since here
# training-data volume is the one thing meant to vary. Only the tide-only model is
# retrained -- it outperformed the meteo-fusion model above, so it is what this tracks.
if RUN_REDUCED_TRAINING:
    reduced_results = [{'model': '14yr (full)', 'horizon': h, 'rmse_m': rmse_s3_tideonly[j],
                        'mae_m': mae_s3_tideonly[j], 'train_years': 14}
                       for j, h in enumerate(HORIZON_NAMES)]
    reduced_utide_only = [{'model': '14yr (full)', 'horizon': h, 'rmse_m': rmse_utide[j],
                           'mae_m': mae_utide[j], 'train_years': 14}
                          for j, h in enumerate(HORIZON_NAMES)]

    for name, start in C.REDUCED_TRAIN_STARTS.items():
        n_years = int(name.replace('yr', ''))
        scen = datasets.prepare_stage3_reduced(
            DEVICE, data, start, HORIZON_STEPS, C.DECODER_SEQ_STEPS, C.TARGET_SLICE_IN_DECODER,
            lookback=LOOKBACK, train_stride=TRAIN_STRIDE, name=name)

        rmse_u, mae_u = training.masked_rmse_mae(scen['Ydec_test'], scen['Y_test'], scen['Flag_test'])
        reduced_utide_only += [{'model': name, 'horizon': h, 'rmse_m': rmse_u[j],
                                'mae_m': mae_u[j], 'train_years': n_years}
                               for j, h in enumerate(HORIZON_NAMES)]

        ckpt_reduced = CHECKPOINT_DIR / f'stage3_tidal_seq2seq_tideonly_reduced_{name}.pt'
        C.set_seed()
        m = make_tideonly().to(DEVICE)
        m, _ = training.load_or_train(
            m, ckpt_reduced,
            lambda mm, scen=scen, name=name: training.train_model(
                mm, predict_stage3_tideonly, scen['train_tensors'], scen['val_tensors'],
                epochs=EPOCHS, batch_size=BATCH_TIDEONLY, lr=LR_TIDEONLY, patience=PATIENCE,
                weight_decay=WD_TIDEONLY, horizon_weights=HORIZON_WEIGHTS,
                label=f'stage3-tideonly-{name}', device=DEVICE),
            label=f'stage3-tideonly-{name}', load_from_checkpoint=LOAD_FROM_CHECKPOINT, device=DEVICE)
        torch.save({'model_state_dict': m.state_dict(),
                    'scaler': {'mean': scen['mean'], 'std': scen['std']},
                    'config': {'train_years': n_years, 'train_start': start}}, ckpt_reduced)
        print(f"saved {ckpt_reduced}")

        rmse_r, mae_r, _ = training.compute_metrics(
            m, predict_stage3_tideonly, scen['test_tensors'], scen['Y_test'], scen['Flag_test'],
            scen['mean'], scen['std'])
        reduced_results += [{'model': name, 'horizon': h, 'rmse_m': rmse_r[j],
                             'mae_m': mae_r[j], 'train_years': n_years}
                            for j, h in enumerate(HORIZON_NAMES)]

    reduced_path = OUTPUT_DIR / 'stage3_reduced_training_data_metrics.csv'
    pd.DataFrame(reduced_results).to_csv(reduced_path, index=False)
    print(f"saved {reduced_path}")

    utide_only_path = OUTPUT_DIR / 'stage3_reduced_training_utide_only_metrics.csv'
    pd.DataFrame(reduced_utide_only).to_csv(utide_only_path, index=False)
    print(f"saved {utide_only_path}")

# %% Done -------------------------------------------------------------------
print(f"\nAll stage-3 outputs are under: {OUTPUT_DIR}")
for p in sorted(OUTPUT_DIR.glob('*')):
    print(f"  {p.name}")
