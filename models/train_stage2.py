"""Stage 2 -- UTide-informed BiLSTM decoder with attention, direct multi-horizon.

Script form of `models/notebooks/stage2_utide_attention.ipynb`. Trains the stage-2
architecture and its comparators, then runs the three robustness checks. Every table it
writes is what `models/plot_stage2.py` draws from; no figures are produced here.

    python models/train_stage2.py
    # or run the `# %%` cells interactively in VS Code

Outputs (all under outputs/stage2_outputs/):
    stage2_metrics.csv                             per-horizon RMSE/MAE, all comparators
    stage2_ablation_metrics.csv                    tide-input ablation (13.1)
    stage2_rmse_by_calendar_year.csv               accuracy later in the test window (13.2)
    stage2_rmse_by_test_week.csv                   the same, by specific weeks
    stage2_reduced_training_data_metrics.csv       training-volume sensitivity (13.3)
    stage2_reduced_training_utide_only_metrics.csv UTide alone per training window
    stage2_history.json                            training curves, for plot_stage2.py
Model weights go to models/checkpoints/ (tracked): stage1_flat_*.pt, stage2_*.pt
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
from utils.models.architectures import FlatMultiHorizonLSTM, FlatMultiHorizonMLP, TidalSeq2Seq

# %% ------------------------------------------------------------------ CONFIG
LOOKBACK = C.LOOKBACK
TRAIN_STRIDE = C.TRAIN_STRIDE
BATCH_SIZE = 512      # halved vs stage 1: attention + multi-horizon decoding costs more memory
EPOCHS = 60
PATIENCE = 10
LR = 1e-3

ENC_HIDDEN_SIZE, ENC_NUM_LAYERS = 64, 1   # stage-1 encoder hyperparameters, reused
DEC_HIDDEN_SIZE, DEC_NUM_LAYERS = 64, 1
ATTN_DIM = 64

HORIZONS = C.STAGE2_HORIZONS
HORIZON_NAMES = list(HORIZONS)
HORIZON_STEPS = list(HORIZONS.values())
N_HORIZONS = len(HORIZON_STEPS)

LOAD_FROM_CHECKPOINT = True   # False forces a full retrain even if checkpoints exist
RUN_TIDE_ABLATION = True      # 13.1 -- retrain with the decoder fed zeros instead of UTide
RUN_YEAR_WEEK_SLICES = True   # 13.2 -- RMSE by calendar year and by specific test weeks
RUN_REDUCED_TRAINING = True   # 13.3 -- retrain on the last 3 / 7 / 10 years

OUTPUT_DIR = paths.ensure_dir(paths.STAGE2_OUTPUT_DIR)      # tables (gitignored scratch)
CHECKPOINT_DIR = paths.ensure_dir(paths.CHECKPOINTS_DIR)     # model weights (tracked)
DEVICE = C.describe_device()
C.set_seed()
torch.backends.cudnn.benchmark = True

# %% Data (UTide fit on train only, reconstructed across the full span) ------
data = datasets.prepare_stage2(DEVICE, HORIZON_STEPS, lookback=LOOKBACK, train_stride=TRAIN_STRIDE)

predict_flat = lambda model, batch: model(batch['X'])
predict_seq2seq = lambda model, batch: model(batch['X'], batch['Ydec'])[0]

# %% Baselines --------------------------------------------------------------
def naive_persistence(X_raw, n_horizons):
    """Last observed value, held flat across every horizon."""
    return np.repeat(X_raw[:, -1:], n_horizons, axis=1)


def tide_aware_persistence(X_raw, horizon_steps, m2_steps=C.M2_STEPS, lookback=LOOKBACK):
    """pred(t+k) = Observed_ODN(t + ref), where ref is k shifted back by whole M2 cycles
    until it lands at or before the forecast origin (ref <= 0) -- so long horizons still
    only use already-observed history. Linear interpolation between the two nearest lag
    steps, since m2_steps is not an integer."""
    preds = np.zeros((X_raw.shape[0], len(horizon_steps)), dtype=np.float32)
    for j, k in enumerate(horizon_steps):
        ref = float(k) - m2_steps
        while ref >= -1e-9:
            ref -= m2_steps
        float_idx = lookback + ref  # fractional position within the lookback array
        idx0 = int(np.floor(float_idx))
        idx0 = max(0, min(idx0, lookback - 1))
        idx1 = min(idx0 + 1, lookback - 1)
        weight = float_idx - idx0
        preds[:, j] = (1 - weight) * X_raw[:, idx0] + weight * X_raw[:, idx1]
    return preds


results = []
naive_test = naive_persistence(data.X_test, N_HORIZONS)
rmse, mae = training.masked_rmse_mae(naive_test, data.Y_test, data.Flag_test)
for j, hname in enumerate(HORIZON_NAMES):
    results.append({'model': 'Naive persistence', 'horizon': hname, 'rmse_m': rmse[j], 'mae_m': mae[j]})

tide_pers_test = tide_aware_persistence(data.X_test, HORIZON_STEPS)
rmse, mae = training.masked_rmse_mae(tide_pers_test, data.Y_test, data.Flag_test)
for j, hname in enumerate(HORIZON_NAMES):
    results.append({'model': 'Tide-aware persistence', 'horizon': hname, 'rmse_m': rmse[j], 'mae_m': mae[j]})

# The UTide standalone comparator is by construction the exact same reconstruction as the
# decoder feature -- just sliced to the windowed test targets and used as a prediction in
# its own right -- so the feature the model sees and the benchmark it is judged against
# can never diverge through a constituent-selection difference.
utide_baseline = data.Ydec_test
rmse_utide, mae_utide = training.masked_rmse_mae(utide_baseline, data.Y_test, data.Flag_test)
for j, hname in enumerate(HORIZON_NAMES):
    results.append({'model': 'UTide standalone', 'horizon': hname, 'rmse_m': rmse_utide[j], 'mae_m': mae_utide[j]})

print(pd.DataFrame(results).pivot(index='model', columns='horizon', values='rmse_m')[HORIZON_NAMES].round(4))

# %% Train: stage-1 baselines reproduced with a flat multi-horizon head ------
histories = {}

ckpt_s1_lstm = CHECKPOINT_DIR / 'stage1_flat_lstm_multihorizon.pt'
C.set_seed()
stage1_lstm_model = FlatMultiHorizonLSTM(ENC_HIDDEN_SIZE, ENC_NUM_LAYERS, N_HORIZONS).to(DEVICE)
stage1_lstm_model, histories['Stage 1 (flat LSTM)'] = training.load_or_train(
    stage1_lstm_model, ckpt_s1_lstm,
    lambda m: training.train_model(m, predict_flat, data.train_tensors_flat, data.val_tensors_flat,
                                   epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE,
                                   label='stage1-lstm', device=DEVICE),
    label='stage1-lstm', load_from_checkpoint=LOAD_FROM_CHECKPOINT, device=DEVICE)

ckpt_s1_mlp = CHECKPOINT_DIR / 'stage1_flat_mlp_multihorizon.pt'
C.set_seed()
stage1_mlp_model = FlatMultiHorizonMLP(LOOKBACK, N_HORIZONS).to(DEVICE)
stage1_mlp_model, histories['Stage 1 (flat MLP)'] = training.load_or_train(
    stage1_mlp_model, ckpt_s1_mlp,
    lambda m: training.train_model(m, predict_flat, data.train_tensors_flat, data.val_tensors_flat,
                                   epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE,
                                   label='stage1-mlp', device=DEVICE),
    label='stage1-mlp', load_from_checkpoint=LOAD_FROM_CHECKPOINT, device=DEVICE)

# %% Train: stage-2 encoder + attention + UTide-informed BiLSTM decoder ------
ckpt_s2 = CHECKPOINT_DIR / 'stage2_tidal_seq2seq.pt'
C.set_seed()
stage2_model = TidalSeq2Seq(ENC_HIDDEN_SIZE, ENC_NUM_LAYERS, DEC_HIDDEN_SIZE, DEC_NUM_LAYERS, ATTN_DIM).to(DEVICE)
stage2_model, histories['Stage 2 (attn decoder)'] = training.load_or_train(
    stage2_model, ckpt_s2,
    lambda m: training.train_model(m, predict_seq2seq, data.train_tensors, data.val_tensors,
                                   epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE,
                                   label='stage2-attn', device=DEVICE),
    label='stage2-attn', load_from_checkpoint=LOAD_FROM_CHECKPOINT, device=DEVICE)

training.save_history(histories, str(OUTPUT_DIR / 'stage2_history.json'))

# %% Evaluation (test set, touched once) ------------------------------------
scaler = (data.train_mean, data.train_std)
rmse_s1_lstm, mae_s1_lstm, _ = training.compute_metrics(
    stage1_lstm_model, predict_flat, data.test_tensors_flat, data.Y_test, data.Flag_test, *scaler)
rmse_s1_mlp, mae_s1_mlp, _ = training.compute_metrics(
    stage1_mlp_model, predict_flat, data.test_tensors_flat, data.Y_test, data.Flag_test, *scaler)
rmse_s2, mae_s2, test_preds_s2 = training.compute_metrics(
    stage2_model, predict_seq2seq, data.test_tensors, data.Y_test, data.Flag_test, *scaler)

for j, hname in enumerate(HORIZON_NAMES):
    results.append({'model': 'Stage-1 baseline (flat LSTM)', 'horizon': hname,
                    'rmse_m': rmse_s1_lstm[j], 'mae_m': mae_s1_lstm[j]})
    results.append({'model': 'Stage-1 baseline (flat MLP)', 'horizon': hname,
                    'rmse_m': rmse_s1_mlp[j], 'mae_m': mae_s1_mlp[j]})
    results.append({'model': 'Stage-2 (attn decoder)', 'horizon': hname,
                    'rmse_m': rmse_s2[j], 'mae_m': mae_s2[j]})

metrics_df = pd.DataFrame(results)
metrics_path = OUTPUT_DIR / 'stage2_metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
print(f"saved {metrics_path}")
print(metrics_df.pivot(index='model', columns='horizon', values='rmse_m')[HORIZON_NAMES].round(4))

# %% Save checkpoints -------------------------------------------------------
torch.save({
    'model_state_dict': stage2_model.state_dict(),
    'config': {'enc_hidden_size': ENC_HIDDEN_SIZE, 'enc_num_layers': ENC_NUM_LAYERS,
               'dec_hidden_size': DEC_HIDDEN_SIZE, 'dec_num_layers': DEC_NUM_LAYERS,
               'attn_dim': ATTN_DIM, 'lookback': LOOKBACK, 'horizons': HORIZONS},
    'scaler': {'mean': data.train_mean, 'std': data.train_std},
}, ckpt_s2)
torch.save({
    'model_state_dict': stage1_lstm_model.state_dict(),
    'config': {'hidden_size': ENC_HIDDEN_SIZE, 'num_layers': ENC_NUM_LAYERS,
               'lookback': LOOKBACK, 'horizons': HORIZONS},
    'scaler': {'mean': data.train_mean, 'std': data.train_std},
}, ckpt_s1_lstm)
torch.save({
    'model_state_dict': stage1_mlp_model.state_dict(),
    'config': {'hidden': (128, 64), 'lookback': LOOKBACK, 'horizons': HORIZONS},
    'scaler': {'mean': data.train_mean, 'std': data.train_std},
}, ckpt_s1_mlp)
print(f"saved {ckpt_s2}\nsaved {ckpt_s1_lstm}\nsaved {ckpt_s1_mlp}")

# %% 13.1 Tide-input ablation ----------------------------------------------
# Retrains the same architecture with the decoder always fed zeros instead of the UTide
# reconstruction; any RMSE gap vs stage2_model is attributable to the tidal input alone.
if RUN_TIDE_ABLATION:
    predict_seq2seq_notide = lambda model, batch: model(batch['X'], torch.zeros_like(batch['Ydec']))[0]
    ckpt_notide = CHECKPOINT_DIR / 'stage2_tidal_seq2seq_notide_ablation.pt'

    C.set_seed()
    stage2_notide_model = TidalSeq2Seq(ENC_HIDDEN_SIZE, ENC_NUM_LAYERS,
                                       DEC_HIDDEN_SIZE, DEC_NUM_LAYERS, ATTN_DIM).to(DEVICE)
    stage2_notide_model, _ = training.load_or_train(
        stage2_notide_model, ckpt_notide,
        lambda m: training.train_model(m, predict_seq2seq_notide, data.train_tensors, data.val_tensors,
                                       epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE,
                                       label='stage2-no-tide-ablation', device=DEVICE),
        label='stage2-no-tide-ablation', load_from_checkpoint=LOAD_FROM_CHECKPOINT, device=DEVICE)
    torch.save({'model_state_dict': stage2_notide_model.state_dict(),
                'scaler': {'mean': data.train_mean, 'std': data.train_std}}, ckpt_notide)
    print(f"saved {ckpt_notide}")

    rmse_notide, mae_notide, _ = training.compute_metrics(
        stage2_notide_model, predict_seq2seq_notide, data.test_tensors, data.Y_test, data.Flag_test, *scaler)

    # The ablation table repeats the baselines so it stands alone as a figure source.
    rmse_naive_a, mae_naive_a = training.masked_rmse_mae(naive_test, data.Y_test, data.Flag_test)
    rmse_tp_a, mae_tp_a = training.masked_rmse_mae(tide_pers_test, data.Y_test, data.Flag_test)

    ablation_rows_by_model = [
        ('Naive persistence', rmse_naive_a, mae_naive_a),
        ('Tide-aware persistence', rmse_tp_a, mae_tp_a),
        ('UTide standalone', rmse_utide, mae_utide),
        ('Stage-2 (with tide)', rmse_s2, mae_s2),
        ('Stage-2 (no tide, ablation)', rmse_notide, mae_notide),
    ]
    ablation_results = [
        {'model': model_name, 'horizon': hname, 'rmse_m': rmses[j], 'mae_m': maes[j]}
        for model_name, rmses, maes in ablation_rows_by_model
        for j, hname in enumerate(HORIZON_NAMES)
    ]

    ablation_df = pd.DataFrame(ablation_results)
    ablation_path = OUTPUT_DIR / 'stage2_ablation_metrics.csv'
    ablation_df.to_csv(ablation_path, index=False)
    print(f"saved {ablation_path}")

    tide_contribution_pct = 100 * (rmse_notide - rmse_s2) / rmse_notide
    for h, pct in zip(HORIZON_NAMES, tide_contribution_pct):
        print(f"  {h:>6s}: removing the tidal input increases RMSE by {pct:.1f}%")

# %% 13.2 Accuracy later in the test window ---------------------------------
if RUN_YEAR_WEEK_SLICES:
    year_rows = []
    for yr in sorted(pd.DatetimeIndex(data.Time_test[:, 0]).year.unique()):
        for j, hname in enumerate(HORIZON_NAMES):
            t = pd.DatetimeIndex(data.Time_test[:, j])
            yr_mask = t.year == yr
            if yr_mask.sum() == 0:
                continue
            err = test_preds_s2[yr_mask, j] - data.Y_test[yr_mask, j]
            err = err[~data.Flag_test[yr_mask, j]]
            year_rows.append({'year': yr, 'horizon': hname,
                              'rmse_m': float(np.sqrt(np.mean(err ** 2))), 'n': len(err)})

    year_df = pd.DataFrame(year_rows)
    year_path = OUTPUT_DIR / 'stage2_rmse_by_calendar_year.csv'
    year_df.to_csv(year_path, index=False)
    print(f"saved {year_path}")
    print(year_df.pivot(index='year', columns='horizon', values='rmse_m')[HORIZON_NAMES].round(4))

    def week_metrics(week_start, label, preds=test_preds_s2):
        week_start = pd.Timestamp(week_start)
        week_end = week_start + pd.Timedelta(days=7)
        rows = []
        for j, hname in enumerate(HORIZON_NAMES):
            t = pd.DatetimeIndex(data.Time_test[:, j])
            mask = (t >= week_start) & (t < week_end)
            if mask.sum() == 0:
                continue
            err = preds[mask, j] - data.Y_test[mask, j]
            err = err[~data.Flag_test[mask, j]]
            rows.append({'week': label, 'horizon': hname,
                         'rmse_m': float(np.sqrt(np.mean(err ** 2))), 'n': len(err)})
        return rows

    week_starts = {'2021-W1 (start of test)': '2021-01-01',
                   '2022-W1': '2022-01-01',
                   '2024-W1 (3yr into test)': '2024-01-01'}
    week_rows = []
    for label, start in week_starts.items():
        week_rows += week_metrics(start, label)

    week_df = pd.DataFrame(week_rows)
    week_path = OUTPUT_DIR / 'stage2_rmse_by_test_week.csv'
    week_df.to_csv(week_path, index=False)
    print(f"saved {week_path}")
    print(week_df.pivot(index='week', columns='horizon', values='rmse_m')[HORIZON_NAMES].round(4))

# %% 13.3 Sensitivity to training-data volume -------------------------------
# Train end fixed at 2018-01-01; only the training window start moves. UTide is refit per
# window, so each scenario sees the decoder input a model trained on that window alone
# would actually have had.
if RUN_REDUCED_TRAINING:
    reduced_results = [{'model': '14yr (full)', 'horizon': h, 'rmse_m': rmse_s2[j],
                        'mae_m': mae_s2[j], 'train_years': 14} for j, h in enumerate(HORIZON_NAMES)]
    reduced_utide_only = [{'model': '14yr (full)', 'horizon': h, 'rmse_m': rmse_utide[j],
                           'mae_m': mae_utide[j], 'train_years': 14} for j, h in enumerate(HORIZON_NAMES)]

    for name, start in C.REDUCED_TRAIN_STARTS.items():
        n_years = int(name.replace('yr', ''))
        scen = datasets.prepare_stage2_reduced(DEVICE, data.df, start, HORIZON_STEPS,
                                               lookback=LOOKBACK, train_stride=TRAIN_STRIDE, name=name)

        rmse_u, mae_u = training.masked_rmse_mae(scen['Ydec_test'], scen['Y_test'], scen['Flag_test'])
        reduced_utide_only += [{'model': name, 'horizon': h, 'rmse_m': rmse_u[j],
                                'mae_m': mae_u[j], 'train_years': n_years}
                               for j, h in enumerate(HORIZON_NAMES)]

        ckpt_reduced = CHECKPOINT_DIR / f'stage2_tidal_seq2seq_reduced_{name}.pt'
        C.set_seed()
        m = TidalSeq2Seq(ENC_HIDDEN_SIZE, ENC_NUM_LAYERS, DEC_HIDDEN_SIZE, DEC_NUM_LAYERS, ATTN_DIM).to(DEVICE)
        m, _ = training.load_or_train(
            m, ckpt_reduced,
            lambda mm, scen=scen, name=name: training.train_model(
                mm, predict_seq2seq, scen['train_tensors'], scen['val_tensors'],
                epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE,
                label=f'stage2-{name}', device=DEVICE),
            label=f'stage2-{name}', load_from_checkpoint=LOAD_FROM_CHECKPOINT, device=DEVICE)
        torch.save({'model_state_dict': m.state_dict(),
                    'scaler': {'mean': scen['mean'], 'std': scen['std']},
                    'config': {'train_years': n_years, 'train_start': start}}, ckpt_reduced)
        print(f"saved {ckpt_reduced}")

        rmse_r, mae_r, _ = training.compute_metrics(
            m, predict_seq2seq, scen['test_tensors'], scen['Y_test'], scen['Flag_test'],
            scen['mean'], scen['std'])
        reduced_results += [{'model': name, 'horizon': h, 'rmse_m': rmse_r[j],
                             'mae_m': mae_r[j], 'train_years': n_years}
                            for j, h in enumerate(HORIZON_NAMES)]

    reduced_df = pd.DataFrame(reduced_results)
    reduced_path = OUTPUT_DIR / 'stage2_reduced_training_data_metrics.csv'
    reduced_df.to_csv(reduced_path, index=False)
    print(f"saved {reduced_path}")

    utide_only_df = pd.DataFrame(reduced_utide_only)
    utide_only_path = OUTPUT_DIR / 'stage2_reduced_training_utide_only_metrics.csv'
    utide_only_df.to_csv(utide_only_path, index=False)
    print(f"saved {utide_only_path}")

# %% Done -------------------------------------------------------------------
print(f"\nAll stage-2 outputs are under: {OUTPUT_DIR}")
for p in sorted(OUTPUT_DIR.glob('*')):
    print(f"  {p.name}")
