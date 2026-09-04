"""Stage 1 -- baseline MLP / RNN / LSTM single-step tide-level models.

Script form of `models/notebooks/initial_modelling.ipynb`. Trains the three baselines,
scores them against persistence and a UTide harmonic fit, and writes every table and
checkpoint the plotting script needs. No figures are produced here -- run
`models/plot_stage1.py` for those.

Run it either way:
    python models/train_stage1.py
    # or open it in VS Code and run the `# %%` cells interactively

Outputs -- tables under outputs/initial_outputs/ (gitignored scratch):
    initial_metrics.csv        one-step-ahead RMSE/MAE, val and test
    initial_rollout_rmse.csv   recursive-rollout RMSE at 10min .. 48h
    initial_history.json       training curves, for plot_stage1.py
and weights under models/checkpoints/ (tracked):
    initial_{mlp,rnn,lstm}.pt  model checkpoints
"""
# %% Bootstrap -- make `utils` importable whether run as a file or cell-by-cell
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
from utils.models.architectures import MLP, LSTMModel, RNNModel

# %% ------------------------------------------------------------------ CONFIG
LOOKBACK = C.LOOKBACK          # 96 -> 16h of 10-minute steps
TRAIN_STRIDE = C.TRAIN_STRIDE  # subsample training windows only
BATCH_SIZE = 1024
EPOCHS = 60
PATIENCE = 10
LR = 1e-3

LOAD_FROM_CHECKPOINT = True    # False forces a full retrain even if checkpoints exist
RUN_ROLLOUT = True             # recursive multi-step evaluation (section 13 of the notebook)

OUTPUT_DIR = paths.ensure_dir(paths.STAGE1_OUTPUT_DIR)      # tables (gitignored scratch)
CHECKPOINT_DIR = paths.ensure_dir(paths.CHECKPOINTS_DIR)     # model weights (tracked)
DEVICE = C.describe_device()
C.set_seed()
torch.backends.cudnn.benchmark = True

# %% Data ------------------------------------------------------------------
data = datasets.prepare_stage1(DEVICE, lookback=LOOKBACK, train_stride=TRAIN_STRIDE)

# %% Baseline: persistence -- y(t+1) = y(t), flagged targets masked out ------
results = []
for split_name, X_raw, y_raw, flg in [('val', data.X_val, data.y_val, data.flag_val),
                                      ('test', data.X_test, data.y_test, data.flag_test)]:
    rmse, mae = training.masked_rmse_mae(X_raw[:, -1], y_raw, flg)
    results.append({'model': 'Persistence', 'split': split_name, 'rmse_m': rmse, 'mae_m': mae})
    print(f"Persistence {split_name}: rmse={rmse:.4f} m, mae={mae:.4f} m")

# %% Baseline: UTide harmonic ----------------------------------------------
# Fit on train+val (2004-2020) -- stage 1 uses UTide purely as a comparator, so unlike
# stages 2/3 it is allowed the validation period too. Flagged points excluded from the fit.
coef = datasets.fit_utide(data.df, end=C.VAL_END)
tide_test = datasets.reconstruct_utide(data.time_test, coef, min_snr=2)
rmse_u, mae_u = training.masked_rmse_mae(tide_test, data.y_test, data.flag_test)
results.append({'model': 'UTide', 'split': 'test', 'rmse_m': rmse_u, 'mae_m': mae_u})
print(f"UTide test: rmse={rmse_u:.4f} m, mae={mae_u:.4f} m")

# %% Train MLP / RNN / LSTM -------------------------------------------------
model_factories = {
    'MLP': lambda: MLP(LOOKBACK),
    'RNN': lambda: RNNModel(hidden_size=64, num_layers=1),
    'LSTM': lambda: LSTMModel(hidden_size=64, num_layers=1),
}
predict_flat = lambda model, batch: model(batch['X'])

trained, histories = {}, {}
for name, factory in model_factories.items():
    C.set_seed()
    model = factory().to(DEVICE)
    checkpoint_path = CHECKPOINT_DIR / f'initial_{name.lower()}.pt'

    def _train(m, name=name):
        print(f"\n=== Training {name} ===")
        return training.train_model(
            m, predict_flat, data.train_tensors, data.val_tensors,
            epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE,
            label=name, device=DEVICE,
        )

    model, history = training.load_or_train(model, checkpoint_path, _train, label=name,
                                            load_from_checkpoint=LOAD_FROM_CHECKPOINT, device=DEVICE)
    trained[name] = model
    histories[name] = history

# %% One-step-ahead metrics -------------------------------------------------
for name, model in trained.items():
    rmse_v, mae_v, _ = training.compute_metrics(model, predict_flat, data.val_tensors,
                                                data.y_val, data.flag_val, data.train_mean, data.train_std)
    rmse_t, mae_t, _ = training.compute_metrics(model, predict_flat, data.test_tensors,
                                                data.y_test, data.flag_test, data.train_mean, data.train_std)
    results.append({'model': name, 'split': 'val', 'rmse_m': rmse_v, 'mae_m': mae_v})
    results.append({'model': name, 'split': 'test', 'rmse_m': rmse_t, 'mae_m': mae_t})

results_df = pd.DataFrame(results)
print(results_df.pivot(index='model', columns='split', values=['rmse_m', 'mae_m']).round(4))

metrics_path = OUTPUT_DIR / 'initial_metrics.csv'
results_df.to_csv(metrics_path, index=False)
print(f"saved {metrics_path}")

# %% Save checkpoints and training curves -----------------------------------
# State dict only: the architecture is fixed by LOOKBACK/hidden sizes above and is not
# swept anywhere, so re-instantiating the same class is enough to load these back.
for name, model in trained.items():
    checkpoint_path = CHECKPOINT_DIR / f'initial_{name.lower()}.pt'
    torch.save({'model_state_dict': model.state_dict(),
                'config': {'lookback': LOOKBACK},
                'scaler': {'mean': data.train_mean, 'std': data.train_std}}, checkpoint_path)
    print(f"saved {checkpoint_path}")

training.save_history(histories, str(OUTPUT_DIR / 'initial_history.json'))

# %% Recursive multi-step rollout evaluation --------------------------------
# Section 10's RMSE is one-step-ahead. Here each neural model is rolled forward
# recursively -- its own prediction is appended to the input window and fed back in as the
# newest observation -- out to 48h, then scored at 10min/1h/6h/24h/48h.
if RUN_ROLLOUT:
    HORIZONS = C.STAGE1_ROLLOUT_HORIZONS
    MAX_HORIZON = max(HORIZONS.values())
    ROLLOUT_STRIDE = C.STAGE1_ROLLOUT_STRIDE

    test_values = data.test_df['Observed_ODN'].to_numpy(dtype=np.float32)
    test_flagged = data.test_df['is_flagged'].to_numpy()
    test_seg_id = (data.test_df['DateTime'].diff() > C.STEP).cumsum().to_numpy()

    # valid start index i: LOOKBACK history before i and MAX_HORIZON steps after i, all
    # within one gap-free segment (seg_id is non-decreasing, so equal endpoints imply
    # constant in between)
    candidates = np.arange(LOOKBACK, len(data.test_df) - MAX_HORIZON + 1)
    same_segment = test_seg_id[candidates - LOOKBACK] == test_seg_id[candidates + MAX_HORIZON - 1]
    rollout_start_idx = candidates[same_segment][::ROLLOUT_STRIDE]
    print(f"{len(rollout_start_idx):,} rollout start points across the test set "
          f"(every {ROLLOUT_STRIDE * 10}min, {MAX_HORIZON * 10}min horizon each)")

    @torch.no_grad()
    def recursive_rollout(model, start_idx, values, horizon_steps, batch_size=2048):
        model.eval()
        n = len(start_idx)
        preds = np.empty((n, horizon_steps), dtype=np.float32)
        for b0 in range(0, n, batch_size):
            b_idx = start_idx[b0:b0 + batch_size]
            hist = np.stack([values[i - LOOKBACK:i] for i in b_idx])
            hist_t = datasets.to_tensor((hist - data.train_mean) / data.train_std, DEVICE)
            for step in range(horizon_steps):
                next_scaled = model(hist_t)
                preds[b0:b0 + len(b_idx), step] = (next_scaled * data.train_std + data.train_mean).cpu().numpy()
                hist_t = torch.cat([hist_t[:, 1:], next_scaled.unsqueeze(-1)], dim=1)
        return preds

    rollout_preds = {}
    for name, model in trained.items():
        rollout_preds[name] = recursive_rollout(model, rollout_start_idx, test_values, MAX_HORIZON)
        print(f"[{name}] rollout done")

    # Persistence: flat continuation of the last observed value -- no mechanism to update
    rollout_preds['Persistence'] = np.repeat(test_values[rollout_start_idx - 1][:, None], MAX_HORIZON, axis=1)

    # UTide: fixed harmonic model, evaluated directly at each target timestamp -- no recursion
    tide_test_full = datasets.reconstruct_utide(data.test_df['DateTime'], coef, min_snr=2)
    rollout_preds['UTide'] = np.stack(
        [tide_test_full[rollout_start_idx + h - 1] for h in range(1, MAX_HORIZON + 1)], axis=1)

    rollout_rows = []
    for name, preds in rollout_preds.items():
        for label, h in HORIZONS.items():
            target_idx = rollout_start_idx + h - 1
            rmse, _ = training.masked_rmse_mae(preds[:, h - 1], test_values[target_idx], test_flagged[target_idx])
            rollout_rows.append({'model': name, 'horizon': label, 'rmse_m': rmse})

    rollout_table = pd.DataFrame(rollout_rows).pivot(
        index='model', columns='horizon', values='rmse_m')[list(HORIZONS)].round(4)
    rollout_table = rollout_table.reindex(['LSTM', 'MLP', 'RNN', 'Persistence', 'UTide'])

    rollout_path = OUTPUT_DIR / 'initial_rollout_rmse.csv'
    rollout_table.to_csv(rollout_path)
    print(f"saved {rollout_path}")
    print(rollout_table)

# %% Done -------------------------------------------------------------------
print(f"\nAll stage-1 outputs are under: {OUTPUT_DIR}")
for p in sorted(OUTPUT_DIR.glob('*')):
    print(f"  {p.name}")
