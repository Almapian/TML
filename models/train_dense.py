"""Dense pipeline -- continuous 0-72h forecasts from the stage 2 and stage 3 tide-only models.

The models are the same classes as in train_stage2.py / train_stage3.py. Only the list of
lead times changes: instead of 7 (stage 2) or 30 (stage 3) sparse horizons, the decoder
walks 10min + every hour to 72h (config.DENSE_HORIZONS, 73 outputs). The output head is
already applied once per decoder step, so it needs no change; the decoder simply gets a
longer UTide sequence. Hyperparameters are the sparse models' own, so a sparse-vs-dense
comparison isolates the change of decoder grid.

    python models/train_dense.py
    # or: python models/run_gauge.py --pipeline dense --gauge richmond --gauge southend

Outputs (outputs/dense/ for Southend, outputs/<gauge>/dense/ otherwise):
    dense_metrics.csv              RMSE/MAE at all 73 lead times, dense models + UTide
    dense_vs_sparse_matched.csv    dense vs the existing sparse model, on the anchors both
                                   test sets share, at the shared lead times
    dense_72h_example.csv          one 72h forecast issued in Feb 2022 (for plot_dense.py)
    dense_72h_example_observed.csv the observed + UTide 10-min series around it
    dense_test_predictions_*.npz   every test forecast (metres) + its anchor time
    dense_history.json             training curves
Weights: models/checkpoints[/<gauge>]/dense/ (tracked, like the main checkpoints).

TIDAL_SMOKE=1 trains one epoch on 4096 windows and writes everything under *_smoke/ dirs --
a mechanics check only.
"""
# %% Bootstrap
import pathlib
import sys

_start = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / '_')).resolve()
REPO_ROOT = next(p for p in _start.parents if (p / 'utils' / 'paths.py').exists())
sys.path.insert(0, str(REPO_ROOT))

# %% Imports
import gc
import time

import numpy as np
import pandas as pd
import torch

from utils import paths
from utils.models import config as C
from utils.models import datasets, training, windowing
from utils.models.architectures import TidalSeq2Seq, TidalSeq2SeqTideOnly

# %% ------------------------------------------------------------------ CONFIG
GAUGE = C.from_env('TIDAL_GAUGE', 'southend')    # run_gauge.py sets it
SMOKE = C.from_env('TIDAL_SMOKE', '0').strip().lower() not in ('', '0', 'false')

LOOKBACK = C.LOOKBACK
TRAIN_STRIDE = C.TRAIN_STRIDE
EPOCHS = 1 if SMOKE else 60
PATIENCE = 10
SMOKE_WINDOWS = 4096

# stage 2's fixed settings (train_stage2.py)
S2_LR, S2_BATCH = 1e-3, 512

DENSE_HORIZONS = C.DENSE_HORIZONS
DENSE_NAMES = list(DENSE_HORIZONS)
DENSE_STEPS = list(DENSE_HORIZONS.values())
S3_HORIZONS, S3_DEC_STEPS, S3_TARGET_SLICE = C.dense_stage3_decoder()
S3_NAMES = list(S3_HORIZONS)
TAIL_DAYS = list(C.DENSE_TAIL_DAYS)

SHARED_LEADS = ['10min', '1h', '6h', '24h', '48h', '72h']   # on both the sparse and dense grids
EXAMPLE_ISSUE = pd.Timestamp('2022-02-17 12:00')            # Storm Eunice surge build-up
EXAMPLE_WINDOW = ('2022-02-01', '2022-03-01')                # fall back to the nearest anchor in it

LOAD_FROM_CHECKPOINT = True

suffix = '_smoke' if SMOKE else ''
OUTPUT_DIR = paths.ensure_dir(str(paths.dense_output_dir(GAUGE)) + suffix)
CHECKPOINT_DIR = paths.ensure_dir(str(paths.dense_checkpoint_dir(GAUGE)) + suffix)
SPARSE_CKPT_DIR = paths.checkpoint_dir(GAUGE)

print(f"gauge: {C.gauge_info(GAUGE)['label']}{'   [SMOKE RUN]' if SMOKE else ''}")
print(f"dense lead times: {len(DENSE_NAMES)} ({DENSE_NAMES[0]}, {DENSE_NAMES[1]}, ... {DENSE_NAMES[-1]}); "
      f"stage-3 decoder: {len(S3_DEC_STEPS)} points, tail days: {TAIL_DAYS or 'none'}")
DEVICE = C.describe_device()
C.set_seed()

predict_s2 = lambda model, batch: model(batch['X'], batch['Ydec'])[0]
predict_tideonly = lambda model, batch: model(batch['X'], batch['Ydec'])[0]
histories = {}
metric_rows, matched_rows = [], []
test_preds = {}


def maybe_subset(tensors):
    if not SMOKE:
        return tensors
    return {k: v[:SMOKE_WINDOWS] for k, v in tensors.items()}


def anchors_of(time_matrix, first_step):
    """Anchor (issue) time of each window = first target time minus its lead."""
    return time_matrix[:, 0] - np.timedelta64(int(first_step) * 10, 'm')


def add_metrics(label, preds, Y, Flag, names):
    rmse, mae = training.masked_rmse_mae(preds, Y, Flag)
    for j, h in enumerate(names):
        metric_rows.append({'model': label, 'horizon': h, 'lead_h': DENSE_HORIZONS.get(h, S3_HORIZONS.get(h)) / 6,
                            'rmse_m': float(rmse[j]), 'mae_m': float(mae[j])})
    return rmse


def matched(label_dense, label_sparse, dense, sparse):
    """Score dense and sparse forecasts on the anchors both test sets contain, at the
    lead times both predict. Each dict: preds (N,H) metres, Y, Flag, anchors, names."""
    common, i_d, i_s = np.intersect1d(dense['anchors'], sparse['anchors'], return_indices=True)
    print(f"matched anchors: {len(common):,} (dense {len(dense['anchors']):,}, sparse {len(sparse['anchors']):,})")
    for lab, src, idx in [(label_dense, dense, i_d), (label_sparse, sparse, i_s),
                          ('UTide standalone', dense, i_d)]:
        cols = [src['names'].index(h) for h in SHARED_LEADS]
        p = (src['utide'] if lab == 'UTide standalone' else src['preds'])[idx][:, cols]
        rmse, mae = training.masked_rmse_mae(p, src['Y'][idx][:, cols], src['Flag'][idx][:, cols])
        for j, h in enumerate(SHARED_LEADS):
            matched_rows.append({'comparison': f'{label_dense} vs sparse', 'model': lab, 'horizon': h,
                                 'rmse_m': float(rmse[j]), 'mae_m': float(mae[j]), 'n_anchors': len(common)})
    return common


# %% ======================== Stage 2, dense =================================
t0 = time.time()
data = datasets.prepare_stage2(DEVICE, DENSE_STEPS, lookback=LOOKBACK, train_stride=TRAIN_STRIDE, gauge=GAUGE)
print(f"dense stage-2 tensors: X {tuple(data.train_tensors['X'].shape)}, "
      f"Ydec {tuple(data.train_tensors['Ydec'].shape)}, Y {tuple(data.train_tensors['Y'].shape)}")
scaler = (data.train_mean, data.train_std)

ckpt_s2 = CHECKPOINT_DIR / 'stage2_dense_seq2seq.pt'
C.set_seed()
s2 = TidalSeq2Seq().to(DEVICE)
print(f"TidalSeq2Seq parameters: {sum(p.numel() for p in s2.parameters()):,} (same class as sparse stage 2)")
s2, histories['Dense stage 2'] = training.load_or_train(
    s2, ckpt_s2,
    lambda m: training.train_model(m, predict_s2, maybe_subset(data.train_tensors), maybe_subset(data.val_tensors),
                                   epochs=EPOCHS, batch_size=S2_BATCH, lr=S2_LR, patience=PATIENCE,
                                   label='dense-stage2', device=DEVICE, resume_path=training.partial_path(ckpt_s2)),
    label='dense-stage2', load_from_checkpoint=LOAD_FROM_CHECKPOINT, device=DEVICE)
training.save_checkpoint({
    'model_state_dict': s2.state_dict(),
    'config': {'model': 'TidalSeq2Seq', 'horizons': DENSE_HORIZONS, 'lookback': LOOKBACK,
               'lr': S2_LR, 'batch_size': S2_BATCH, 'weight_decay': 0.0},
    'scaler': {'mean': data.train_mean, 'std': data.train_std},
}, ckpt_s2, LOAD_FROM_CHECKPOINT)

_, _, preds_s2 = training.compute_metrics(s2, predict_s2, data.test_tensors, data.Y_test, data.Flag_test, *scaler)
add_metrics('Dense stage 2', preds_s2, data.Y_test, data.Flag_test, DENSE_NAMES)
add_metrics('UTide standalone', data.Ydec_test, data.Y_test, data.Flag_test, DENSE_NAMES)
anchors_s2 = anchors_of(data.Time_test, DENSE_STEPS[0])
dense2 = {'preds': preds_s2, 'utide': data.Ydec_test, 'Y': data.Y_test, 'Flag': data.Flag_test,
          'anchors': anchors_s2, 'names': DENSE_NAMES}
test_preds['stage2'] = dense2

# sparse stage 2 on its own test windows, rebuilt from the same (identically fitted) test split
sp_steps = list(C.STAGE2_HORIZONS.values())
Xs, Yds, Ys, Fs, Ts = windowing.build_windows_multi(data.test_df, LOOKBACK, sp_steps)
sc = lambda a: (a - data.train_mean) / data.train_std
sparse_s2_model = TidalSeq2Seq().to(DEVICE)
sparse_s2_model, _ = training.load_checkpoint_or_raise(sparse_s2_model, SPARSE_CKPT_DIR / 'stage2_tidal_seq2seq.pt',
                                                       'sparse-stage2', device=DEVICE)
_, _, preds_sp2 = training.compute_metrics(
    sparse_s2_model, predict_s2, {'X': datasets.to_tensor(sc(Xs), DEVICE), 'Ydec': datasets.to_tensor(sc(Yds), DEVICE)},
    Ys, Fs, *scaler)
sparse2 = {'preds': preds_sp2, 'utide': Yds, 'Y': Ys, 'Flag': Fs,
           'anchors': anchors_of(Ts, sp_steps[0]), 'names': list(C.STAGE2_HORIZONS)}
matched('Dense stage 2', 'Sparse stage 2', dense2, sparse2)

# observed + UTide 10-min series for the example forecast (kept while test_df is here)
obs_series = data.test_df[['DateTime', 'Observed_ODN', 'utide_recon', 'is_flagged']].copy()
print(f"stage 2 done in {(time.time() - t0) / 60:.1f} min")
del data, Xs, Yds, Ys, Fs, Ts
gc.collect()

# %% ======================== Stage 3 tide-only, dense =======================
t0 = time.time()
data3 = datasets.prepare_stage3(DEVICE, list(S3_HORIZONS.values()), S3_DEC_STEPS, S3_TARGET_SLICE,
                                lookback=LOOKBACK, train_stride=TRAIN_STRIDE, gauge=GAUGE)
print(f"dense stage-3 tide-only tensors: X {tuple(data3.train_tensors_tideonly['X'].shape)}, "
      f"Ydec {tuple(data3.train_tensors_tideonly['Ydec'].shape)}, Y {tuple(data3.train_tensors_tideonly['Y'].shape)}")
scaler3 = (data3.train_mean, data3.train_std)

sparse_cfg = torch.load(SPARSE_CKPT_DIR / 'stage3_tidal_seq2seq_tideonly.pt', map_location='cpu',
                        weights_only=False)['config']
S3_LR, S3_WD, S3_BATCH = sparse_cfg['lr'], sparse_cfg['weight_decay'], int(sparse_cfg['batch_size'])
print(f"stage-3 tide-only hyperparameters (from the sparse checkpoint): lr={S3_LR:.2e} wd={S3_WD:.0e} batch={S3_BATCH}")
weights = C.dense_horizon_weights(len(DENSE_HORIZONS), len(TAIL_DAYS))
weights = None if weights is None else weights.to(DEVICE)

ckpt_s3 = CHECKPOINT_DIR / 'stage3_dense_tideonly.pt'
C.set_seed()
s3 = TidalSeq2SeqTideOnly(len(S3_DEC_STEPS), S3_TARGET_SLICE).to(DEVICE)
print(f"TidalSeq2SeqTideOnly parameters: {sum(p.numel() for p in s3.parameters()):,} (same class as sparse stage 3)")
s3, histories['Dense stage 3 (tide-only)'] = training.load_or_train(
    s3, ckpt_s3,
    lambda m: training.train_model(m, predict_tideonly, maybe_subset(data3.train_tensors_tideonly),
                                   maybe_subset(data3.val_tensors_tideonly), epochs=EPOCHS, batch_size=S3_BATCH,
                                   lr=S3_LR, patience=PATIENCE, weight_decay=S3_WD, horizon_weights=weights,
                                   label='dense-stage3-tideonly', device=DEVICE,
                                   resume_path=training.partial_path(ckpt_s3)),
    label='dense-stage3-tideonly', load_from_checkpoint=LOAD_FROM_CHECKPOINT, device=DEVICE)
training.save_checkpoint({
    'model_state_dict': s3.state_dict(),
    'config': {'model': 'TidalSeq2SeqTideOnly', 'horizons': S3_HORIZONS, 'decoder_seq_steps': S3_DEC_STEPS,
               'target_slice_in_decoder': [S3_TARGET_SLICE.start, S3_TARGET_SLICE.stop],
               'tail_days': TAIL_DAYS, 'tail_loss_share': C.DENSE_TAIL_LOSS_SHARE if TAIL_DAYS else None,
               'lookback': LOOKBACK, 'lr': S3_LR, 'weight_decay': S3_WD, 'batch_size': S3_BATCH},
    'scaler': {'mean': data3.train_mean, 'std': data3.train_std},
}, ckpt_s3, LOAD_FROM_CHECKPOINT)

_, _, preds_s3 = training.compute_metrics(s3, predict_tideonly, data3.test_tensors_tideonly,
                                          data3.Y_test, data3.Flag_test, *scaler3)
add_metrics('Dense stage 3 (tide-only)', preds_s3, data3.Y_test, data3.Flag_test, S3_NAMES)
dense3 = {'preds': preds_s3, 'utide': data3.Ydec_targets_test, 'Y': data3.Y_test, 'Flag': data3.Flag_test,
          'anchors': data3.Anchor_test, 'names': S3_NAMES}
test_preds['stage3_tideonly'] = dense3

# sparse stage-3 tide-only on its own test windows
sp3_steps = list(C.STAGE3_HORIZONS.values())
X3, _, _, Yd3, Y3, F3, _, A3 = windowing.build_windows_stage3(
    data3.test_df, LOOKBACK, sp3_steps, data3.grids, data3.utide_full, data3.utide_start, C.DECODER_SEQ_STEPS)
sc3 = lambda a: (a - data3.train_mean) / data3.train_std
sparse_s3_model = TidalSeq2SeqTideOnly(C.DECODER_SEQ_LEN, C.TARGET_SLICE_IN_DECODER).to(DEVICE)
sparse_s3_model, _ = training.load_checkpoint_or_raise(
    sparse_s3_model, SPARSE_CKPT_DIR / 'stage3_tidal_seq2seq_tideonly.pt', 'sparse-stage3-tideonly', device=DEVICE)
_, _, preds_sp3 = training.compute_metrics(
    sparse_s3_model, predict_tideonly, {'X': datasets.to_tensor(sc3(X3), DEVICE), 'Ydec': datasets.to_tensor(sc3(Yd3), DEVICE)},
    Y3, F3, *scaler3)
sparse3 = {'preds': preds_sp3, 'utide': Yd3[:, C.TARGET_SLICE_IN_DECODER], 'Y': Y3, 'Flag': F3,
           'anchors': A3, 'names': list(C.STAGE3_HORIZONS)}
matched('Dense stage 3 (tide-only)', 'Sparse stage 3 (tide-only)', dense3, sparse3)
print(f"stage 3 done in {(time.time() - t0) / 60:.1f} min")

# %% Tables ------------------------------------------------------------------
metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(OUTPUT_DIR / 'dense_metrics.csv', index=False)
matched_df = pd.DataFrame(matched_rows)
matched_df.to_csv(OUTPUT_DIR / 'dense_vs_sparse_matched.csv', index=False)
print(f"saved {OUTPUT_DIR / 'dense_metrics.csv'}\nsaved {OUTPUT_DIR / 'dense_vs_sparse_matched.csv'}")
print("\nmatched-anchor RMSE (m):")
print(matched_df.pivot_table(index=['comparison', 'model'], columns='horizon', values='rmse_m')[SHARED_LEADS].round(4))

for key, d in test_preds.items():
    np.savez_compressed(OUTPUT_DIR / f'dense_test_predictions_{key}.npz', preds=d['preds'].astype(np.float32),
                        anchors=d['anchors'], lead_names=np.array(d['names']))
training.save_history(histories, str(OUTPUT_DIR / 'dense_history.json'))

# %% One 72h forecast issued in Feb 2022, for plot_dense.py -------------------
lo, hi = np.datetime64(EXAMPLE_WINDOW[0]), np.datetime64(EXAMPLE_WINDOW[1])
pool = np.intersect1d(np.intersect1d(dense2['anchors'], dense3['anchors']), sparse2['anchors'])
in_window = pool[(pool >= lo) & (pool < hi)]
if len(in_window) == 0:
    in_window = pool[(pool >= np.datetime64('2022-01-01')) & (pool < np.datetime64('2023-01-01'))]
if len(in_window) == 0:
    raise RuntimeError("no anchor in 2022 is valid for all three models")
issue = in_window[np.argmin(np.abs(in_window - np.datetime64(EXAMPLE_ISSUE)))]
print(f"\nexample forecast issued at {pd.Timestamp(issue)} (requested {EXAMPLE_ISSUE})")

rows = []
for label, d in [('Dense stage 2', dense2), ('Dense stage 3 (tide-only)', dense3), ('Sparse stage 2', sparse2)]:
    i = int(np.where(d['anchors'] == issue)[0][0])
    steps = DENSE_HORIZONS if d is dense2 else (S3_HORIZONS if d is dense3 else C.STAGE2_HORIZONS)
    for j, h in enumerate(d['names']):
        if steps[h] > 432:
            continue
        rows.append({'model': label, 'issue_time': pd.Timestamp(issue), 'horizon': h, 'lead_h': steps[h] / 6,
                     'valid_time': pd.Timestamp(issue) + pd.Timedelta(minutes=10 * steps[h]),
                     'forecast_m': float(d['preds'][i, j])})
pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'dense_72h_example.csv', index=False)
win = obs_series[(obs_series['DateTime'] >= pd.Timestamp(issue) - pd.Timedelta(hours=24))
                 & (obs_series['DateTime'] <= pd.Timestamp(issue) + pd.Timedelta(hours=73))]
win.to_csv(OUTPUT_DIR / 'dense_72h_example_observed.csv', index=False)
print(f"saved {OUTPUT_DIR / 'dense_72h_example.csv'} and the observed series around it")

# %% Done
print(f"\nAll dense outputs are under: {OUTPUT_DIR}")
