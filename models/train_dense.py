"""Dense pipeline -- continuous 0-72h forecasts from the stage 2 and stage 3 tide-only models.

The models are the same classes as in train_stage2.py / train_stage3.py. Only the list of
lead times changes: instead of 7 (stage 2) or 30 (stage 3) sparse horizons, the decoder
walks 10min + every hour to 72h (config.DENSE_HORIZONS, 73 outputs). The output head is
already applied once per decoder step, so it needs no change; the decoder simply gets a
longer UTide sequence. Hyperparameters are the sparse models' own, so a sparse-vs-dense
comparison isolates the change of decoder grid.

    python models/train_dense.py                      # validation scoring only
    python models/train_dense.py --score-test         # also score the test set (final numbers only)
    # or: python models/run_gauge.py --pipeline dense [--variant dense_v2_skip] [--score-test] --gauge ...

Scoring. By default only the VALIDATION split (2018-2020) is scored, so development runs
never show test numbers. --score-test (env TIDAL_SCORE_TEST=1) additionally scores the
test split and writes the test tables below; use it only for the final numbers of a
finished configuration, and record it in CHANGELOG.md. The legacy variant ('dense') is
score-only and cannot be test-scored again: its published test tables are kept as they are.

Outputs (outputs/[<gauge>/]<variant>/; the legacy variant is 'dense'):
  always
    dense_val_metrics.csv              RMSE/MAE at all 73 lead times on validation, dense models + UTide
    dense_val_vs_sparse_matched.csv    dense vs the sparse model, on the validation anchors both share
    dense_val_predictions_*.npz        every validation forecast (metres) + its anchor time
    dense_history.json                 training curves (only when this run trained something)
  with --score-test
    dense_metrics.csv                  as above, on test (the file plot_dense / eval_baselines read)
    dense_vs_sparse_matched.csv
    dense_test_predictions_*.npz
    dense_72h_example.csv              one 72h forecast issued in Feb 2022 (for plot_dense.py)
    dense_72h_example_observed.csv     the observed + UTide 10-min series around it
Weights: models/checkpoints/[<gauge>/]<variant>/ (tracked, like the main checkpoints).

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
SCORE_TEST = ('--score-test' in sys.argv[1:]
              or C.from_env('TIDAL_SCORE_TEST', '0').strip().lower() not in ('', '0', 'false'))

LOOKBACK = C.LOOKBACK
TRAIN_STRIDE = C.TRAIN_STRIDE
# TIDAL_SANITY_EPOCHS=N: the full training set for N epochs only, into *_sanityN/ dirs (a quick
# check that a variant learns, with real per-epoch timings; never the result of record)
SANITY_EPOCHS = C.from_env('TIDAL_SANITY_EPOCHS', None, int)
EPOCHS = 1 if SMOKE else (SANITY_EPOCHS or 60)
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

# A variant only ever loads (a finished model) or resumes (an interrupted one) from its
# OWN checkpoint directory, so a new variant always trains from scratch. The legacy
# variant is score-only: its checkpoints must exist and it is never retrained.
VARIANT = C.dense_variant()
LEGACY = VARIANT == C.DENSE_LEGACY_VARIANT
FLAGS = C.dense_variant_flags(VARIANT)
SKIP, PAD, WLOSS = FLAGS['residual_skip'], FLAGS['decoder_pad_steps'], FLAGS['persistence_loss_weights']
LOAD_FROM_CHECKPOINT = True

if SANITY_EPOCHS and (LEGACY or SCORE_TEST):
    raise SystemExit('TIDAL_SANITY_EPOCHS is for new variants on validation only')

if LEGACY and SCORE_TEST:
    raise SystemExit("the legacy dense model's test tables are already published in "
                     f"{paths.dense_output_dir(GAUGE)}; --score-test is refused so they are never rewritten. "
                     "Test-score a new --variant instead.")

SPLITS = ['val', 'test'] if SCORE_TEST else ['val']
FILE_TAG = {'val': 'dense_val_', 'test': 'dense_'}   # test keeps the original filenames

suffix = '_smoke' if SMOKE else (f'_sanity{SANITY_EPOCHS}' if SANITY_EPOCHS else '')
OUTPUT_DIR = paths.ensure_dir(str(paths.dense_output_dir(GAUGE, VARIANT)) + suffix)
CHECKPOINT_DIR = paths.ensure_dir(str(paths.dense_checkpoint_dir(GAUGE, VARIANT)) + suffix)
SPARSE_CKPT_DIR = paths.checkpoint_dir(GAUGE)

print(f"gauge: {C.gauge_info(GAUGE)['label']}{'   [SMOKE RUN]' if SMOKE else ''}")
print(f"variant: {VARIANT}{'  (legacy: score only, never retrained)' if LEGACY else ''}\n"
      f"  outputs:     {OUTPUT_DIR}\n  checkpoints: {CHECKPOINT_DIR}\n  flags: {FLAGS}"
      f"{f'   [SANITY: {SANITY_EPOCHS} epochs]' if SANITY_EPOCHS else ''}")
print(f"scoring: {' + '.join(SPLITS)}{'   [TEST SCORING ON: final numbers only]' if SCORE_TEST else ''}")
print(f"dense lead times: {len(DENSE_NAMES)} ({DENSE_NAMES[0]}, {DENSE_NAMES[1]}, ... {DENSE_NAMES[-1]}); "
      f"stage-3 decoder: {len(S3_DEC_STEPS)} points, tail days: {TAIL_DAYS or 'none'}")
DEVICE = C.describe_device()
C.set_seed()

predict_s2 = lambda model, batch: model(batch['X'], batch['Ydec'])[0]           # sparse / legacy call
predict_dense2 = ((lambda model, batch: model(batch['X'], batch['Ydec'], batch['Ua'])[0]) if SKIP
                  else predict_s2)
predict_tideonly = lambda model, batch: model(batch['X'], batch['Ydec'])[0]
histories = {}
metric_rows = {s: [] for s in SPLITS}
matched_rows = {s: [] for s in SPLITS}
preds_by_split = {s: {} for s in SPLITS}     # key -> {'preds', 'anchors', 'names', ...}
sparse_by_split = {s: {} for s in SPLITS}


def peak_ram_gb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return getattr(mi, 'peak_wset', mi.rss) / 1e9
    except ImportError:
        return float('nan')


def persistence_metres(Yd, X_last, Ua):
    """Residual persistence in metres: UTide(t+k) + obs(t) - UTide(t)."""
    return Yd + (X_last - Ua)[:, None]


def require_legacy_checkpoint(path):
    """The legacy model is the comparison baseline: refuse to (re)train it."""
    if LEGACY and not path.exists():
        raise FileNotFoundError(f"legacy dense checkpoint {path} is missing; the legacy model is score-only "
                                f"and is never retrained. Pass a new --variant to train.")


def maybe_subset(tensors):
    if not SMOKE:
        return tensors
    return {k: v[:SMOKE_WINDOWS] for k, v in tensors.items()}


def anchors_of(time_matrix, first_step):
    """Anchor (issue) time of each window = first target time minus its lead."""
    return time_matrix[:, 0] - np.timedelta64(int(first_step) * 10, 'm')


def add_metrics(split, label, preds, Y, Flag, names):
    rmse, mae = training.masked_rmse_mae(preds, Y, Flag)
    for j, h in enumerate(names):
        metric_rows[split].append({'split': split, 'model': label, 'horizon': h,
                                   'lead_h': DENSE_HORIZONS.get(h, S3_HORIZONS.get(h)) / 6,
                                   'rmse_m': float(rmse[j]), 'mae_m': float(mae[j])})
    return rmse


def matched(split, label_dense, label_sparse, dense, sparse):
    """Score dense and sparse forecasts on the anchors both window sets of `split` contain,
    at the lead times both predict. Each dict: preds (N,H) metres, Y, Flag, anchors, names."""
    common, i_d, i_s = np.intersect1d(dense['anchors'], sparse['anchors'], return_indices=True)
    print(f"[{split}] matched anchors: {len(common):,} (dense {len(dense['anchors']):,}, "
          f"sparse {len(sparse['anchors']):,})")
    for lab, src, idx in [(label_dense, dense, i_d), (label_sparse, sparse, i_s),
                          ('UTide standalone', dense, i_d)]:
        cols = [src['names'].index(h) for h in SHARED_LEADS]
        p = (src['utide'] if lab == 'UTide standalone' else src['preds'])[idx][:, cols]
        rmse, mae = training.masked_rmse_mae(p, src['Y'][idx][:, cols], src['Flag'][idx][:, cols])
        for j, h in enumerate(SHARED_LEADS):
            matched_rows[split].append({'split': split, 'comparison': f'{label_dense} vs sparse', 'model': lab,
                                        'horizon': h, 'rmse_m': float(rmse[j]), 'mae_m': float(mae[j]),
                                        'n_anchors': len(common)})
    return common


# %% ======================== Stage 2, dense =================================
t0 = time.time()
data = datasets.prepare_stage2(DEVICE, DENSE_STEPS, lookback=LOOKBACK, train_stride=TRAIN_STRIDE, gauge=GAUGE,
                               with_anchor_utide=True, decoder_pad_steps=PAD)
print(f"dense stage-2 tensors: X {tuple(data.train_tensors['X'].shape)}, "
      f"Ydec {tuple(data.train_tensors['Ydec'].shape)}, Y {tuple(data.train_tensors['Y'].shape)}")
scaler = (data.train_mean, data.train_std)
sc = lambda a: (a - data.train_mean) / data.train_std

ckpt_s2 = CHECKPOINT_DIR / 'stage2_dense_seq2seq.pt'
require_legacy_checkpoint(ckpt_s2)
C.set_seed()
s2 = TidalSeq2Seq(residual_skip=SKIP, n_out=len(DENSE_NAMES) if PAD else None).to(DEVICE)
print(f"TidalSeq2Seq parameters: {sum(p.numel() for p in s2.parameters()):,} (same class as sparse stage 2)")

s2_weights = None
if WLOSS:
    # per-lead residual-persistence MSE on the TRAINING windows (scaled units), computed once
    tt = data.train_tensors
    base = tt['Ydec'][:, :len(DENSE_NAMES)] + (tt['X'][:, -1] - tt['Ua']).unsqueeze(-1)
    ok = ~tt['Flag']
    pers_mse = (((base - tt['Y']) ** 2) * ok).sum(0) / ok.sum(0)
    s2_weights = 1.0 / pers_mse
    s2_weights = (s2_weights / s2_weights.mean()).to(DEVICE)
    print(f"persistence loss weights (mean 1): 10min {s2_weights[0]:.2f}, 1h {s2_weights[1]:.2f}, "
          f"24h {s2_weights[24]:.3f}, 72h {s2_weights[-1]:.3f}")

t_train = time.time()
s2, histories['Dense stage 2'] = training.load_or_train(
    s2, ckpt_s2,
    lambda m: training.train_model(m, predict_dense2, maybe_subset(data.train_tensors), maybe_subset(data.val_tensors),
                                   epochs=EPOCHS, batch_size=S2_BATCH, lr=S2_LR, patience=PATIENCE,
                                   horizon_weights=s2_weights,
                                   label='dense-stage2', device=DEVICE, resume_path=training.partial_path(ckpt_s2)),
    label='dense-stage2', load_from_checkpoint=LOAD_FROM_CHECKPOINT, device=DEVICE)
training.save_checkpoint({
    'model_state_dict': s2.state_dict(),
    'config': {'model': 'TidalSeq2Seq', 'variant': VARIANT, 'flags': FLAGS, 'horizons': DENSE_HORIZONS,
               'lookback': LOOKBACK, 'lr': S2_LR, 'batch_size': S2_BATCH, 'weight_decay': 0.0,
               'residual_skip': SKIP, 'decoder_pad_steps': PAD, 'n_out': len(DENSE_NAMES) if PAD else None},
    'scaler': {'mean': data.train_mean, 'std': data.train_std},
}, ckpt_s2, LOAD_FROM_CHECKPOINT)

print(f"stage 2 training wall time {(time.time() - t_train) / 60:.1f} min; peak RAM so far {peak_ram_gb():.1f} GB")
sparse_s2_model = TidalSeq2Seq().to(DEVICE)
sparse_s2_model, _ = training.load_checkpoint_or_raise(sparse_s2_model, SPARSE_CKPT_DIR / 'stage2_tidal_seq2seq.pt',
                                                       'sparse-stage2', device=DEVICE)
sp_steps = list(C.STAGE2_HORIZONS.values())
obs_series = None

for split in SPLITS:
    if split == 'test':
        tensors, Y, Flag, T, Yd, split_df = (data.test_tensors, data.Y_test, data.Flag_test, data.Time_test,
                                             data.Ydec_test, data.test_df)
        X_last, Ua = data.X_test[:, -1], data.Ua_test
    else:
        # validation windows rebuilt raw (unscaled) from the same split prepare_stage2 used
        X, Yd, Y, Flag, T, Ua = windowing.build_windows_multi(data.val_df, LOOKBACK, DENSE_STEPS,
                                                              return_anchor_utide=True)
        Yd_in = Yd if not PAD else np.concatenate(
            [Yd, datasets.pad_utide(T, DENSE_STEPS, PAD, data.utide_full, data.utide_start)[0]], axis=1)
        tensors = {'X': datasets.to_tensor(sc(X), DEVICE), 'Ydec': datasets.to_tensor(sc(Yd_in), DEVICE)}
        if SKIP:
            tensors['Ua'] = datasets.to_tensor(sc(Ua), DEVICE)
        split_df = data.val_df
        X_last = X[:, -1]
    _, _, preds = training.compute_metrics(s2, predict_dense2, tensors, Y, Flag, *scaler)
    add_metrics(split, 'Dense stage 2', preds, Y, Flag, DENSE_NAMES)
    add_metrics(split, 'UTide standalone', Yd, Y, Flag, DENSE_NAMES)
    add_metrics(split, 'Residual persistence', persistence_metres(Yd, X_last, Ua), Y, Flag, DENSE_NAMES)
    dense2 = {'preds': preds, 'utide': Yd, 'Y': Y, 'Flag': Flag,
              'anchors': anchors_of(T, DENSE_STEPS[0]), 'names': DENSE_NAMES}
    preds_by_split[split]['stage2'] = dense2

    # sparse stage 2 on its own windows of the same split (identically fitted UTide)
    Xs, Yds, Ys, Fs, Ts = windowing.build_windows_multi(split_df, LOOKBACK, sp_steps)
    _, _, preds_sp2 = training.compute_metrics(
        sparse_s2_model, predict_s2,
        {'X': datasets.to_tensor(sc(Xs), DEVICE), 'Ydec': datasets.to_tensor(sc(Yds), DEVICE)}, Ys, Fs, *scaler)
    sparse2 = {'preds': preds_sp2, 'utide': Yds, 'Y': Ys, 'Flag': Fs,
               'anchors': anchors_of(Ts, sp_steps[0]), 'names': list(C.STAGE2_HORIZONS)}
    sparse_by_split[split]['stage2'] = sparse2
    matched(split, 'Dense stage 2', 'Sparse stage 2', dense2, sparse2)
    if split == 'test':   # observed + UTide 10-min series for the example forecast
        obs_series = data.test_df[['DateTime', 'Observed_ODN', 'utide_recon', 'is_flagged']].copy()
    del tensors, Xs, Yds, Ys, Fs, Ts

print(f"stage 2 done in {(time.time() - t0) / 60:.1f} min")
del data
gc.collect()

# %% ======================== Stage 3 tide-only, dense =======================
if not FLAGS['train_stage3']:
    print(f"variant {VARIANT} changes stage 2 only: dense stage 3 tide-only is not retrained "
          f"(the legacy model in {paths.dense_output_dir(GAUGE)} is its comparison)")
else:
    t0 = time.time()
    data3 = datasets.prepare_stage3(DEVICE, list(S3_HORIZONS.values()), S3_DEC_STEPS, S3_TARGET_SLICE,
                                    lookback=LOOKBACK, train_stride=TRAIN_STRIDE, gauge=GAUGE)
    print(f"dense stage-3 tide-only tensors: X {tuple(data3.train_tensors_tideonly['X'].shape)}, "
          f"Ydec {tuple(data3.train_tensors_tideonly['Ydec'].shape)}, Y {tuple(data3.train_tensors_tideonly['Y'].shape)}")
    scaler3 = (data3.train_mean, data3.train_std)
    sc3 = lambda a: (a - data3.train_mean) / data3.train_std

    sparse_cfg = torch.load(SPARSE_CKPT_DIR / 'stage3_tidal_seq2seq_tideonly.pt', map_location='cpu',
                            weights_only=False)['config']
    S3_LR, S3_WD, S3_BATCH = sparse_cfg['lr'], sparse_cfg['weight_decay'], int(sparse_cfg['batch_size'])
    print(f"stage-3 tide-only hyperparameters (from the sparse checkpoint): lr={S3_LR:.2e} wd={S3_WD:.0e} batch={S3_BATCH}")
    weights = C.dense_horizon_weights(len(DENSE_HORIZONS), len(TAIL_DAYS))
    weights = None if weights is None else weights.to(DEVICE)

    ckpt_s3 = CHECKPOINT_DIR / 'stage3_dense_tideonly.pt'
    require_legacy_checkpoint(ckpt_s3)
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
        'config': {'model': 'TidalSeq2SeqTideOnly', 'variant': VARIANT, 'horizons': S3_HORIZONS,
                   'decoder_seq_steps': S3_DEC_STEPS,
                   'target_slice_in_decoder': [S3_TARGET_SLICE.start, S3_TARGET_SLICE.stop],
                   'tail_days': TAIL_DAYS, 'tail_loss_share': C.DENSE_TAIL_LOSS_SHARE if TAIL_DAYS else None,
                   'lookback': LOOKBACK, 'lr': S3_LR, 'weight_decay': S3_WD, 'batch_size': S3_BATCH},
        'scaler': {'mean': data3.train_mean, 'std': data3.train_std},
    }, ckpt_s3, LOAD_FROM_CHECKPOINT)

    sparse_s3_model = TidalSeq2SeqTideOnly(C.DECODER_SEQ_LEN, C.TARGET_SLICE_IN_DECODER).to(DEVICE)
    sparse_s3_model, _ = training.load_checkpoint_or_raise(
        sparse_s3_model, SPARSE_CKPT_DIR / 'stage3_tidal_seq2seq_tideonly.pt', 'sparse-stage3-tideonly', device=DEVICE)
    sp3_steps = list(C.STAGE3_HORIZONS.values())

    for split in SPLITS:
        if split == 'test':
            tensors, Y, Flag, A, Yd_t, split_df = (data3.test_tensors_tideonly, data3.Y_test, data3.Flag_test,
                                                   data3.Anchor_test, data3.Ydec_targets_test, data3.test_df)
        else:
            X, _, _, Yd, Y, Flag, _, A = windowing.build_windows_stage3(
                data3.val_df, LOOKBACK, list(S3_HORIZONS.values()), data3.grids, data3.utide_full, data3.utide_start,
                S3_DEC_STEPS)
            tensors = {'X': datasets.to_tensor(sc3(X), DEVICE), 'Ydec': datasets.to_tensor(sc3(Yd), DEVICE)}
            Yd_t = Yd[:, S3_TARGET_SLICE]
            split_df = data3.val_df
        _, _, preds = training.compute_metrics(s3, predict_tideonly, tensors, Y, Flag, *scaler3)
        add_metrics(split, 'Dense stage 3 (tide-only)', preds, Y, Flag, S3_NAMES)
        dense3 = {'preds': preds, 'utide': Yd_t, 'Y': Y, 'Flag': Flag, 'anchors': A, 'names': S3_NAMES}
        preds_by_split[split]['stage3_tideonly'] = dense3

        # sparse stage-3 tide-only on its own windows of the same split
        X3, _, _, Yd3, Y3, F3, _, A3 = windowing.build_windows_stage3(
            split_df, LOOKBACK, sp3_steps, data3.grids, data3.utide_full, data3.utide_start, C.DECODER_SEQ_STEPS)
        _, _, preds_sp3 = training.compute_metrics(
            sparse_s3_model, predict_tideonly,
            {'X': datasets.to_tensor(sc3(X3), DEVICE), 'Ydec': datasets.to_tensor(sc3(Yd3), DEVICE)}, Y3, F3, *scaler3)
        sparse3 = {'preds': preds_sp3, 'utide': Yd3[:, C.TARGET_SLICE_IN_DECODER], 'Y': Y3, 'Flag': F3,
                   'anchors': A3, 'names': list(C.STAGE3_HORIZONS)}
        matched(split, 'Dense stage 3 (tide-only)', 'Sparse stage 3 (tide-only)', dense3, sparse3)
        del tensors, X3, Yd3, Y3, F3, A3

    print(f"stage 3 done in {(time.time() - t0) / 60:.1f} min")

# %% Tables ------------------------------------------------------------------
for split in SPLITS:
    tag = FILE_TAG[split]
    metrics_df = pd.DataFrame(metric_rows[split])
    matched_df = pd.DataFrame(matched_rows[split])
    if split == 'test':   # the published test tables never had a split column; keep their format
        metrics_df, matched_df = metrics_df.drop(columns='split'), matched_df.drop(columns='split')
    metrics_df.to_csv(OUTPUT_DIR / f'{tag}metrics.csv', index=False)
    matched_df.to_csv(OUTPUT_DIR / f'{tag}vs_sparse_matched.csv', index=False)
    print(f"saved {OUTPUT_DIR / f'{tag}metrics.csv'}\nsaved {OUTPUT_DIR / f'{tag}vs_sparse_matched.csv'}")
    print(f"\n[{split}] matched-anchor RMSE (m):")
    print(matched_df.pivot_table(index=['comparison', 'model'], columns='horizon', values='rmse_m')[SHARED_LEADS].round(4))

    pred_name = 'dense_val_predictions' if split == 'val' else 'dense_test_predictions'
    for key, d in preds_by_split[split].items():
        np.savez_compressed(OUTPUT_DIR / f'{pred_name}_{key}.npz', preds=d['preds'].astype(np.float32),
                            anchors=d['anchors'], lead_names=np.array(d['names']))
training.save_history(histories, str(OUTPUT_DIR / 'dense_history.json'))

# %% One 72h forecast issued in Feb 2022 (test period), for plot_dense.py ------
if SCORE_TEST:
    dense2, dense3 = preds_by_split['test']['stage2'], preds_by_split['test'].get('stage3_tideonly')
    sparse2 = sparse_by_split['test']['stage2']
    lo, hi = np.datetime64(EXAMPLE_WINDOW[0]), np.datetime64(EXAMPLE_WINDOW[1])
    pool = np.intersect1d(dense2['anchors'], sparse2['anchors'])
    if dense3 is not None:
        pool = np.intersect1d(pool, dense3['anchors'])
    in_window = pool[(pool >= lo) & (pool < hi)]
    if len(in_window) == 0:
        in_window = pool[(pool >= np.datetime64('2022-01-01')) & (pool < np.datetime64('2023-01-01'))]
    if len(in_window) == 0:
        raise RuntimeError("no anchor in 2022 is valid for all three models")
    issue = in_window[np.argmin(np.abs(in_window - np.datetime64(EXAMPLE_ISSUE)))]
    print(f"\nexample forecast issued at {pd.Timestamp(issue)} (requested {EXAMPLE_ISSUE})")

    rows = []
    for label, d in [('Dense stage 2', dense2), ('Dense stage 3 (tide-only)', dense3), ('Sparse stage 2', sparse2)]:
        if d is None:
            continue
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
print(f"\npeak RAM (process working set): {peak_ram_gb():.1f} GB")
print(f"All dense outputs are under: {OUTPUT_DIR}")
