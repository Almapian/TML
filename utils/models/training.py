"""Training loop, masked losses, metrics and checkpoint handling, shared by all stages.

The three stage notebooks each carried a near-identical copy of this loop. The unified
version here takes the dict-of-tensors form stages 2 and 3 used (`predict_fn(model, batch)`
decouples the loop from each model's call signature); stage 1's single-step tensors are
just the degenerate case where every array is 1-D.

Two entry points matter for the scripts:

    load_or_train(...)         used by `models/train_stage*.py` -- trains, or reuses a
                               checkpoint when one already exists
    load_checkpoint_or_raise() used by `models/plot_stage*.py` -- loads weights and never
                               trains, so a plotting run can never silently start a
                               multi-hour training job
"""
import json
import os
from pathlib import Path

import numpy as np
import torch


# ======================================================================================
# Losses and metrics -- flagged (chatter/stuck/imputed) targets are excluded everywhere
# ======================================================================================
def masked_mse(pred, true, flag_b, horizon_weights=None):
    sq_err = (pred - true) ** 2
    if horizon_weights is not None:
        sq_err = sq_err * horizon_weights
    return sq_err[~flag_b].mean()


def masked_rmse_mae(pred, true, flagged):
    """Per-horizon RMSE/MAE for (N, H) arrays; scalar RMSE/MAE for 1-D arrays."""
    pred, true, flagged = np.asarray(pred), np.asarray(true), np.asarray(flagged)
    if pred.ndim == 1:
        mask = ~flagged
        err = pred[mask] - true[mask]
        return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))

    rmses, maes = [], []
    for j in range(pred.shape[1]):
        mask = ~flagged[:, j]
        err = pred[mask, j] - true[mask, j]
        rmses.append(float(np.sqrt(np.mean(err ** 2))))
        maes.append(float(np.mean(np.abs(err))))
    return np.array(rmses), np.array(maes)


def batch_iter(tensors, batch_size, shuffle):
    n = tensors['X'].shape[0]
    idx = torch.randperm(n, device=tensors['X'].device) if shuffle else torch.arange(n, device=tensors['X'].device)
    for start in range(0, n, batch_size):
        b = idx[start:start + batch_size]
        yield {k: v[b] for k, v in tensors.items()}


# ======================================================================================
# Training
# ======================================================================================
def partial_path(checkpoint_path):
    """Where `train_model` keeps its per-epoch resume state for a given final checkpoint.

    Kept under outputs/ (gitignored scratch) rather than next to the tracked checkpoint,
    mirroring the checkpoint's position relative to models/checkpoints/ so two gauges'
    identically named models never share a resume file.
    """
    from .. import paths
    checkpoint_path = Path(checkpoint_path).resolve()
    try:
        rel = checkpoint_path.relative_to(paths.CHECKPOINTS_DIR.resolve())
    except ValueError:
        rel = Path(checkpoint_path.name)
    return paths.OUTPUTS_DIR / '_resume' / rel.with_suffix('.partial.pt')


def _save_atomic(obj, path):
    """torch.save via a temp file, so a crash mid-write can't leave a corrupt resume file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_model(model, predict_fn, train_tensors, val_tensors, epochs, batch_size, lr,
                patience, label, weight_decay=0.0, horizon_weights=None, device=None,
                resume_path=None):
    """Masked-MSE training with early stopping on validation loss and ReduceLROnPlateau.

    With `resume_path` set, the full training state (weights, optimiser, scheduler, best
    weights so far, history, early-stopping counter, RNG) is written there after every
    epoch, and an interrupted run picks up from the last completed epoch. That is what
    makes multi-hour CPU runs safe on a laptop: a sleep, crash or reboot costs one epoch.
    A run that finished is marked as such, so re-running returns its result immediately.

    Returns (model with best-validation weights restored, history dict).
    """
    device = device or next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    use_amp = device.type == 'cuda'

    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}
    start_epoch = 0
    finished = False

    if resume_path is not None and os.path.exists(resume_path):
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['opt'])
        sched.load_state_dict(state['sched'])
        best_val, best_state = state['best_val'], state['best_state']
        epochs_no_improve, history = state['epochs_no_improve'], state['history']
        start_epoch, finished = state['epoch'] + 1, state['finished']
        torch.set_rng_state(state['rng_cpu'])
        if torch.cuda.is_available() and state.get('rng_cuda') is not None:
            torch.cuda.set_rng_state_all(state['rng_cuda'])
        print(f"[{label}] resuming from {resume_path} after epoch {start_epoch} "
              f"(best val_loss so far {best_val:.5f}){' -- run already finished' if finished else ''}")

    def _checkpoint(epoch, done):
        if resume_path is None:
            return
        _save_atomic({
            'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
            'best_val': best_val, 'best_state': best_state, 'epochs_no_improve': epochs_no_improve,
            'history': history, 'epoch': epoch, 'finished': done,
            'rng_cpu': torch.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, resume_path)

    for epoch in range(start_epoch, epochs):
        if finished:
            break
        model.train()
        running_loss, n_seen = 0.0, 0
        for batch in batch_iter(train_tensors, batch_size, shuffle=True):
            opt.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                pred = predict_fn(model, batch)
                loss = masked_mse(pred, batch['Y'], batch['Flag'], horizon_weights)
            loss.backward()
            opt.step()
            n_valid = int((~batch['Flag']).sum())
            running_loss += loss.item() * n_valid
            n_seen += n_valid
        train_loss = running_loss / n_seen

        model.eval()
        val_loss_sum, n_val = 0.0, 0
        with torch.no_grad():
            for batch in batch_iter(val_tensors, batch_size, shuffle=False):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                    pred = predict_fn(model, batch)
                    loss = masked_mse(pred, batch['Y'], batch['Flag'], horizon_weights)
                n_valid = int((~batch['Flag']).sum())
                val_loss_sum += loss.item() * n_valid
                n_val += n_valid
        val_loss = val_loss_sum / n_val

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        sched.step(val_loss)

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"[{label}] epoch {epoch + 1:03d}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}"
              f"  lr={opt.param_groups[0]['lr']:.1e}{'  *' if improved else ''}")

        stop = epochs_no_improve >= patience
        _checkpoint(epoch, done=stop)   # an exhausted epoch budget needs no flag: range() is empty
        if stop:
            print(f"[{label}] early stopping at epoch {epoch + 1} (best val_loss={best_val:.5f})")
            break

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def compute_metrics(model, predict_fn, tensors, y_raw, flagged, mean, std, batch_size=4096):
    """Predict in batches, un-scale to metres, and score against the observed targets."""
    model.eval()
    preds = []
    n = tensors['X'].shape[0]
    for start in range(0, n, batch_size):
        batch = {k: v[start:start + batch_size] for k, v in tensors.items() if k not in ('Y', 'Flag')}
        preds.append(predict_fn(model, batch))
    preds_m = torch.cat(preds).cpu().numpy() * std + mean
    rmse, mae = masked_rmse_mae(preds_m, y_raw, flagged)
    return rmse, mae, preds_m


# ======================================================================================
# Checkpoints
# ======================================================================================
def load_or_train(model, checkpoint_path, train_fn, label, load_from_checkpoint=True, device=None):
    """Load saved weights if `load_from_checkpoint` and a checkpoint exists, else train.

    Architecture hyperparameters are fixed constants in every stage script (only
    lr/weight_decay/batch_size are searched, in stage 3), so re-instantiating the same
    class is enough to make a saved state dict load cleanly.
    """
    if load_from_checkpoint and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device or 'cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        print(f"[{label}] loaded weights from {checkpoint_path}")
        return model, None

    # A forced retrain must not short-circuit on a *finished* resume file left by the
    # previous run; an unfinished one is a retrain that was interrupted, so it is kept.
    resume = partial_path(checkpoint_path)
    if not load_from_checkpoint and resume.exists():
        if torch.load(resume, map_location='cpu', weights_only=False)['finished']:
            os.remove(resume)
    return train_fn(model)


def save_checkpoint(payload, checkpoint_path, load_from_checkpoint=True):
    """torch.save, unless this run merely loaded `checkpoint_path` -- re-saving a loaded
    checkpoint would rewrite a tracked file with identical weights (and, for older
    checkpoints, a different set of keys). A path that didn't exist before this run, or a
    forced retrain, always saves."""
    if load_from_checkpoint and os.path.exists(checkpoint_path):
        print(f"{checkpoint_path} was loaded, not retrained -- left as-is")
        return
    torch.save(payload, checkpoint_path)
    print(f"saved {checkpoint_path}")


def load_checkpoint_or_raise(model, checkpoint_path, label, device=None):
    """Plot-script counterpart of `load_or_train`: loads weights, or fails loudly.

    Deliberately takes no `train_fn` -- a plotting run cannot start training by accident.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"[{label}] no checkpoint at {checkpoint_path}. "
            f"Run the matching train_stage*.py first (or copy the checkpoints down from Colab)."
        )
    ckpt = torch.load(checkpoint_path, map_location=device or 'cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[{label}] loaded weights from {checkpoint_path}")
    return model, ckpt


def save_history(histories, path):
    """Persist training curves so the plotting scripts can redraw loss figures without
    retraining. `histories` maps a display name -> the history dict train_model returned
    (None for models that were themselves loaded from a checkpoint, which are skipped).
    """
    payload = {name: h for name, h in histories.items() if h is not None}
    if not payload:
        print(f"no new training history to save ({path} left as-is)")
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)
    print(f"saved {path}")
    return path


def load_history(path):
    """Read back what `save_history` wrote; returns {} when the file does not exist."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)
