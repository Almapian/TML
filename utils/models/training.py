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
def train_model(model, predict_fn, train_tensors, val_tensors, epochs, batch_size, lr,
                patience, label, weight_decay=0.0, horizon_weights=None, device=None):
    """Masked-MSE training with early stopping on validation loss and ReduceLROnPlateau.

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

    for epoch in range(epochs):
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

        if epochs_no_improve >= patience:
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
    return train_fn(model)


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
