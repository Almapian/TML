"""Constants shared by every modelling stage.

These are the values the stage 1-3 notebooks each declared for themselves; keeping
one copy here is what stops the three stages from silently drifting apart on, say,
the train/validation boundary or the site latitude used for the UTide fit.
"""
import random

import numpy as np
import pandas as pd
import torch

SEED = 42

# --- data geometry ---
STEP = pd.Timedelta('10min')       # native gauge resolution
LOOKBACK = 96                       # 16h of 10-minute steps (> one semi-diurnal cycle)
TRAIN_STRIDE = 3                    # subsample *training* windows only, to cut epoch time
SITE_LAT = 51.5145                  # Southend Pier
M2_STEPS = 74.5                     # ~M2 semi-diurnal period, in 10-minute steps

# --- chronological split (train 2004-2017, validation 2018-2020, test 2021-2024) ---
TRAIN_END = '2018-01-01'
VAL_END = '2021-01-01'

# --- columns read from the gauge CSV ---
GAUGE_USECOLS = ['DateTime', 'Observed_ODN', 'is_chatter_flagged', 'is_stuck_flagged', 'is_imputed']


def get_device():
    """CUDA when available, CPU otherwise. Printed by every script at startup."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def describe_device(device=None):
    device = get_device() if device is None else device
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
        print(f"BF16 supported:  {torch.cuda.is_bf16_supported()}")
    else:
        print("No GPU detected -- running on CPU. Training will be slow; plotting scripts are fine.")
    return device


def set_seed(seed=SEED):
    """Seed python/numpy/torch together, as every stage notebook did before each model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================================================================================
# Forecast horizons
# ======================================================================================
STEPS_PER_DAY = 144  # 24h * 6 ten-minute steps

# Stage 1 scores a single-step model by recursive rollout out to 48h.
STAGE1_ROLLOUT_HORIZONS = {'10min': 1, '1h': 6, '6h': 36, '24h': 144, '48h': 288}
STAGE1_ROLLOUT_STRIDE = 144  # a new rollout start point every 24h across the test set

# Stage 2 predicts these seven directly. Stage 3 keeps them as its "headline" set, so its
# numbers stay directly comparable, and extends daily out to 30 days.
HEADLINE_HORIZONS = {'10min': 1, '1h': 6, '6h': 36, '24h': 144, '48h': 288, '72h': 432, '168h': 1008}
STAGE2_HORIZONS = dict(HEADLINE_HORIZONS)

EXTENDED_DAY_NUMS = list(range(8, 31))  # 8..30 inclusive -> 23 entries
EXTENDED_HORIZONS = {f'{d}d': d * STEPS_PER_DAY for d in EXTENDED_DAY_NUMS}
STAGE3_HORIZONS = {**HEADLINE_HORIZONS, **EXTENDED_HORIZONS}

# Stage 3's decoder input sequence: a leading "now" point, the 30 real (trained) horizons,
# and a small trailing margin past 30d, purely so neither 10min nor 30d sits at the literal
# edge of the BiLSTM sequence.
TRAIL_PAD_DAYS = [31, 32, 33]
DECODER_SEQ_STEPS = [0] + list(STAGE3_HORIZONS.values()) + [d * STEPS_PER_DAY for d in TRAIL_PAD_DAYS]
DECODER_SEQ_LEN = len(DECODER_SEQ_STEPS)
TARGET_SLICE_IN_DECODER = slice(1, 1 + len(STAGE3_HORIZONS))

UTIDE_EXT_DAYS = 35          # UTide reconstruction buffer past the last gauge row
DAILY_GRID_BUFFER_DAYS = 40  # daily meteo grid buffer, same reason

# --- stage 3 branch lookbacks ---
METEO_LOOKBACK_HOURS = 120   # 5 days; EDA/meteo_eda.ipynb s.2.3: pressure ACF < 0.2 by ~121h
RIVER_LOOKBACK_DAYS = 30     # comfortably longer than the ~2-day rainfall -> discharge lag

# --- reduced-training-window ablation (stage 2 s.13.3, stage 3 s.12) ---
REDUCED_TRAIN_STARTS = {'3yr': '2015-01-01', '7yr': '2011-01-01', '10yr': '2008-01-01'}
