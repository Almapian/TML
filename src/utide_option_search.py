"""Search over UTide's constituent set and fitting method, per gauge, before any modelling.

The stage 2/3 models use UTide twice: as the decoder's known-future input and as the
harmonic baseline. Every existing checkpoint used `constit='auto'` (68 constituents) and
`method='ols'`. This script asks whether a different choice fits better, scored the same
way the models are: fit on train (2004-2017), choose on validation (2018-2020), report test
(2021-2024), flagged points excluded everywhere.

    python src/utide_option_search.py            # both gauges
    python src/utide_option_search.py --smoke    # 2-year train slice, quick mechanics check
    python src/utide_option_search.py --plot-only  # redraw the figures from the saved CSVs
    # or run the `# %%` cells interactively in VS Code

The grid is constituent set x method:

* constit: 'auto', or 'auto' plus the shallow-water/compound constituents UTide knows
  but never selects automatically (SW_EXTRA). At Richmond the tide is strongly distorted
  by the shallow upper estuary, which is exactly what higher harmonics describe.
* method: 'ols', or 'robust' (iteratively reweighted least squares) with three weight
  functions. Robust fitting down-weights large residuals -- fluvial floods and surges --
  so the constituents describe the tide rather than the river.

The search runs at fit stride 6 (hourly; equivalent to stride 1 within ~1 mm on
Southend, see config.GAUGES). The validation winner and the previous configuration
(auto/ols) are then refit at stride 1, the stride the checkpoints were trained with, so
the final comparison is like-for-like with the earlier results.

Outputs, per gauge, under outputs/<gauge>/utide_search/:
    utide_option_search.csv      every fit: options, constituents, train/val/test RMSE & MAE
    utide_choice.json            the validation winner (stride 1), for the tuned model runs
    shallow_water_amplitudes.csv winner's amplitudes for the constituents beyond M4
and report_images/utide_option_search_<gauge>.png (test RMSE per option set).

Note: "train RMSE" is in-sample -- the fit scored on the data it was fitted to. It is
reported because a harmonic fit is often quoted that way; it is not comparable to the
validation/test numbers or to the model results.
"""
# %% Bootstrap
import pathlib
import sys

_start = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / '_')).resolve()
REPO_ROOT = next(p for p in _start.parents if (p / 'utils' / 'paths.py').exists())
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
if not hasattr(sys, 'ps1') and 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')

# %% Imports
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import paths
from utils import plot_config  # noqa: F401  (report style)
from utils.plot_config import ANNOTATION_INK
from utils.models import config as C
from utils.models import datasets

# %% ------------------------------------------------------------------ CONFIG
GAUGES = ['richmond', 'southend']
SMOKE = '--smoke' in sys.argv          # 2-year train slice, stride-6 only: checks mechanics
SEARCH_STRIDE = 6                      # fit stride for the grid
CONFIRM_STRIDE = 1                     # stride for the winner + baseline refit (None to skip)

# In UTide's constituent table but outside its 'auto' set (checked against utide 0.3.1).
# M12 (2.07 h) is left out: it sits at the 2 h Nyquist limit of an hourly fit.
SW_EXTRA = ['N4', '3MS4', 'KN4', 'SL4',
            'MNO5', '2MO5', '3MP5', 'MNK5', '2MP5', 'MSK5', '3KM5',
            '2NM6', 'MSN6', 'MKN6', 'NSK6', 'S6',
            'M7', '3MN8', '3MS8', '3MK8', '4MK9', 'M10']

CONSTIT_SETS = ['auto', 'auto+SW']
# 'huber' is not offered: utide 0.3.1's huber() calls max() on an array and always fails.
# 'fair' is the closest gentle (convex) alternative; 'bisquare' is the aggressive one.
METHODS = [('ols', None), ('robust', 'cauchy'), ('robust', 'fair'), ('robust', 'bisquare')]
BASELINE = ('auto', 'ols', None)       # what every existing checkpoint used

# shallow-water constituents (4th-diurnal and higher) reported for the winner
SW_REPORT = ['MN4', 'M4', 'SN4', 'MS4', 'MK4', 'S4', 'SK4', *SW_EXTRA[:4],
             '2MK5', '2SK5', *SW_EXTRA[4:11], '2MN6', 'M6', '2MS6', '2MK6', '2SM6', 'MSK6',
             *SW_EXTRA[11:16], '3MK7', 'M7', 'M8', *SW_EXTRA[17:20], '4MK9', 'M10']

FIT_START = '2016-01-01' if SMOKE else None   # smoke: train on 2016-2017 only


# %% Helpers ------------------------------------------------------------------
def label_of(constit_set, method, wfunc):
    return f"{constit_set} / {method}" + (f" ({wfunc})" if wfunc else "")


def scores(df, recon):
    """RMSE/MAE (m) of the reconstruction per split, unflagged rows only."""
    t = df['DateTime']
    ok = ~df['is_flagged'].to_numpy()
    err = df['Observed_ODN'].to_numpy() - recon
    splits = {'train': (t < C.TRAIN_END) & ((t >= FIT_START) if FIT_START else True),
              'val': (t >= C.TRAIN_END) & (t < C.VAL_END),
              'test': t >= C.VAL_END}
    out = {}
    for name, m in splits.items():
        e = err[m.to_numpy() & ok]
        out[f'{name}_rmse_m'] = float(np.sqrt(np.mean(e ** 2)))
        out[f'{name}_mae_m'] = float(np.mean(np.abs(e)))
    return out


def run_fit(df, lat, auto_names, constit_set, method, wfunc, stride):
    constit = 'auto' if constit_set == 'auto' else list(auto_names) + SW_EXTRA
    robust_kw = {'weight_function': wfunc} if wfunc else None
    t0 = time.time()
    coef = datasets.fit_utide(df, start=FIT_START, end=C.TRAIN_END, lat=lat, stride=stride,
                              verbose=False, constit=constit, method=method, robust_kw=robust_kw)
    recon = datasets.reconstruct_utide(df['DateTime'], coef)
    row = {'option_set': label_of(constit_set, method, wfunc), 'constit_set': constit_set,
           'method': method, 'weight_function': wfunc or '', 'stride': stride,
           'n_constituents': len(coef.name), **scores(df, recon),
           'fit_minutes': round((time.time() - t0) / 60, 2)}
    print(f"  stride {stride}  {row['option_set']:26s} n={row['n_constituents']:3d}  "
          f"train {row['train_rmse_m']:.4f}  val {row['val_rmse_m']:.4f}  "
          f"test {row['test_rmse_m']:.4f} m  ({row['fit_minutes']:.1f} min)", flush=True)
    return row, coef


def plot_search(res, gauge, path):
    """Validation (open) and test (filled) RMSE per option set at the search stride, the
    previous configuration and the validation winner highlighted. Dots rather than bars:
    the differences are ~1%, and a bar axis that doesn't start at zero would exaggerate them."""
    grid = res[res['stride'] == SEARCH_STRIDE].reset_index(drop=True)
    base_label = label_of(*BASELINE)
    winner = grid.loc[grid['val_rmse_m'].idxmin(), 'option_set']
    colours = ['#8c564b' if s == base_label else ('#2ca02c' if s == winner else '0.45')
               for s in grid['option_set']]
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    y = np.arange(len(grid))[::-1]
    ax.hlines(y, grid[['val_rmse_m', 'test_rmse_m']].min(axis=1),
              grid[['val_rmse_m', 'test_rmse_m']].max(axis=1), color='0.8', lw=1, zorder=1)
    ax.scatter(grid['val_rmse_m'], y, facecolors='white', edgecolors=colours, s=30, zorder=2,
               label='validation (selection)')
    ax.scatter(grid['test_rmse_m'], y, c=colours, s=30, zorder=3, label='test')
    for yi, v in zip(y, grid['test_rmse_m']):
        ax.text(v, yi + 0.28, f"{v:.4f}", ha='center', fontsize=7, color=ANNOTATION_INK)
    ax.set_yticks(y, grid['option_set'])
    for tick, c in zip(ax.get_yticklabels(), colours):
        tick.set_color(c if c != '0.45' else 'black')
    ax.set_ylim(-0.6, len(grid) - 0.3)
    ax.set_xlabel('RMSE (m), UTide alone')
    ax.grid(axis='y', visible=False)
    handles = [plt.Line2D([], [], ls='', marker='o', mfc='white', mec='0.3', label='validation (selection)'),
               plt.Line2D([], [], ls='', marker='o', color='0.3', label='test'),
               plt.Line2D([], [], ls='', marker='s', color='#8c564b', label='previous (auto / ols)'),
               plt.Line2D([], [], ls='', marker='s', color='#2ca02c', label='validation winner')]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4,
              fontsize=7, frameon=False, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {path}")


# %% Run ----------------------------------------------------------------------
if '--plot-only' in sys.argv:   # redraw the figures from saved CSVs, no refitting
    for gauge in GAUGES:
        res = pd.read_csv(paths.OUTPUTS_DIR / gauge / 'utide_search' / 'utide_option_search.csv',
                          keep_default_na=False)
        plot_search(res, gauge, paths.REPORT_IMAGES_DIR / f"utide_option_search_{gauge}.png")
    sys.exit(0)

for gauge in GAUGES:
    info = C.gauge_info(gauge)
    out_dir = paths.ensure_dir(paths.OUTPUTS_DIR / gauge / ('utide_search_smoke' if SMOKE else 'utide_search'))
    print(f"\n===== {info['label']}{'  [SMOKE: 2016-2017 fit]' if SMOKE else ''} =====")
    df = datasets.load_gauge_df(info['csv'], verbose=False)

    rows, coefs = [], {}
    auto_names = None
    for constit_set in CONSTIT_SETS:            # 'auto' first: its names seed 'auto+SW'
        for method, wfunc in METHODS:
            row, coef = run_fit(df, info['lat'], auto_names, constit_set, method, wfunc, SEARCH_STRIDE)
            if auto_names is None:
                auto_names = list(coef.name)
            rows.append(row)
            coefs[(constit_set, method, wfunc, SEARCH_STRIDE)] = coef

    res = pd.DataFrame(rows)
    best = res.loc[res['val_rmse_m'].idxmin()]
    win_key = (best['constit_set'], best['method'], best['weight_function'] or None)
    print(f"validation winner (stride {SEARCH_STRIDE}): {best['option_set']}")

    confirm_key = BASELINE
    if CONFIRM_STRIDE is not None and not SMOKE:
        print(f"refitting the winner and the previous configuration at stride {CONFIRM_STRIDE}:")
        for key in dict.fromkeys([BASELINE, win_key]):   # one fit if the winner is the baseline
            row, coef = run_fit(df, info['lat'], auto_names, *key, CONFIRM_STRIDE)
            rows.append(row)
            coefs[(*key, CONFIRM_STRIDE)] = coef
        res = pd.DataFrame(rows)
    confirm_stride = CONFIRM_STRIDE if (CONFIRM_STRIDE is not None and not SMOKE) else SEARCH_STRIDE

    res.to_csv(out_dir / 'utide_option_search.csv', index=False)
    print(f"saved {out_dir / 'utide_option_search.csv'}")

    win_row = res[(res['option_set'] == label_of(*win_key)) & (res['stride'] == confirm_stride)].iloc[0]
    base_row = res[(res['option_set'] == label_of(*BASELINE)) & (res['stride'] == confirm_stride)].iloc[0]
    choice = {
        'gauge': gauge, 'selected_on': 'validation RMSE (2018-2020)', 'search_stride': SEARCH_STRIDE,
        'constit': 'auto' if win_key[0] == 'auto' else auto_names + SW_EXTRA,
        'constit_set': win_key[0], 'method': win_key[1],
        'robust_kw': {'weight_function': win_key[2]} if win_key[2] else None,
        'stride': int(confirm_stride),
        'winner_test_rmse_m': win_row['test_rmse_m'], 'previous_test_rmse_m': base_row['test_rmse_m'],
        'winner_val_rmse_m': win_row['val_rmse_m'], 'previous_val_rmse_m': base_row['val_rmse_m'],
    }
    (out_dir / 'utide_choice.json').write_text(json.dumps(choice, indent=2), encoding='utf-8')
    print(f"saved {out_dir / 'utide_choice.json'}")

    coef = coefs[(*win_key, confirm_stride)]
    amp = pd.Series(np.asarray(coef.A), index=[n.strip() for n in coef.name])
    sw = pd.DataFrame({'constituent': SW_REPORT,
                       'amplitude_m': [amp.get(n, np.nan) for n in SW_REPORT],
                       'in_auto_set': [n in auto_names for n in SW_REPORT]})
    sw.to_csv(out_dir / 'shallow_water_amplitudes.csv', index=False)
    print(f"saved {out_dir / 'shallow_water_amplitudes.csv'}")

    fig_name = f"utide_option_search_{gauge}{'_smoke' if SMOKE else ''}.png"
    plot_search(res, gauge, (out_dir if SMOKE else paths.ensure_dir(paths.REPORT_IMAGES_DIR)) / fig_name)

    print(f"\n{info['label']}: previous (auto/ols) test RMSE {base_row['test_rmse_m']:.4f} m -> "
          f"winner ({label_of(*win_key)}) {win_row['test_rmse_m']:.4f} m at stride {confirm_stride} "
          f"({100 * (base_row['test_rmse_m'] - win_row['test_rmse_m']) / base_row['test_rmse_m']:+.1f}%)")

# %% Done
print("\nDone. Review outputs/<gauge>/utide_search/ before the tuned model runs.")
