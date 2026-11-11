"""Write the overnight queue's results report. Never fails on missing pieces: every section that
cannot be built says why. Reads the queue state, selection files and every metrics CSV produced.

    python models/overnight_report.py --state logs/overnight_2026-11-10/queue_state.json \
        --out reports/2026-11-nov-fixes/02_overnight_results.md [--suffix _smoke]
"""
import pathlib
import sys

_start = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / '_')).resolve()
REPO_ROOT = next(p for p in _start.parents if (p / 'utils' / 'paths.py').exists())
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use('Agg')

import argparse
import json
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import paths
from utils.models import baselines as B
from utils.models import config as C

GAUGES = ['southend', 'richmond']
CANDIDATES = ['dense', 'dense_v2_skip', 'dense_v2_skip_wloss', 'dense_v2_skip_scaled', 'dense_v2_pad']
EXTRA = ['dense_v2_skip_longmem']
LEADS = list(C.DENSE_HORIZONS)
SHOW = ['10min', '1h', '6h', '24h', '48h', '72h']
LEAD_H = np.array([s / 6 for s in C.DENSE_HORIZONS.values()])
# validated categorical order (dataviz check, Oct 2026) + neutrals for references
COL = {'dense': '#52514e', 'dense_v2_skip': '#2a78d6', 'dense_v2_skip_wloss': '#eb6834',
       'dense_v2_skip_scaled': '#1baf7a', 'dense_v2_pad': '#8a4fd8', 'dense_v2_skip_longmem': '#c9a227',
       'Residual persistence': '#9a9993', 'UTide standalone': '#0b0b0b'}
STYLE = {'Residual persistence': (0, (4, 2)), 'UTide standalone': (0, (1, 1.5))}


def vdir(gauge, v, suffix):
    return pathlib.Path(str(paths.dense_output_dir(gauge, v)) + ('' if v == C.DENSE_LEGACY_VARIANT else suffix))


def table(df, floatfmt=4):
    if df is None or len(df) == 0:
        return '_(empty)_\n'
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda x: '' if pd.isna(x) else f'{x:.{floatfmt}f}')
    cols = [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(str(x) for x in r.values) + ' |')
    return '\n'.join(lines) + '\n'


def section(title, fn, out):
    out.append(f'\n## {title}\n')
    try:
        out.append(fn())
    except Exception as exc:
        out.append(f'_Not available: {type(exc).__name__}: {exc}_\n\n```\n{traceback.format_exc()[-1500:]}\n```\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--suffix', default='')
    a = ap.parse_args()
    suffix = a.suffix
    state = json.loads(pathlib.Path(a.state).read_text(encoding='utf-8'))
    out_md = pathlib.Path(a.out)
    fig_dir = paths.ensure_dir(out_md.parent / 'figures')
    tag = 'overnight' + ('_dryrun' if suffix else '')
    run_dir = pathlib.Path(state['run_dir'])
    md = [f"# Overnight results{' (DRY RUN, smoke settings: numbers meaningless)' if suffix else ''}\n",
          f"Queue started {state.get('started')} from commit `{state.get('commit')}` on branch "
          f"`{state.get('branch')}`. Report written {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
          f"Deadline {state.get('deadline')}. Development decisions use validation only; test was scored only "
          f"for the winner per gauge (see CHANGELOG).\n"]

    def status():
        rows = []
        for s in state['steps']:
            rows.append({'step': s['name'], 'status': s.get('status', 'pending'),
                         'start': s.get('start', ''), 'end': s.get('end', ''),
                         'minutes': s.get('duration_min', ''),
                         'note / log': (s.get('note', '') + (f" log: `{s['log']}`" if s.get('status') == 'failed' else ''))})
        return table(pd.DataFrame(rows), 1)
    section('Step status', status, md)

    sel = {}

    def selection():
        txt = ('Rule (pre-registered, amended 2026-11-10): mean over the 73 leads of validation RMSE divided by '
               'min(residual persistence, UTide) RMSE at that lead; lowest wins.\n\n')
        rows = []
        for g in GAUGES:
            f = run_dir / f'selection_{g}{suffix}.json'
            if not f.exists():
                txt += f'- {g}: no selection file\n'
                continue
            r = json.loads(f.read_text())
            sel[g] = r
            for v, sc in sorted(r['scores'].items(), key=lambda kv: kv[1]):
                rows.append({'gauge': g, 'candidate': v, 'score': sc, 'winner': 'yes' if v == r['winner'] else ''})
            if r['missing']:
                txt += f"- {g}: did not finish: {', '.join(r['missing'])}\n"
        return txt + '\n' + table(pd.DataFrame(rows))
    section('Selection (validation)', selection, md)

    def val_curves():
        txt = ''
        for g in GAUGES:
            frames = {}
            for v in CANDIDATES + EXTRA:
                f = vdir(g, v, suffix) / 'dense_val_metrics.csv'
                if f.exists():
                    frames[v] = pd.read_csv(f)
            if not frames:
                txt += f'\n### {g}\n_no validation metrics_\n'
                continue
            ref = next((m for k, m in frames.items() if k != 'dense'), None)
            rows, curves = [], {}
            for v, m in frames.items():
                s = m[m.model == 'Dense stage 2'].set_index('horizon').loc[LEADS, 'rmse_m']
                curves[v] = s.to_numpy()
                rows.append({'model': v, **{h: s[h] for h in SHOW}, 'mean 73': s.mean()})
            if ref is not None:
                for rname in ('Residual persistence', 'UTide standalone'):
                    s = ref[ref.model == rname].set_index('horizon').loc[LEADS, 'rmse_m']
                    curves[rname] = s.to_numpy()
                    rows.append({'model': rname, **{h: s[h] for h in SHOW}, 'mean 73': s.mean()})
            fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.4), gridspec_kw={'width_ratios': [1, 2]})
            for ax, (lo, hi) in zip(axes, [(0, 6.1), (0, 72.5)]):
                for k, y in curves.items():
                    ax.plot(LEAD_H, y, color=COL.get(k, '#777'), lw=1.6 if k in frames else 1.1,
                            ls=STYLE.get(k, '-'), label=k)
                ax.set_xlim(lo, hi); ax.set_xlabel('Lead (h)')
            axes[0].set_ylim(0, None); axes[0].set_ylabel('Validation RMSE (m)')
            axes[1].legend(fontsize=6.5, frameon=False, ncol=2, loc='lower right')
            fig.suptitle(f"{C.gauge_info(g)['label']}: validation RMSE by lead (left: first 6h)", fontsize=9)
            fig.tight_layout()
            name = f'{tag}_val_rmse_by_lead_{g}.png'
            fig.savefig(fig_dir / name, dpi=200); plt.close(fig)
            txt += f'\n### {C.gauge_info(g)["label"]}\n\n![{g}](figures/{name})\n\n' + table(pd.DataFrame(rows))
        return txt
    section('Validation RMSE by lead, every candidate (m)', val_curves, md)

    def bias():
        txt = ''
        for g in GAUGES:
            s = B.load_series(g, verbose=False)
            a = B.split_anchors(s, 'val')
            Y, F = B.targets(s.level, s.flagged, a)
            ok = ~F & ~np.isnan(Y)
            fig, ax = plt.subplots(figsize=(6.3, 3.0))
            rows = []
            for v in CANDIDATES + EXTRA:
                f = vdir(g, v, suffix) / 'dense_val_predictions_stage2.npz'
                if not f.exists():
                    continue
                z = np.load(f)
                assert np.array_equal(z['anchors'].astype('datetime64[ns]'), s.time[a].astype('datetime64[ns]'))
                e = np.where(ok, z['preds'] - Y, np.nan)
                b = np.nanmean(e, axis=0)
                ax.plot(LEAD_H, b, color=COL.get(v, '#777'), lw=1.6, label=v)
                rows.append({'model': v, **{h: b[LEADS.index(h)] for h in SHOW}})
            ax.axhline(0, color='#888', lw=0.6)
            ax.set_xlabel('Lead (h)'); ax.set_ylabel('Validation bias (m), forecast - observed')
            ax.legend(fontsize=6.5, frameon=False, ncol=3)
            ax.set_title(C.gauge_info(g)['label'], fontsize=9, loc='left')
            fig.tight_layout()
            name = f'{tag}_val_bias_by_lead_{g}.png'
            fig.savefig(fig_dir / name, dpi=200); plt.close(fig)
            txt += f'\n### {C.gauge_info(g)["label"]}\n\n![{g}](figures/{name})\n\n' + table(pd.DataFrame(rows))
        return txt
    section('Validation bias by lead (m)', bias, md)

    def test_winner():
        txt = ''
        for g in GAUGES:
            w = sel.get(g, {}).get('winner')
            if w is None:
                txt += f'- {g}: no winner\n'
                continue
            f = vdir(g, w, suffix) / 'dense_metrics.csv'
            if not f.exists():
                txt += f'- {g}: winner {w}, no test metrics file\n'
                continue
            m = pd.read_csv(f)
            rows = []
            for name in ('Dense stage 2', 'Residual persistence', 'UTide standalone'):
                s = m[m.model == name].set_index('horizon')
                if len(s):
                    rows.append({'model': f'{w}' if name == 'Dense stage 2' else name,
                                 **{h: s.loc[h, 'rmse_m'] for h in SHOW}, 'mean 73': s.loc[LEADS, 'rmse_m'].mean()})
            leg = pd.read_csv(paths.dense_output_dir(g) / 'dense_metrics.csv')
            s = leg[leg.model == 'Dense stage 2'].set_index('horizon')
            rows.append({'model': 'old dense (published)', **{h: s.loc[h, 'rmse_m'] for h in SHOW},
                         'mean 73': s.loc[LEADS, 'rmse_m'].mean()})
            txt += f'\n### {C.gauge_info(g)["label"]}: winner `{w}`\n\n' + table(pd.DataFrame(rows))
        return txt
    section('TEST RMSE of the winner per gauge (m)', test_winner, md)

    def baselines():
        txt = ''
        for g in GAUGES:
            w = sel.get(g, {}).get('winner')
            d = pathlib.Path(str(paths.baseline_output_dir(g)) + f'_vs_{w}' + ('' if w == C.DENSE_LEGACY_VARIANT else suffix))
            if w is None or not d.exists():
                txt += f'- {g}: not available\n'
                continue
            j = pd.read_csv(d / 'baseline_joint_tests.csv')
            m = pd.read_csv(d / 'baseline_metrics.csv')
            base_winners = list(json.loads((d / 'winners.json').read_text())['baseline_winners'].values())
            keep = [f'Dense stage 2 ({w})', 'UTide standalone'] + base_winners
            mm = m[m.model.isin(keep) & m.horizon.isin(SHOW)].pivot_table(index='model', columns='horizon', values='rmse_m')[SHOW].reset_index()
            txt += (f'\n### {C.gauge_info(g)["label"]} (test, vs `{w}`)\n\nJoint tests: mean over leads of RMSE(baseline) - '
                    f'RMSE(dense), 95% block-bootstrap CI (> 0 favours dense):\n\n' + table(j.drop(columns=['comparator'])) +
                    '\nRMSE at selected leads (m):\n\n' + table(mm) + f'\nFull tables: `{d}`\n')
        return txt
    section('Classical baselines vs the winner (test)', baselines, md)

    def post():
        txt = ''
        for g in GAUGES:
            w = sel.get(g, {}).get('winner')
            if w is None:
                continue
            for split in ('val', 'test'):
                d = vdir(g, w, suffix) / ('postprocess_subset' if suffix else 'postprocess')
                f = d / f'{split}_rmse_433_leads.csv'
                if not f.exists():
                    txt += f'- {g} {split}: no post-processing output\n'
                    continue
                r = pd.read_csv(f)
                pv = r[r.lead_min.isin([10, 20, 30, 40, 50, 60, 360, 1440, 4320])].pivot_table(
                    index='model', columns='lead_min', values='rmse_m').reset_index()
                pv.columns = ['model'] + [f'{c} min' for c in pv.columns[1:]]
                hw = pd.read_csv(d / f'{split}_high_low_water.csv')
                hw = hw[hw.lead_bin_h == '0-72'].drop(columns=['lead_bin_h'])
                chk = json.loads((d / f'{split}_extremes_check.json').read_text())
                txt += (f'\n### {C.gauge_info(g)["label"]}, `{w}`, {split}\n\nRMSE on the 10-min trajectory (m):\n\n' + table(pv) +
                        '\nHigh/low water errors over 0-72h (height in m, timing in minutes):\n\n' + table(hw) +
                        f"\nHigh waters per day {chk['high_waters_per_day']:.3f}; low waters scored: {chk['lw_scored']} "
                        f"(LW plateau median {chk['lw_plateau_median_min']:.0f} min vs HW {chk['hw_plateau_median_min']:.0f} min).\n")
        return txt
    section('Interpolation to 10 min and high-water metric (winner)', post, md)

    out_md.write_text('\n'.join(md), encoding='utf-8')
    print(f'wrote {out_md}')


if __name__ == '__main__':
    main()
