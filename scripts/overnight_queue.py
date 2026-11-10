"""Overnight queue, 2026-11-10: dense v2 variants, selection, test scoring of the winner, baselines,
post-processing, report, commit, push, PR comment. Runs WITHOUT Claude Code.

    pythonw scripts/overnight_queue.py            # the real run (launch detached, see README below)
    pythonw scripts/overnight_queue.py --resume   # skip steps already marked ok (scripts/resume_queue.cmd)
    python  scripts/overnight_queue.py --dry-run  # every step with smoke settings, throwaway report

Design
- Each step is a subprocess with its own log (logs/overnight_2026-11-10/<step>.log); the master log
  records start/end/status per step and queue_state.json is rewritten after every change.
- A failed step is retried once (training resumes from its last epoch). Steps whose dependencies
  failed are skipped with a message. Independent later steps still run. The report step ALWAYS runs.
- Time budget: everything must end by DEADLINE (Europe/London). No step runs past
  DEADLINE - REPORT_RESERVE (it is killed and marked failed); optional steps start only if their
  estimate fits.
- RAM: warns when free RAM < 2 GB; kills a step whose process tree exceeds 26 GB.
- Windows is kept awake (SetThreadExecutionState) for the whole run and released at the end.
- git/gh run non-interactively (GIT_TERMINAL_PROMPT=0, ssh BatchMode, timeouts). If the push fails,
  the report stays committed locally and the failure is logged (and sent to ntfy if configured).
- ntfy: put a topic name in ~/.tidal_ntfy_topic (or env TIDAL_NTFY_TOPIC) to get phone pushes.
"""
import argparse
import ctypes
import datetime as dt
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from zoneinfo import ZoneInfo

import psutil

REPO = pathlib.Path(__file__).resolve().parents[1]
PY = r'C:\Users\alfat\anaconda3\envs\geospatial\python.exe'
GIT = shutil.which('git') or r'C:\Program Files\Git\cmd\git.exe'
GH = shutil.which('gh') or r'C:\Program Files\GitHub CLI\gh.exe'
SSH = r'C:/Windows/System32/OpenSSH/ssh.exe'
BRANCH = 'paper/nov-fixes'
PR_NUMBER = '1'
TZ = ZoneInfo('Europe/London')
DEADLINE = dt.datetime(2026, 11, 10, 17, 0, tzinfo=TZ)
REPORT_RESERVE = dt.timedelta(minutes=25)
RAM_KILL_GB = 26.0
RAM_WARN_FREE_GB = 2.0
POLL_S = 15
CREATE_NO_WINDOW = 0x08000000
GIT_ENV = {'GIT_TERMINAL_PROMPT': '0', 'GIT_SSH_COMMAND': f'{SSH} -o BatchMode=yes', 'GH_PROMPT_DISABLED': '1'}
REPORT_URL = f'https://github.com/Almapian/TML/blob/{BRANCH}/reports/2026-11-nov-fixes/02_overnight_results.md'


def now():
    return dt.datetime.now(TZ)


class Queue:
    def __init__(self, dry, resume):
        self.dry = dry
        self.run_dir = REPO / 'logs' / ('overnight_2026-11-10' + ('_dryrun' if dry else ''))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = REPO / 'outputs' / ('overnight_2026-11-10' + ('_dryrun' if dry else ''))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / 'queue_state.json'
        self.master = self.run_dir / 'master.log'
        self.suffix = '_smoke' if dry else ''
        self.report = REPO / 'reports' / '2026-11-nov-fixes' / ('dryrun_overnight_results.md' if dry else '02_overnight_results.md')
        self.topic = os.environ.get('TIDAL_NTFY_TOPIC') or self._read_topic()
        old = json.loads(self.state_path.read_text(encoding='utf-8')) if (resume and self.state_path.exists()) else None
        self.steps = self.define_steps()
        if old:
            done = {s['name']: s for s in old['steps']}
            for s in self.steps:
                if done.get(s['name'], {}).get('status') == 'ok':
                    s.update({k: v for k, v in done[s['name']].items() if k in ('status', 'start', 'end', 'duration_min', 'note', 'log')})
                    s['note'] = (s.get('note', '') + ' (completed in an earlier launch)').strip()
        self.state = {'started': (old or {}).get('started', now().strftime('%Y-%m-%d %H:%M')),
                      'commit': self.git(['rev-parse', '--short', 'HEAD'])[1].strip(),
                      'branch': self.git(['rev-parse', '--abbrev-ref', 'HEAD'])[1].strip(),
                      'deadline': DEADLINE.strftime('%Y-%m-%d %H:%M %Z'), 'dry_run': dry,
                      'run_dir': str(self.out_dir), 'pid': os.getpid(), 'steps': self.steps}
        self.save()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _read_topic():
        f = pathlib.Path.home() / '.tidal_ntfy_topic'
        return f.read_text(encoding='utf-8').strip() if f.exists() else None

    def log(self, msg):
        line = f"{now():%Y-%m-%d %H:%M:%S} {msg}"
        with open(self.master, 'a', encoding='utf-8') as fh:
            fh.write(line + '\n')
        print(line, flush=True)

    def save(self):
        tmp = self.state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.state, indent=2, default=str), encoding='utf-8')
        os.replace(tmp, self.state_path)

    def notify(self, title, body):
        if not self.topic:
            return
        try:
            req = urllib.request.Request(f'https://ntfy.sh/{self.topic}', data=body.encode('utf-8'),
                                         headers={'Title': title.encode('ascii', 'ignore').decode()})
            urllib.request.urlopen(req, timeout=15).read()
        except Exception as exc:
            self.log(f'ntfy failed: {exc}')

    def git(self, args, timeout=120, exe=None):
        env = dict(os.environ, **GIT_ENV)
        try:
            p = subprocess.run([exe or GIT] + args, cwd=REPO, env=env, capture_output=True, text=True,
                               timeout=timeout, stdin=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
            return p.returncode, p.stdout + p.stderr
        except subprocess.TimeoutExpired:
            return 124, f'timed out after {timeout}s'
        except Exception as exc:
            return 1, str(exc)

    def env(self, gauge=None, variant=None, score_test=False):
        e = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
        if gauge:
            e['TIDAL_GAUGE'] = gauge
        if variant:
            e['TIDAL_DENSE_VARIANT'] = variant
        e['TIDAL_SCORE_TEST'] = '1' if score_test else '0'
        e['TIDAL_SMOKE'] = '1' if self.dry else '0'
        return e

    def winner(self, gauge):
        f = self.out_dir / f'selection_{gauge}{self.suffix}.json'
        return json.loads(f.read_text(encoding='utf-8'))['winner']

    # ------------------------------------------------------------------ steps
    def define_steps(self):
        S = []

        def train(name, gauge, variant, est, optional=False):
            S.append({'name': name, 'kind': 'cmd', 'optional': optional, 'est_min': est, 'deps': [],
                      'cmd': lambda g=gauge, v=variant: ([PY, 'models/train_dense.py'], self.env(g, v)),
                      'notify': True})

        train('01_train_skip_wloss_southend', 'southend', 'dense_v2_skip_wloss', 100)
        train('02_train_skip_scaled_southend', 'southend', 'dense_v2_skip_scaled', 90)
        train('03_train_pad_southend', 'southend', 'dense_v2_pad', 100)
        train('04_train_skip_wloss_richmond', 'richmond', 'dense_v2_skip_wloss', 90)
        train('05_train_skip_scaled_richmond', 'richmond', 'dense_v2_skip_scaled', 80)
        train('06_train_pad_richmond', 'richmond', 'dense_v2_pad', 90)
        train('07_train_skip_southend', 'southend', 'dense_v2_skip', 85)
        for g in ('southend', 'richmond'):
            extra = ['--exclude-legacy'] if self.dry else []   # the dry run must exercise the v2 test path
            S.append({'name': f'08_select_{g}', 'kind': 'cmd', 'optional': False, 'est_min': 1, 'deps': [],
                      'cmd': lambda g=g, extra=extra: ([PY, 'models/select_dense_winner.py', '--gauge', g,
                                                         '--suffix', self.suffix, '--out', str(self.out_dir)] + extra,
                                                        self.env())})
        for g in ('southend', 'richmond'):
            S.append({'name': f'08_testscore_winner_{g}', 'kind': 'testscore', 'gauge': g, 'optional': False,
                      'est_min': 8, 'deps': [f'08_select_{g}']})
        for g in ('southend', 'richmond'):
            S.append({'name': f'09_baselines_vs_winner_{g}', 'kind': 'cmd', 'optional': False, 'est_min': 15,
                      'deps': [f'08_testscore_winner_{g}'],
                      'cmd': lambda g=g: ([PY, 'models/eval_baselines_vs_dense.py'], self.env(g, self.winner(g)))})
        for g in ('southend', 'richmond'):
            for split, deps in (('val', [f'08_select_{g}']), ('test', [f'08_testscore_winner_{g}'])):
                S.append({'name': f'10_postprocess_{g}_{split}', 'kind': 'cmd', 'optional': False, 'est_min': 5,
                          'deps': deps,
                          'cmd': lambda g=g, split=split: ([PY, 'models/postprocess_hw.py', '--gauge', g, '--variant',
                                                            self.winner(g), '--split', split, '--with-ar']
                                                           + (['--max-anchors', '2000', '--smoke-suffix', '_smoke'] if self.dry else []),
                                                           self.env())})
        train('11_optional_train_skip_richmond', 'richmond', 'dense_v2_skip', 75, optional=True)
        train('11_optional_train_skip_longmem_southend', 'southend', 'dense_v2_skip_longmem', 100, optional=True)
        S.append({'name': '12_report_commit_push', 'kind': 'report', 'optional': False, 'est_min': 10, 'deps': []})
        for s in S:
            s.setdefault('status', 'pending')
            s['log'] = str(self.run_dir / f"{s['name']}.log")
        return S

    # ------------------------------------------------------------------ running
    def run_cmd(self, step, cmd, env, hard_stop):
        with open(step['log'], 'a', encoding='utf-8') as fh:
            fh.write(f"\n===== {step['name']} attempt at {now():%Y-%m-%d %H:%M:%S}: {' '.join(cmd)} =====\n")
            fh.flush()
            proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
            ps = psutil.Process(proc.pid)
            peak, warned = 0.0, False
            while proc.poll() is None:
                time.sleep(POLL_S)
                try:
                    rss = sum(p.memory_info().rss for p in [ps] + ps.children(recursive=True)) / 1e9
                except psutil.Error:
                    rss = 0.0
                peak = max(peak, rss)
                free = psutil.virtual_memory().available / 1e9
                if free < RAM_WARN_FREE_GB and not warned:
                    self.log(f"WARNING {step['name']}: free RAM {free:.1f} GB (step uses {rss:.1f} GB)")
                    warned = True
                reason = None
                if rss > RAM_KILL_GB:
                    reason = f'killed: process tree {rss:.1f} GB > {RAM_KILL_GB} GB'
                elif now() > hard_stop:
                    reason = f'killed: time budget (hard stop {hard_stop:%H:%M})'
                if reason:
                    for p in ps.children(recursive=True) + [ps]:
                        try:
                            p.kill()
                        except psutil.Error:
                            pass
                    proc.wait()
                    fh.write(f"===== {reason} =====\n")
                    return 1, reason, peak
            fh.write(f"===== exit code {proc.returncode}, peak RAM {peak:.1f} GB =====\n")
            return proc.returncode, '', peak

    def run_step(self, step):
        hard_stop = DEADLINE - REPORT_RESERVE
        failed_deps = [d for d in step['deps'] if self.by_name(d)['status'] != 'ok']
        if failed_deps:
            step.update(status='skipped', note=f"dependency not ok: {', '.join(failed_deps)}")
            self.log(f"SKIP {step['name']}: {step['note']}")
            return
        if not self.dry:
            if now() >= hard_stop:
                step.update(status='skipped', note='time budget: past the hard stop')
                self.log(f"SKIP {step['name']}: {step['note']}")
                return
            if step['optional'] and now() + dt.timedelta(minutes=step['est_min']) > hard_stop:
                step.update(status='skipped', note=f"time budget: needs ~{step['est_min']} min, "
                                                   f"only {(hard_stop - now()).seconds // 60} min left")
                self.log(f"SKIP {step['name']}: {step['note']}")
                return
        t0 = now()
        step.update(status='running', start=t0.strftime('%H:%M'))
        self.save()
        self.log(f"START {step['name']}")
        for attempt in (1, 2):
            try:
                if step['kind'] == 'testscore':
                    w = self.winner(step['gauge'])
                    if w == 'dense':
                        code, reason, peak = 0, '', 0.0
                        step['note'] = 'old dense model won: its test numbers are already published, no new test scoring'
                    else:
                        code, reason, peak = self.run_cmd(step, [PY, 'models/train_dense.py'],
                                                          self.env(step['gauge'], w, score_test=True), hard_stop)
                        step['note'] = f'test-scored winner {w}'
                else:
                    cmd, env = step['cmd']()
                    code, reason, peak = self.run_cmd(step, cmd, env, hard_stop)
                step['peak_ram_gb'] = round(peak, 1)
            except Exception as exc:
                code, reason = 1, f'queue error: {exc}'
                with open(step['log'], 'a', encoding='utf-8') as fh:
                    fh.write(traceback.format_exc())
            if code == 0:
                break
            step['note'] = reason or f'exit code {code}'
            if reason.startswith('killed') or attempt == 2:
                break
            self.log(f"RETRY {step['name']} after failure ({step['note']})")
        end = now()
        step.update(status='ok' if code == 0 else 'failed', end=end.strftime('%H:%M'),
                    duration_min=round((end - t0).total_seconds() / 60, 1))
        self.save()
        self.log(f"{step['status'].upper()} {step['name']} in {step['duration_min']} min {step.get('note', '')}")
        if step.get('notify'):
            self.notify(f"{step['name']}: {step['status']}", f"{step['duration_min']} min. {step.get('note', '')}")

    def by_name(self, name):
        return next(s for s in self.steps if s['name'] == name)

    # ------------------------------------------------------------------ report + git
    def summary_lines(self):
        st = [s['status'] for s in self.steps if s['kind'] != 'report']
        lines = [f"{'[DRY RUN] ' if self.dry else ''}Overnight queue from `{self.state['commit']}`: "
                 f"{st.count('ok')} ok, {st.count('failed')} failed, {st.count('skipped')} skipped."]
        for g in ('southend', 'richmond'):
            try:
                r = json.loads((self.out_dir / f'selection_{g}{self.suffix}.json').read_text())
                best = sorted(r['scores'].items(), key=lambda kv: kv[1])
                lines.append(f"{g}: winner `{r['winner']}` (score {best[0][1]:.3f}); runner-up "
                             f"{best[1][0] if len(best) > 1 else '-'} {best[1][1] if len(best) > 1 else float('nan'):.3f}.")
            except Exception:
                lines.append(f'{g}: no selection.')
        for g in ('southend', 'richmond'):
            try:
                w = self.winner(g)
                import pandas as pd
                base = REPO / 'outputs' / ('' if g == 'southend' else g)
                vd = pathlib.Path(str(base / w) + ('' if w == 'dense' else self.suffix))
                m = pd.read_csv(vd / 'dense_metrics.csv')
                s = m[m.model == 'Dense stage 2'].set_index('horizon').rmse_m
                lines.append(f"{g} winner TEST RMSE: 10min {s['10min']:.3f} m, 24h {s['24h']:.3f} m, 72h {s['72h']:.3f} m.")
                j = pd.read_csv(pathlib.Path(str(base / 'baselines') + f'_vs_{w}' + ('' if w == 'dense' else self.suffix))
                                / 'baseline_joint_tests.csv')
                j = j[j.leads == '24h to 72h']
                lines.append(f"{g} joint test 24-72h (baseline minus dense): " + '; '.join(
                    f"{r.baseline} {r.mean_rmse_diff_m:+.4f} m [{r.ci_low:+.4f}, {r.ci_high:+.4f}]" for r in j.itertuples()))
            except Exception:
                lines.append(f'{g}: winner test / baseline numbers not available.')
        failed = [s['name'] for s in self.steps if s['status'] == 'failed']
        if failed:
            lines.append('Failed: ' + ', '.join(failed) + ' (logs in ' + str(self.run_dir.relative_to(REPO)) + ').')
        lines.append(f"Report: {REPORT_URL if not self.dry else self.report.relative_to(REPO).as_posix()}")
        return lines[:10]

    def append_changelog(self):
        cl = REPO / 'CHANGELOG.md'
        text = cl.read_text(encoding='utf-8')
        winners = {}
        for g in ('southend', 'richmond'):
            try:
                winners[g] = self.winner(g)
            except Exception:
                winners[g] = None
        rows = '\n'.join(f"- {s['name']}: {s['status']}{' (' + s['note'] + ')' if s.get('note') else ''}" for s in self.steps
                         if s['kind'] != 'report')
        tested = [f"{g}: {w}" for g, w in winners.items() if w and w != 'dense' and self.by_name(f'08_testscore_winner_{g}')['status'] == 'ok']
        entry = (f"\n## {now():%Y-%m-%d}: overnight queue results (automatic entry)\n\n"
                 f"Queue `scripts/overnight_queue.py` from commit `{self.state['commit']}`, started {self.state['started']}. "
                 f"Selection by the amended pre-registered rule; winners {winners}.\n\n"
                 f"**Test viewed for:** {', '.join(tested) if tested else 'no new model'} (winner test scoring), the classical "
                 f"baselines re-scored against each winner (test, joint tests), and post-processing on the winner's test "
                 f"predictions. No other variant was scored on test.\n\n"
                 f"Steps:\n{rows}\n\nReport: `reports/2026-11-nov-fixes/02_overnight_results.md`. Logs: `{self.run_dir.relative_to(REPO).as_posix()}/`.\n")
        marker = '---\n'
        i = text.index(marker) + len(marker)
        cl.write_text(text[:i] + entry + text[i:], encoding='utf-8')

    def report_and_publish(self, step):
        t0 = now()
        step.update(status='running', start=t0.strftime('%H:%M'))
        self.save()
        notes = []
        with open(step['log'], 'a', encoding='utf-8') as fh:
            p = subprocess.run([PY, 'models/overnight_report.py', '--state', str(self.state_path), '--out', str(self.report),
                                '--suffix', self.suffix], cwd=REPO, env=self.env(), stdout=fh, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, timeout=1800, creationflags=CREATE_NO_WINDOW)
        if p.returncode != 0 or not self.report.exists():
            notes.append(f'report script exit {p.returncode}; writing a minimal report')
            self.report.write_text('# Overnight results (minimal: the report script failed)\n\n' +
                                   '\n'.join(f"- {s['name']}: {s['status']} {s.get('note', '')}" for s in self.steps) + '\n',
                                   encoding='utf-8')
        summary = self.summary_lines()
        (self.run_dir / 'summary.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')
        paths_to_commit = ['reports/2026-11-nov-fixes']
        if not self.dry:
            try:
                self.append_changelog()
                paths_to_commit.append('CHANGELOG.md')
            except Exception as exc:
                notes.append(f'CHANGELOG append failed: {exc}')
        code, out = self.git(['rev-parse', '--abbrev-ref', 'HEAD'])
        pushed = False
        if out.strip() != BRANCH:
            notes.append(f'not on {BRANCH} (on {out.strip()}): nothing committed')
        else:
            self.git(['add', '--'] + paths_to_commit)
            msg = (f"{'[dry run] ' if self.dry else ''}Overnight queue results ({now():%Y-%m-%d %H:%M})\n\n"
                   + '\n'.join(summary) + '\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n'
                   'Claude-Session: https://claude.ai/code/session_01HzC8XP21r5mzJs4wUGBq7r\n')
            code, out = self.git(['commit', '-m', msg, '--'] + paths_to_commit)
            notes.append(f'commit exit {code}')
            code, out = self.git(['push', 'origin', BRANCH], timeout=180)
            pushed = code == 0
            notes.append('push ok' if pushed else f'PUSH FAILED (exit {code}): {out.strip()[-300:]}')
            with open(step['log'], 'a', encoding='utf-8') as fh:
                fh.write(f'push output: {out}\n')
        if pushed:
            body = self.run_dir / 'pr_comment.md'
            body.write_text('\n'.join(f'{i + 1}. {l}' for i, l in enumerate(summary)) + '\n', encoding='utf-8')
            code, out = self.git(['pr', 'comment', PR_NUMBER, '--body-file', str(body)], timeout=90, exe=GH)
            notes.append('PR comment ok' if code == 0 else f'PR comment failed (exit {code}): {out.strip()[-200:]}')
        self.notify('Overnight queue done' + ('' if pushed else ' (push failed)'), '\n'.join(summary))
        end = now()
        step.update(status='ok' if pushed else 'failed', end=end.strftime('%H:%M'),
                    duration_min=round((end - t0).total_seconds() / 60, 1), note='; '.join(notes))
        self.save()
        self.log(f"{step['status'].upper()} {step['name']}: {step['note']}")

    def run(self):
        self.log(f"QUEUE START pid {os.getpid()} dry={self.dry} commit {self.state['commit']} deadline {DEADLINE:%Y-%m-%d %H:%M}")
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        try:
            for step in self.steps:
                if step['kind'] == 'report' or step['status'] == 'ok':
                    continue
                try:
                    self.run_step(step)
                except Exception as exc:
                    step.update(status='failed', note=f'queue error: {exc}')
                    self.log(f"FAILED {step['name']}: queue error {exc}\n{traceback.format_exc()}")
                    self.save()
        finally:
            try:
                self.report_and_publish(self.by_name('12_report_commit_push'))
            except Exception as exc:
                self.log(f'REPORT STEP ERROR: {exc}\n{traceback.format_exc()}')
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            self.log('QUEUE END')
            (self.run_dir / 'queue.pid').unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    run_dir = REPO / 'logs' / ('overnight_2026-11-10' + ('_dryrun' if a.dry_run else ''))
    run_dir.mkdir(parents=True, exist_ok=True)
    pid_file = run_dir / 'queue.pid'
    if pid_file.exists():
        try:
            other = int(pid_file.read_text())
            if psutil.pid_exists(other) and 'python' in psutil.Process(other).name().lower():
                print(f'another queue is running (pid {other}); refusing to start')
                sys.exit(1)
        except (ValueError, psutil.Error):
            pass
    pid_file.write_text(str(os.getpid()))
    Queue(a.dry_run, a.resume).run()


if __name__ == '__main__':
    main()
