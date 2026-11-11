"""Is the overnight queue running, and where is it?

    python scripts/queue_status.py [--dry-run]
"""
import json
import pathlib
import sys

import psutil

REPO = pathlib.Path(__file__).resolve().parents[1]
run_dir = REPO / 'logs' / ('overnight_2026-11-10' + ('_dryrun' if '--dry-run' in sys.argv else ''))
pid_file, state_file = run_dir / 'queue.pid', run_dir / 'queue_state.json'

if pid_file.exists():
    pid = int(pid_file.read_text())
    alive = psutil.pid_exists(pid)
    print(f"queue pid {pid}: {'RUNNING' if alive else 'NOT RUNNING (stale pid file: it crashed or was killed)'}")
else:
    print('queue: not running (no pid file; it finished or never started)')
if state_file.exists():
    st = json.loads(state_file.read_text(encoding='utf-8'))
    print(f"commit {st['commit']}  started {st['started']}  deadline {st['deadline']}")
    for s in st['steps']:
        print(f"  {s['name']:42s} {s.get('status', ''):8s} {s.get('start', ''):>5s}-{s.get('end', ''):<5s} "
              f"{str(s.get('duration_min', '')):>6s} min  {s.get('note', '')[:70]}")
master = run_dir / 'master.log'
if master.exists():
    print('\nlast master log lines:')
    print(''.join(master.read_text(encoding='utf-8').splitlines(keepends=True)[-5:]))
