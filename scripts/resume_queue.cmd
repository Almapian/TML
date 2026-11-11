@echo off
rem Resume the overnight queue detached, skipping steps already marked ok in queue_state.json.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch_queue.ps1" -Resume
