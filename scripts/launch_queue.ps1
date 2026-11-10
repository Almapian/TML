# Launch the overnight queue fully detached from this terminal and from Claude Code.
# Win32_Process.Create starts it outside the caller's process tree and job object, so closing
# the terminal or Claude Code does not kill it. pythonw has no console window.
#   powershell -ExecutionPolicy Bypass -File scripts\launch_queue.ps1            # real run
#   powershell -ExecutionPolicy Bypass -File scripts\launch_queue.ps1 -Resume    # skip completed steps
#   powershell -ExecutionPolicy Bypass -File scripts\launch_queue.ps1 -DryRun
param([switch]$Resume, [switch]$DryRun)
$repo = Split-Path -Parent $PSScriptRoot
$pyw = 'C:\Users\alfat\anaconda3\envs\geospatial\pythonw.exe'
$arguments = @()
if ($Resume) { $arguments += '--resume' }
if ($DryRun) { $arguments += '--dry-run' }
$cmd = "`"$pyw`" `"$repo\scripts\overnight_queue.py`" " + ($arguments -join ' ')
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd; CurrentDirectory = $repo}
if ($r.ReturnValue -ne 0) { Write-Error "launch failed: $($r.ReturnValue)"; exit 1 }
Write-Output "launched: pid $($r.ProcessId)  ($cmd)"
