param(
    [switch]$SkipInstall
)

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

# Try to activate `venv` in the current window if it exists
$venvActivate = Join-Path $RepoRoot 'venv\Scripts\Activate.ps1'
if (Test-Path $venvActivate) {
    Write-Host "Activating virtualenv..."
    & $venvActivate
}

# If not skipping installs, start them in background PowerShell windows so
# servers can be started immediately.
if (-not $SkipInstall) {
    if (Test-Path (Join-Path $RepoRoot 'requirements.txt')) {
        Write-Host "Starting Python deps install in background..."
        $installPyCmd = "cd `"$RepoRoot`"; if (Test-Path 'venv\\Scripts\\Activate.ps1') { & 'venv\\Scripts\\Activate.ps1' }; python -m pip install --upgrade pip; python -m pip install -r `"$RepoRoot\\requirements.txt`""
        Start-Process -FilePath "powershell" -ArgumentList "-NoProfile","-NoExit","-Command",$installPyCmd -WorkingDirectory $RepoRoot
    }
    $frontendPkg = Join-Path $RepoRoot 'frontend\package.json'
    if (Test-Path $frontendPkg) {
        Write-Host "Starting frontend deps install in background..."
        $installFrontendCmd = "Set-Location -LiteralPath '$($RepoRoot)\\frontend'; npm ci --no-audit --no-fund -s || npm install --no-audit --no-fund -s"
        Start-Process -FilePath "powershell" -ArgumentList "-NoProfile","-NoExit","-Command",$installFrontendCmd -WorkingDirectory (Join-Path $RepoRoot 'frontend')
    }
}

# Start backend in a new PowerShell window (immediately)
$backendArgs = @(
    "-NoProfile",
    "-NoExit",
    "-Command",
    "python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000"
)
Start-Process -FilePath "powershell" -ArgumentList $backendArgs -WorkingDirectory $RepoRoot

# Start frontend in a new PowerShell window (immediately)
$frontendCmd = "Set-Location -LiteralPath '$($RepoRoot)\\frontend'; npm run dev"
$frontendArgs = @(
    "-NoProfile",
    "-NoExit",
    "-Command",
    $frontendCmd
)
Start-Process -FilePath "powershell" -ArgumentList $frontendArgs -WorkingDirectory $RepoRoot

Write-Host "Launched backend (http://127.0.0.1:8000) and frontend (http://localhost:5173)."
