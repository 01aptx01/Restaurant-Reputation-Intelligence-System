# Add Python 3.12 to the current user's PATH (fixes "python is not recognized").
# Run once, then open a new terminal:  .\scripts\fix-python-path.ps1

$ErrorActionPreference = "Stop"

$pythonRoot = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312"
$pythonScripts = Join-Path $pythonRoot "Scripts"

if (-not (Test-Path (Join-Path $pythonRoot "python.exe"))) {
    Write-Host "Python 3.12 not found at $pythonRoot"
    Write-Host "Install from https://www.python.org/downloads/ and check 'Add python.exe to PATH'."
    Write-Host "Or use the launcher:  py -3.12 -m venv .venv"
    exit 1
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$segments = @($pythonRoot, $pythonScripts)
$missing = $segments | Where-Object { $userPath -notlike "*$_*" }

if ($missing.Count -eq 0) {
    Write-Host "Python 3.12 is already on your User PATH."
} else {
    $newPath = ($userPath.TrimEnd(";") + ";" + ($missing -join ";")).Trim(";")
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "Updated User PATH. Added:"
    $missing | ForEach-Object { Write-Host "  $_" }
    Write-Host ""
    Write-Host "Close this terminal and open a new one, then run:  python --version"
}

# Refresh PATH for this session only
$env:Path = "$env:Path;$pythonRoot;$pythonScripts"
& "$pythonRoot\python.exe" --version
