param(
    [switch]$Full
)

Write-Host "Setting up Python virtualenv for FinAlze (PowerShell)"

# Helper: find python and invoke commands robustly (python | py -3 | python3)
function Invoke-Python {
    param([string]$Args)
    if (Get-Command python -ErrorAction SilentlyContinue) {
        python $Args
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        py -3 $Args
    } elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
        python3 $Args
    } else {
        Write-Error "Python not found in PATH. Install Python 3 and ensure 'python' or 'py' is on PATH."
        exit 1
    }
}

 $venvPath = Join-Path $PSScriptRoot "..\.venv"
 if (-Not (Test-Path $venvPath)) {
     Invoke-Python "-m venv `"$venvPath`""
 }

Write-Host "Activating venv (if available)..."
$activatePath = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activatePath) {
    & $activatePath
} else {
    Write-Host "Activation script not found; venv may not have been created. You can activate manually after fixing Python." 
}

Write-Host "Upgrading pip..."
Invoke-Python "-m pip install --upgrade pip"

Write-Host "Installing API requirements and test helpers..."
Invoke-Python "-m pip install -r requirements-api.txt"
Invoke-Python "-m pip install pytest pytest-asyncio pillow pandas rapidfuzz"

if ($Full) {
    Write-Host "Installing full requirements (this may take long and require wheels)..."
    Invoke-Python "-m pip install -r requirements.txt"
}

Write-Host "Setup complete. Activate the venv with:`n  .\.venv\Scripts\Activate.ps1"
