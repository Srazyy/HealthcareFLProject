# FL-Healthcare setup (Windows / PowerShell)
# Run from the repo root: .\setup\setup_windows.ps1

Write-Host "== FL-Healthcare setup (Windows) =="

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "python not found on PATH. Install Python 3.10/3.11 from python.org first."
    exit 1
}

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

# Detect NVIDIA GPU via nvidia-smi
$hasGpu = $false
try {
    nvidia-smi | Out-Null
    $hasGpu = $true
} catch {
    $hasGpu = $false
}

if ($hasGpu) {
    Write-Host "NVIDIA GPU detected — installing CUDA-enabled torch (cu121)."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "No NVIDIA GPU detected — installing CPU-only torch."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

pip install -r requirements/base.txt

Write-Host ""
Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Verify GPU (CUDA) availability with:"
Write-Host "  python -c ""import torch; print(torch.cuda.is_available())"""
