param([switch]$Resume, [switch]$AllowCpu, [switch]$Hmm, [switch]$Gui, [string]$TorchIndex = 'https://download.pytorch.org/whl/cu128')
$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $ProjectDir
$ToolsDir = Join-Path $ProjectDir '.tools'
New-Item -ItemType Directory -Path $ToolsDir -Force | Out-Null
$UvPath = Join-Path $ToolsDir 'uv.exe'
if (-not (Test-Path -LiteralPath $UvPath)) {
    $UvVersion = '0.12.21'
    $ArchiveName = 'uv-x86_64-pc-windows-msvc.zip'
    $ArchivePath = Join-Path $ToolsDir $ArchiveName
    $BaseUrl = "https://github.com/astral-sh/uv/releases/download/$UvVersion"
    Write-Host 'Installing isolated Python bootstrap from astral-sh/uv...'
    Invoke-WebRequest "$BaseUrl/$ArchiveName" -OutFile $ArchivePath
    $ChecksumPath = Join-Path $ToolsDir "$ArchiveName.sha256"
    Invoke-WebRequest "$BaseUrl/$ArchiveName.sha256" -OutFile $ChecksumPath
    $ExpectedHash = ((Get-Content -LiteralPath $ChecksumPath -Raw) -split '\s+')[0].Trim()
    $ActualHash = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash
    if ($ActualHash -ne $ExpectedHash) { throw 'Bootstrap SHA256 mismatch' }
    Expand-Archive -LiteralPath $ArchivePath -DestinationPath $ToolsDir -Force
}
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ToolsDir 'python'
$env:UV_CACHE_DIR = Join-Path $ToolsDir 'cache'
$PythonPath = Join-Path $ProjectDir '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $PythonPath)) {
    & $UvPath venv --python 3.12 .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed' }
}
if ($Gui) {
    # Settings window: installs the full training environment once, then all jobs run from the GUI.
    if ($AllowCpu) { $TorchIndex = 'https://download.pytorch.org/whl/cpu' }
    & $UvPath pip install --python $PythonPath --index-url $TorchIndex 'torch==2.7.1'
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed. Check GPU/CUDA installation instructions.' }
    & $UvPath pip install --python $PythonPath -r requirements-runtime.txt 'onnx==1.21.0' -e '.[hmm]'
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
    Write-Host 'Opening the training window. Keep this console open; errors appear here too.'
    & $PythonPath -m upbit_bot.cli gui
    if ($LASTEXITCODE -ne 0) { throw 'GUI exited with an error' }
    exit 0
}
if ($Hmm) {
    & $UvPath pip install --python $PythonPath -r requirements-runtime.txt -e '.[hmm]'
    if ($LASTEXITCODE -ne 0) { throw 'HMM dependency installation failed' }
    $HmmArgs = @('-m', 'upbit_bot.cli', 'train-hmm')
    if (Test-Path -LiteralPath (Join-Path $ProjectDir 'artifacts\manifest.json')) {
        Write-Host 'Comparing with saved LSTM on its original dates and saved candles. No download.'
        $HmmArgs += @('--compare-lstm', 'artifacts')
    } else {
        Write-Host 'No LSTM bundle found: HMM-only research. Run TRAIN.cmd, then HMM.cmd for comparison.'
        & $PythonPath -m upbit_bot.cli download
        if ($LASTEXITCODE -ne 0) { throw 'Download failed; HMM.cmd can resume it.' }
    }
    & $PythonPath @HmmArgs
    if ($LASTEXITCODE -ne 0) { throw 'HMM research failed. Review the error above.' }
    Write-Host 'DONE: artifacts\hmm\comparison.html' -ForegroundColor Green
    Start-Process -FilePath (Join-Path $ProjectDir 'artifacts\hmm\comparison.html')
    exit 0
}
if ($AllowCpu) { $TorchIndex = 'https://download.pytorch.org/whl/cpu' }
& $UvPath pip install --python $PythonPath --index-url $TorchIndex 'torch==2.7.1'
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed. Check GPU/CUDA installation instructions.' }
& $UvPath pip install --python $PythonPath -r requirements-runtime.txt 'onnx==1.21.0' -e .
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $PythonPath -m upbit_bot.cli doctor
if (-not $AllowCpu) {
    & $PythonPath -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU: update NVIDIA driver or run with -AllowCpu intentionally'"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA check failed before data download' }
}
if (-not $Resume) {
    & $PythonPath -m upbit_bot.cli download
    if ($LASTEXITCODE -ne 0) { throw 'Download failed. TRAIN.cmd resumes historical downloads automatically.' }
}
$TrainArgs = @('-m', 'upbit_bot.cli', 'train')
if ($Resume) { $TrainArgs += '--resume' }
if ($AllowCpu) { $TrainArgs += '--allow-cpu' }
& $PythonPath @TrainArgs
if ($LASTEXITCODE -ne 0) { throw 'Training failed. Use RESUME.cmd for unchanged data/config.' }
& $PythonPath -m upbit_bot.cli package
if ($LASTEXITCODE -ne 0) { throw 'Deployment packaging failed' }
Write-Host 'DONE: dist\oracle-upload.zip  |  Results: artifacts\report.html' -ForegroundColor Green
Start-Process -FilePath (Join-Path $ProjectDir 'artifacts\report.html')
