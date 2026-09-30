param([string]$Server, [string]$KeyPath)
$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $ProjectDir
$BundlePath = Join-Path $ProjectDir 'dist\oracle-upload.zip'
if (-not (Test-Path -LiteralPath $BundlePath)) { throw 'Run TRAIN.cmd first; model upload ZIP is missing.' }
if (-not $Server) { $Server = Read-Host 'Oracle Ubuntu server public IPv4 or DNS name' }
if ($Server -notmatch '^[A-Za-z0-9][A-Za-z0-9.-]+$') { throw 'Invalid server name' }
if (-not $KeyPath) { $KeyPath = Read-Host 'Path to your Oracle SSH private key (.key)' }
$KeyPath = (Resolve-Path -LiteralPath $KeyPath).Path
if (-not (Get-Command ssh.exe -ErrorAction SilentlyContinue)) { throw 'Install Windows OpenSSH Client first.' }
if (-not (Get-Command scp.exe -ErrorAction SilentlyContinue)) { throw 'Install Windows OpenSSH Client first.' }
$UploadName = 'upbit-upload-' + (Get-Date -Format 'yyyyMMddHHmmss')
Write-Host 'Verify the SSH host fingerprint against your Oracle instance before accepting a first connection.'
& scp.exe -i $KeyPath $BundlePath "ubuntu@${Server}:$UploadName.zip"
if ($LASTEXITCODE -ne 0) { throw 'Upload failed' }
$RemoteCommand = "sudo apt-get update && sudo apt-get install -y unzip && mkdir -p ~/$UploadName && unzip ~/$UploadName.zip -d ~/$UploadName && cd ~/$UploadName && sudo bash deploy/install.sh"
& ssh.exe -t -i $KeyPath "ubuntu@$Server" $RemoteCommand
if ($LASTEXITCODE -ne 0) { throw 'Server installation failed; inspect the output.' }
Write-Host 'Deployment finished. Paper mode is the default on first install. See README for real-order setup.' -ForegroundColor Green
