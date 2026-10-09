@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\train.ps1" -Hmm
if errorlevel 1 echo HMM research stopped. Read and save the error above.
pause
