@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\train.ps1"
if errorlevel 1 echo Training stopped. Read the error above. Do not close until you have saved it.
pause
