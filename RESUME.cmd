@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\train.ps1" -Resume
if errorlevel 1 echo Resume stopped. Read the error above.
pause
