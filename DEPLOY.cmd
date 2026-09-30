@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\deploy.ps1"
if errorlevel 1 echo Deployment stopped. Read the error above.
pause
