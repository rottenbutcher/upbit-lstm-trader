@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\train.ps1" -Gui
if errorlevel 1 echo GUI stopped. Read the error above.
pause
