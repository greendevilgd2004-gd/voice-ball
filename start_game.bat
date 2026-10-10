@echo off
cd /d "%~dp0"
venv\Scripts\python.exe run_game.py %*
pause
