@echo off
REM Double-click to start the Owed-Value Ledger as a desktop app.
REM First run: installs requirements and builds the demo data (about 1 minute). Later runs open instantly.
cd /d "%~dp0"
python -m pip install -q -r requirements.txt
python run.py --serve-only --app
pause
