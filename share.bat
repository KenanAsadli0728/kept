@echo off
REM Double-click to run the app on this computer and let your PHONE open it (same Wi-Fi).
REM Edit the PIN below (or leave it empty for no PIN).
set PIN=1234
cd /d "%~dp0"
python -m pip install -q -r requirements.txt
python run.py --serve-only --share --pin %PIN% --open
pause
