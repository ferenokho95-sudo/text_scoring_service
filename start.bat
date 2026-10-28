@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe goto install
py -3.12 -m venv .venv
if errorlevel 1 goto fail
:install
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail
if exist artifacts\model.joblib goto serve
.venv\Scripts\python.exe -m scripts.train
if errorlevel 1 goto fail
:serve
echo Open http://127.0.0.1:8000/docs in your browser.
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log
if errorlevel 1 goto fail
exit /b 0
:fail
echo Failed. Install Python 3.12 with Python Launcher and check the output above.
pause
exit /b 1
