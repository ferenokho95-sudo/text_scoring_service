@echo off
cd /d "%~dp0"
echo Open http://127.0.0.1:5000 in your browser.
.venv\Scripts\mlflow.exe server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --allowed-hosts localhost:5000,127.0.0.1:5000
pause
