@echo off
setlocal

cd /d "%~dp0"

echo [0/4] Freeing ports 5173 and 8000 if needed...
for %%P in (5173 8000) do (
    for /f "tokens=5" %%I in ('netstat -ano ^| findstr /r /c":%%P .*LISTENING"') do (
        echo   - Stopping process %%I on port %%P
        taskkill /F /PID %%I >nul 2>nul
    )
)

echo [1/4] Installing backend dependencies...
if exist "%~dp0venv\Scripts\activate.bat" (
    call "%~dp0venv\Scripts\activate.bat"
)
if exist "%~dp0requirements.txt" (
    echo Installing Python dependencies from requirements.txt...
    python -m pip install --upgrade pip
    python -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 goto :error
)

echo [2/4] Installing frontend dependencies...
cd /d "%~dp0frontend"
call npm ci || call npm install
if errorlevel 1 goto :error

echo [3/4] Starting backend server in a new terminal...
start "PlantDisease Backend" cmd /k cd /d "%~dp0" ^&^& if exist "%~dp0venv\Scripts\python.exe" ("%~dp0venv\Scripts\python.exe" -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000) else (python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000)

echo [4/4] Starting frontend server in a new terminal...
start "PlantDisease Frontend" cmd /k cd /d "%~dp0frontend" ^&^& npm run dev

echo Servers launched.
exit /b 0

:error
echo Failed to install dependencies.
exit /b 1