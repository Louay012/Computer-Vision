@echo off
setlocal enableextensions

echo Creating virtual environment...

REM Prefer 64-bit Python for scientific stack and torch wheels.
set "PY_CMD="
py -3.10-64 --version >nul 2>&1
if %errorlevel%==0 (
	set "PY_CMD=py -3.10-64"
) else (
	py -3-64 --version >nul 2>&1
	if %errorlevel%==0 (
		set "PY_CMD=py -3-64"
	)
)

if "%PY_CMD%"=="" (
	echo ERROR: 64-bit Python was not found via py launcher.
	echo Install 64-bit Python 3.10+ then rerun setup.
	exit /b 1
)

%PY_CMD% -m venv venv
if errorlevel 1 (
	echo ERROR: Failed to create virtual environment.
	exit /b 1
)

set "VENV_PY=venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
	echo ERROR: Virtual environment Python not found at %VENV_PY%.
	exit /b 1
)

echo Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
	echo ERROR: Failed to upgrade pip/setuptools/wheel.
	exit /b 1
)

echo Installing dependencies...
"%VENV_PY%" -m pip install --only-binary=:all: numpy pandas matplotlib seaborn ^
opencv-python pillow scikit-learn scikit-image notebook
if errorlevel 1 (
	echo ERROR: Failed to install base dependencies.
	exit /b 1
)

echo Installing PyTorch (CUDA 12.1)...
"%VENV_PY%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
	echo ERROR: Failed to install PyTorch CUDA wheels.
	exit /b 1
)

echo Saving dependencies...
"%VENV_PY%" -m pip freeze > requirements.txt
if errorlevel 1 (
	echo ERROR: Failed to write requirements.txt.
	exit /b 1
)

echo Setup complete.
echo Activate with: venv\Scripts\activate
pause