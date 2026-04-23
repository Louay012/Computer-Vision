# Run servers helper

This repository includes helper scripts to install dependencies and run the backend (FastAPI) and frontend (Vite) dev servers.

Windows (PowerShell):

- Run installs and start servers:

  powershell -ExecutionPolicy Bypass -File .\run_servers.ps1

- Skip installs:

  powershell -ExecutionPolicy Bypass -File .\run_servers.ps1 -SkipInstall

Windows (cmd):

- Run installs and start servers:

  .\run_servers.bat

- Skip installs:

  .\run_servers.bat --skip-install

WSL / macOS / Linux:

- Run installs and start servers:

  ./scripts/run_servers.sh

- Skip installs:

  ./scripts/run_servers.sh --skip-install

Notes:
- Ensure `python`, `pip`, and `npm` are in your PATH.
- The PowerShell script will attempt to activate `venv\` if present.
- Backend will run on http://127.0.0.1:8000 and frontend on http://localhost:5173 by default.
