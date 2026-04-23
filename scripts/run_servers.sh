#!/usr/bin/env bash
set -euo pipefail
# scripts/run_servers.sh - install deps and start backend/frontend (Unix/WSL/macOS)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ "${1-}" != "--skip-install" ]; then
  if [ -f requirements.txt ]; then
    echo "Starting Python deps install in background... (logs -> install-python.log)"
    (python -m pip install --upgrade pip && python -m pip install -r requirements.txt) > install-python.log 2>&1 &
  fi
  if [ -f frontend/package.json ]; then
    echo "Starting frontend deps install in background... (logs -> frontend/install-frontend.log)"
    (cd frontend && (npm ci || npm install)) > frontend/install-frontend.log 2>&1 &
  fi
fi

echo "Starting backend in background..."
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

echo "Starting frontend (foreground)..."
(cd frontend && npm run dev)

echo "Frontend exited, waiting for backend (pid $BACKEND_PID)"
wait $BACKEND_PID || true
