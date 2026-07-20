@echo off
echo ===================================================
echo 🛰️ Launching PG-SMDNet Full-Stack Presentation Stack
echo ===================================================

echo 1. Starting FastAPI Backend on port 8000...
start /B python api.py

echo 2. Launching React Frontend on port 5173...
cd frontend
start /B npm run dev

echo 3. Opening browser to React presentation portal...
timeout /t 4 >nul
start http://localhost:5173

echo ===================================================
echo Presentation stack is live! Close this terminal to exit.
echo ===================================================
pause
