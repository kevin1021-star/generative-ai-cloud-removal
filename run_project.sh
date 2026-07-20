#!/bin/bash
echo "==================================================="
echo "🛰️ Launching PG-SMDNet Full-Stack Presentation Stack"
echo "==================================================="

# Start Backend
echo "1. Starting FastAPI Backend on port 8000..."
python api.py &
BACKEND_PID=$!

# Start Frontend
echo "2. Launching React Frontend on port 5173..."
cd frontend
npm run dev &
FRONTEND_PID=$!

# Open Browser
echo "3. Opening browser to React presentation portal..."
sleep 4
if [ "$(uname)" == "Darwin" ]; then
    open http://localhost:5173
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    xdg-open http://localhost:5173
fi

# Cleanup background tasks on exit
trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
