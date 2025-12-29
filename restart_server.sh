#!/bin/bash

# Script to restart the Flask server with updated code

echo "========================================================================="
echo "                    RESTARTING PDF Q&A SERVER"
echo "========================================================================="
echo ""

# Stop any running Flask processes
echo "[1/4] Stopping existing Flask processes..."
pkill -f "python.*app.py" 2>/dev/null
sleep 2

# Check if git repo
if [ -d ".git" ]; then
    echo "[2/4] Pulling latest changes from git..."
    git pull
else
    echo "[2/4] Not a git repository, skipping pull"
fi

# Activate virtual environment
echo "[3/4] Activating virtual environment..."
source venv/bin/activate

# Check if Pillow is installed
echo "[3.5/4] Checking Pillow installation..."
python3 -c "from PIL import Image" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing Pillow..."
    pip install Pillow
fi

# Start the Flask server
echo "[4/4] Starting Flask server..."
echo ""
echo "Server will start on http://0.0.0.0:5000"
echo "Press Ctrl+C to stop"
echo ""
echo "========================================================================="
echo ""

python3 app.py
