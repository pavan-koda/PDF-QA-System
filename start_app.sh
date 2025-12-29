#!/bin/bash

echo "Starting PDF Q&A System..."
echo ""

# Get server IP address
SERVER_IP=$(hostname -I | awk '{print $1}')
if [ -z "$SERVER_IP" ]; then
    SERVER_IP="localhost"
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo ""
    echo "Installing dependencies..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo ""
    echo "Setup complete!"
else
    source venv/bin/activate
fi

# Create necessary directories
mkdir -p uploads data logs static templates

echo ""
echo "Starting Flask application..."
echo "Server will be accessible at:"
echo "  - Local:   http://localhost:5000"
echo "  - Network: http://$SERVER_IP:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the application
python3 app.py
