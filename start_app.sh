#!/bin/bash

# PDF Q&A System - Startup Script
# Bash equivalent of start_app.bat

# Color codes for terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo ""
echo "========================================================================"
echo "                    PDF Q&A SYSTEM - STARTUP"
echo "========================================================================"
echo ""

# Get server IP address
SERVER_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "$SERVER_IP" ]; then
    SERVER_IP="localhost"
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${BLUE}[SETUP]${NC} Virtual environment not found. Creating one..."
    echo ""
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Failed to create virtual environment${NC}"
        echo "Please ensure Python 3.8+ is installed"
        exit 1
    fi
    echo ""
    echo -e "${BLUE}[SETUP]${NC} Installing dependencies..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Failed to install dependencies${NC}"
        exit 1
    fi
    echo ""
    echo -e "${GREEN}[SETUP]${NC} Initial setup complete!"
    echo ""
else
    source venv/bin/activate
fi

# Check if config exists
if [ ! -f "config.py" ]; then
    echo ""
    echo "========================================================================"
    echo "                      FIRST-TIME MODEL CONFIGURATION"
    echo "========================================================================"
    echo ""
    python3 model_selector.py
    if [ $? -ne 0 ]; then
        echo ""
        echo "Model selection cancelled or failed."
        exit 1
    fi
else
    echo ""
    echo -e "${BLUE}[INFO]${NC} Existing configuration found"
    echo ""
    read -p "Do you want to reconfigure models? [y/n]: " RECONFIG
    if [[ "$RECONFIG" =~ ^[Yy]$ ]]; then
        echo ""
        echo "========================================================================"
        echo "                      MODEL RECONFIGURATION"
        echo "========================================================================"
        echo ""
        python3 model_selector.py
        if [ $? -ne 0 ]; then
            echo ""
            echo "Model selection cancelled or failed."
            exit 1
        fi
    else
        echo -e "${BLUE}[INFO]${NC} Using existing model configuration"
        echo ""
    fi
fi

# Check for missing dependencies (sentencepiece, protobuf for Mistral/Llama models)
echo ""
echo -e "${BLUE}[INFO]${NC} Checking for required dependencies..."
python3 -c "import sentencepiece" >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}[WARNING]${NC} sentencepiece not found - installing now..."
    echo ""
    pip install sentencepiece protobuf
    echo ""
    echo -e "${GREEN}[SUCCESS]${NC} Dependencies installed!"
    echo ""
fi

# Check if config uses gated models (Gemma, Llama) that need HuggingFace auth
NEEDS_AUTH=0
if [ -f "config.py" ]; then
    if grep -iq -E "gemma|llama" config.py; then
        NEEDS_AUTH=1
    fi
fi

# Only check HuggingFace auth if gated models are selected
if [ "$NEEDS_AUTH" -eq 1 ]; then
    echo ""
    echo "========================================================================"
    echo "                    HUGGINGFACE AUTHENTICATION CHECK"
    echo "========================================================================"
    echo ""
    echo -e "${BLUE}[INFO]${NC} Detected gated model (Gemma/Llama) - checking authentication..."
    echo ""
    huggingface-cli whoami >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}[WARNING]${NC} Not logged in to HuggingFace"
        echo ""
        echo "Your selected model requires HuggingFace authentication."
        echo ""
        echo "To proceed:"
        echo "  1. Get a token from: https://huggingface.co/settings/tokens"
        echo "  2. Accept model license at: https://huggingface.co/google/gemma-3-4b-it"
        echo ""
        read -p "Do you want to login to HuggingFace now? [y/n]: " DO_LOGIN
        if [[ "$DO_LOGIN" =~ ^[Yy]$ ]]; then
            echo ""
            echo -e "${BLUE}[INFO]${NC} Starting HuggingFace login..."
            echo "Paste your token when prompted (it won't be visible)"
            echo ""
            huggingface-cli login
            if [ $? -ne 0 ]; then
                echo ""
                echo -e "${YELLOW}[WARNING]${NC} Login failed or was skipped"
                echo "The app may fail to download the model"
                echo ""
            else
                echo ""
                echo -e "${GREEN}[SUCCESS]${NC} Successfully logged in to HuggingFace!"
                echo ""
            fi
        else
            echo -e "${YELLOW}[WARNING]${NC} Skipping login - model download may fail"
            echo ""
        fi
    else
        echo -e "${GREEN}[SUCCESS]${NC} Already logged in to HuggingFace"
        echo ""
    fi
fi

# Create necessary directories
mkdir -p uploads data logs static templates

echo ""
echo "========================================================================"
echo "                    STARTING APPLICATION"
echo "========================================================================"
echo ""
echo "Starting Flask server..."
echo ""
echo "Once started, open your browser to:"
echo "  > http://localhost:5000"
echo "  > http://127.0.0.1:5000"
if [ "$SERVER_IP" != "localhost" ]; then
    echo "  > http://$SERVER_IP:5000 (network access)"
fi
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "========================================================================"
echo ""

python3 app.py

echo ""
echo "Server stopped."
