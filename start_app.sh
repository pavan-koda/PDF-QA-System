#!/bin/bash

# PDF Q&A System - Startup Script
# Bash equivalent of start_app.bat

echo ""
echo "========================================================================"
echo "                    PDF Q&A SYSTEM - STARTUP"
echo "========================================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[SETUP] Virtual environment not found. Creating one..."
    echo ""
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        echo "Please ensure Python 3.8+ is installed"
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo ""
    echo "[SETUP] Installing dependencies..."
    source venv/bin/activate
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo ""
    echo "[SETUP] Initial setup complete!"
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
        read -p "Press Enter to exit..."
        exit 1
    fi
else
    echo ""
    echo "[INFO] Existing configuration found"
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
            read -p "Press Enter to exit..."
            exit 1
        fi
    else
        echo "[INFO] Using existing model configuration"
        echo ""
    fi
fi

# Check for missing dependencies (sentencepiece, protobuf for Mistral/Llama models)
echo ""
echo "[INFO] Checking for required dependencies..."
venv/bin/python3 -c "import sentencepiece" >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "[WARNING] sentencepiece not found - installing now..."
    echo ""
    pip install sentencepiece protobuf
    echo ""
    echo "[SUCCESS] Dependencies installed!"
    echo ""
fi

# Check if config uses gated models (Gemma, Llama) that need HuggingFace auth
NEEDS_AUTH=0
if [ -f "config.py" ]; then
    # Check in config.py file
    if grep -iq -E "gemma|llama" config.py; then
        NEEDS_AUTH=1
    fi
    # Also check using Python to import the config
    if python3 -c "from config import GENERATOR_CONFIG; model=GENERATOR_CONFIG.get('model_name',''); exit(0 if 'gemma' in model.lower() or 'llama' in model.lower() else 1)" 2>/dev/null; then
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
    echo "[INFO] Detected gated model (Gemma/Llama) - checking authentication..."
    echo ""
    huggingface-cli whoami >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "[WARNING] Not logged in to HuggingFace"
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
            echo "[INFO] Starting HuggingFace login..."
            echo "Paste your token when prompted (it won't be visible)"
            echo ""
            huggingface-cli login
            if [ $? -ne 0 ]; then
                echo ""
                echo "[WARNING] Login failed or was skipped"
                echo "The app may fail to download the model"
                echo ""
            else
                echo ""
                echo "[SUCCESS] Successfully logged in to HuggingFace!"
                echo ""
            fi
        else
            echo "[WARNING] Skipping login - model download may fail"
            echo ""
        fi
    else
        echo "[SUCCESS] Already logged in to HuggingFace"
        echo ""
    fi
fi

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
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "========================================================================"
echo ""

python3 app.py

read -p "Press Enter to exit..."
