#!/bin/bash
# Instant dependency installer - just run this after fresh clone

echo "Installing Vision PDF QA dependencies..."

# Activate venv (assumes it exists)
if [ -d "venv" ]; then
    source venv/bin/activate
else
    python3 -m venv venv
    source venv/bin/activate
fi

# Install everything at once
pip install -q --upgrade pip

# Install all packages in one command (faster)
pip install -q \
    Flask==3.0.0 \
    Werkzeug==3.0.1 \
    PyMuPDF==1.23.8 \
    pypdf==3.17.4 \
    Pillow==10.1.0 \
    chromadb==0.4.22 \
    hnswlib==0.8.0 \
    pydantic==2.5.3 \
    transformers==4.36.2 \
    torch==2.1.2 \
    torchvision==0.16.2 \
    sentence-transformers==2.2.2 \
    faiss-cpu==1.7.4 \
    ollama==0.1.6 \
    requests==2.31.0 \
    numpy==1.26.2 \
    tqdm==4.66.1 \
    python-dotenv==1.0.0 \
    python-json-logger==2.0.7 \
    colorlog==6.8.0

echo "✅ All dependencies installed!"
echo ""
echo "Now run: python app_vision.py"
