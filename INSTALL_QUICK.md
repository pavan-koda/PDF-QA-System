# Quick Installation Guide

## For Linux/macOS Users

If you're getting "ModuleNotFoundError: No module named 'fitz'", follow these steps:

### Method 1: Automatic (Recommended)

```bash
# Make scripts executable
chmod +x install_deps.sh start_vision_app.sh

# Install all dependencies
./install_deps.sh

# Run the application
./start_vision_app.sh
```

### Method 2: Manual Installation

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements_vision.txt

# Verify installation
python -c "import fitz; import chromadb; import flask; print('✓ All imports successful!')"

# Run application
python app_vision.py
```

### Method 3: If requirements_vision.txt fails

```bash
# Activate venv
source venv/bin/activate

# Install core dependencies one by one
pip install PyMuPDF==1.23.8
pip install chromadb==0.4.22
pip install Flask==3.0.0
pip install requests==2.31.0
pip install Pillow==10.1.0
pip install numpy==1.26.2
pip install torch==2.1.2
pip install transformers==4.36.2
pip install sentence-transformers==2.2.2
pip install faiss-cpu==1.7.4
pip install ollama==0.1.6

# Verify
python -c "import fitz; print('PyMuPDF installed!')"

# Run
python app_vision.py
```

## Common Issues

### Issue: "No module named 'fitz'"

**Solution:**
```bash
source venv/bin/activate
pip install PyMuPDF
```

### Issue: Virtual environment not activating

**Solution:**
```bash
# Delete old venv
rm -rf venv

# Create fresh one
python3 -m venv venv
source venv/bin/activate

# Install dependencies
./install_deps.sh
```

### Issue: Ollama not found

**Solution:**
```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# macOS
brew install ollama

# Then pull the model
ollama pull llama3.2-vision:11b
```

## Verify Installation

After installation, run:

```bash
python check_setup.py
```

This will show you:
- ✅ What's installed correctly
- ❌ What's missing
- 💡 How to fix issues

## Quick Start After Installation

```bash
# 1. Make sure Ollama is running
ollama serve &

# 2. Start the application
python app_vision.py

# 3. Open browser
# http://localhost:5000
```

## Need Help?

1. Check logs in `logs/` directory
2. Run `python check_setup.py` to diagnose
3. Review error messages carefully
4. Make sure you're using Python 3.9+

## All Working?

You should see:

```
========================================================================
Vision PDF QA System Starting
========================================================================
Ollama URL: http://localhost:11434
Vision Model: llama3.2-vision:11b
ColPali Enabled: True
========================================================================
Server running at: http://localhost:5000
========================================================================
```

Then open http://localhost:5000 and start uploading PDFs! 🎉
