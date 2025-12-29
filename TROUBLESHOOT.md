# Troubleshooting Guide

## Problem: Script Hangs at "Checking dependencies..."

This happens when Python import checks take too long.

### Solution 1: Use Quick Install (Fastest)

```bash
# Cancel the hanging script (Ctrl+C)

# Run quick install instead
chmod +x quick_install.sh
./quick_install.sh

# Then start app directly
python app_vision.py
```

### Solution 2: Manual Install (Most Reliable)

```bash
# Activate virtual environment
source venv/bin/activate

# Install packages manually
pip install Flask PyMuPDF chromadb transformers torch sentence-transformers faiss-cpu numpy ollama requests

# Run app
python app_vision.py
```

### Solution 3: Fresh Start

```bash
# Delete virtual environment
rm -rf venv

# Run quick install
chmod +x quick_install.sh
./quick_install.sh

# Run app
python app_vision.py
```

---

## Problem: "No module named 'fitz'"

### Solution:

```bash
source venv/bin/activate
pip install PyMuPDF
python app_vision.py
```

---

## Problem: "No module named 'chromadb'"

### Solution:

```bash
source venv/bin/activate
pip install chromadb
python app_vision.py
```

---

## Problem: Ollama Connection Failed

### Solution:

```bash
# Check if Ollama is installed
ollama --version

# If not, install it:
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve &

# Download model
ollama pull llama3.2-vision:11b

# Try app again
python app_vision.py
```

---

## Problem: Port 5000 Already in Use

### Solution:

```bash
# Find what's using port 5000
sudo lsof -i :5000

# Kill it (replace PID with actual number)
kill -9 PID

# Or use different port - edit app_vision.py:
# Change: app.run(host='0.0.0.0', port=5000)
# To:     app.run(host='0.0.0.0', port=5001)
```

---

## Quick Commands Reference

### Install Everything Fast
```bash
./quick_install.sh
```

### Install Step by Step
```bash
source venv/bin/activate
pip install PyMuPDF chromadb Flask torch transformers sentence-transformers faiss-cpu numpy ollama
```

### Verify Installation
```bash
source venv/bin/activate
python -c "import fitz; import chromadb; import flask; print('✓ All working!')"
```

### Start Application
```bash
source venv/bin/activate
python app_vision.py
```

### Check Ollama
```bash
ollama list
curl http://localhost:11434/api/tags
```

---

## Still Not Working?

1. **Check Python version:**
   ```bash
   python3 --version
   # Should be 3.9 or higher
   ```

2. **Check if you're in virtual environment:**
   ```bash
   which python
   # Should show: .../venv/bin/python
   ```

3. **Reinstall everything:**
   ```bash
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   ./quick_install.sh
   ```

4. **Check error logs:**
   ```bash
   # Run app and save errors
   python app_vision.py 2>&1 | tee error.log
   # Check error.log file
   ```

---

## Working Installation Looks Like:

```bash
$ python app_vision.py

INFO:root:Initializing Vision QA Engine with llama3.2-vision:11b
INFO:root:Ollama server is running
INFO:root:Model llama3.2-vision:11b is available
========================================================================
Vision PDF QA System Starting
========================================================================
Ollama URL: http://localhost:11434
Vision Model: llama3.2-vision:11b
========================================================================
Server running at: http://localhost:5000
========================================================================
 * Running on http://127.0.0.1:5000
```

Then open: **http://localhost:5000**

---

## Emergency: Nuclear Option

If nothing works, start completely fresh:

```bash
# Go to home directory
cd ~

# Delete everything
rm -rf PDF-QA-System

# Clone fresh
git clone https://github.com/pavan-koda/PDF-QA-System.git
cd PDF-QA-System

# Quick install
chmod +x quick_install.sh
./quick_install.sh

# Run
python app_vision.py
```

---

**Most common fix: Just use `quick_install.sh` - it's the fastest and most reliable!**
