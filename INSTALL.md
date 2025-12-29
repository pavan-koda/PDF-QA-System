# 📥 Complete Installation Guide - Vision PDF QA System

Step-by-step installation guide for all platforms.

## 🎯 What You're Installing

A complete AI-powered PDF analysis system that:
- Sees and understands diagrams, charts, and images
- Handles 500+ page documents
- Works 100% offline (after setup)
- Runs on your local machine

## 📋 Prerequisites

### Hardware Requirements

**Minimum:**
- CPU: Intel i5 / AMD Ryzen 5 (or equivalent)
- RAM: 16GB
- Storage: 20GB free space
- Internet: For initial download only

**Recommended:**
- CPU: Intel i7 / AMD Ryzen 7 (or better)
- RAM: 32GB
- Storage: 50GB free space
- GPU: NVIDIA with 8GB+ VRAM (optional, for faster processing)

### Software Requirements

- **Python 3.9 or higher** (3.10+ recommended)
- **Git** (optional, for cloning)
- **Internet connection** (for initial setup only)

---

## 🪟 Windows Installation

### Step 1: Install Python

1. **Download Python:**
   - Visit: https://www.python.org/downloads/
   - Download Python 3.10+ for Windows
   - Run the installer

2. **Important:** Check "Add Python to PATH"

3. **Verify installation:**
```bash
python --version
# Should show: Python 3.10.x or higher
```

### Step 2: Install Ollama

1. **Download Ollama:**
   - Visit: https://ollama.ai/download
   - Download Windows installer
   - Run the installer

2. **Verify installation:**
```bash
ollama --version
# Should show version number
```

3. **Ollama starts automatically** - check system tray

### Step 3: Download Llama 3.2-Vision

```bash
# Open Command Prompt or PowerShell
ollama pull llama3.2-vision:11b
```

This will download ~7GB. Takes 5-15 minutes depending on internet speed.

### Step 4: Get the Code

**Option A: Download ZIP**
1. Download this repository as ZIP
2. Extract to: `d:\A\LLM\GPT2(M)\PDF-QA-System`

**Option B: Clone with Git**
```bash
git clone <repository-url>
cd PDF-QA-System
```

### Step 5: Run Setup

```bash
# Navigate to project directory
cd "d:\A\LLM\GPT2(M)\PDF-QA-System"

# Run the startup script
start_vision_app.bat
```

The script will:
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Verify Ollama and models
- ✅ Start the application

### Step 6: Open Browser

```
http://localhost:5000
```

### Troubleshooting (Windows)

**Python not found:**
```bash
# Reinstall Python and check "Add to PATH"
# Or use full path:
C:\Users\YourName\AppData\Local\Programs\Python\Python310\python.exe app_vision.py
```

**Ollama not starting:**
```bash
# Start manually from system tray
# Or run: ollama serve
```

---

## 🍎 macOS Installation

### Step 1: Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install Python

```bash
# Install Python 3.10+
brew install python@3.10

# Verify
python3 --version
```

### Step 3: Install Ollama

```bash
# Install via Homebrew
brew install ollama

# Or download from: https://ollama.ai/download

# Verify
ollama --version
```

### Step 4: Start Ollama

```bash
# Ollama should start automatically
# If not, run:
ollama serve &
```

### Step 5: Download Llama 3.2-Vision

```bash
ollama pull llama3.2-vision:11b
```

### Step 6: Get the Code

```bash
# Clone or download
cd ~/Documents
git clone <repository-url>
cd PDF-QA-System
```

### Step 7: Run Setup

```bash
# Make script executable
chmod +x start_vision_app.sh

# Run
./start_vision_app.sh
```

### Step 8: Open Browser

```
http://localhost:5000
```

### Troubleshooting (macOS)

**Permission denied:**
```bash
chmod +x start_vision_app.sh
```

**Python command not found:**
```bash
# Use python3
python3 app_vision.py
```

**Ollama connection failed:**
```bash
# Check if running
curl http://localhost:11434/api/tags

# Start if needed
ollama serve &
```

---

## 🐧 Linux Installation (Ubuntu/Debian)

### Step 1: Update System

```bash
sudo apt update
sudo apt upgrade
```

### Step 2: Install Python

```bash
# Install Python 3.10+
sudo apt install python3.10 python3.10-venv python3-pip

# Verify
python3 --version
```

### Step 3: Install Ollama

```bash
# Official installation script
curl -fsSL https://ollama.ai/install.sh | sh

# Verify
ollama --version
```

### Step 4: Start Ollama Service

```bash
# Start Ollama
sudo systemctl start ollama

# Enable auto-start
sudo systemctl enable ollama

# Check status
sudo systemctl status ollama
```

### Step 5: Download Llama 3.2-Vision

```bash
ollama pull llama3.2-vision:11b
```

### Step 6: Get the Code

```bash
cd ~/
git clone <repository-url>
cd PDF-QA-System
```

### Step 7: Run Setup

```bash
# Make executable
chmod +x start_vision_app.sh

# Run
./start_vision_app.sh
```

### Step 8: Open Browser

```
http://localhost:5000
```

### Troubleshooting (Linux)

**Permission issues:**
```bash
# Fix permissions
chmod +x start_vision_app.sh

# Run with sudo if needed
sudo systemctl start ollama
```

**Port already in use:**
```bash
# Check what's using port 5000
sudo lsof -i :5000

# Kill process or use different port in app_vision.py
```

**Ollama not found:**
```bash
# Check if installed
which ollama

# Add to PATH if needed
export PATH=$PATH:/usr/local/bin
```

---

## 🔍 Verification

### Check Installation

Run the verification script:

```bash
python check_setup.py
```

This will check:
- ✅ Python version
- ✅ Required Python modules
- ✅ Ollama installation
- ✅ Ollama running status
- ✅ Llama 3.2-Vision model
- ✅ Project files
- ✅ Directory structure
- ✅ GPU availability (optional)

Expected output:
```
==================================================================
  VISION PDF QA SYSTEM - SETUP VERIFICATION
==================================================================

Checking Python environment...
[✓] Python                         OK
    Version: 3.10.11

Checking Python modules...
[✓] Module: flask                  OK
[✓] Module: PyMuPDF                OK
[✓] Module: chromadb               OK
...

Checking Ollama...
[✓] Ollama Installed               OK
    ollama version is 0.1.17
[✓] Ollama Running                 OK
    Server is active
[✓] Llama 3.2-Vision Model         OK
    Model found

==================================================================
  SUMMARY
==================================================================
Checks Passed: 15/15
Critical Checks: 10/10

✅ ALL CRITICAL CHECKS PASSED!
   You're ready to use the Vision PDF QA System!
```

### Manual Verification

**1. Test Python:**
```bash
python --version
# or
python3 --version
```

**2. Test Ollama:**
```bash
ollama list
# Should show llama3.2-vision:11b
```

**3. Test Ollama API:**
```bash
curl http://localhost:11434/api/tags
# Should return JSON with models list
```

**4. Test Dependencies:**
```bash
pip list | grep -E "flask|chromadb|transformers|torch"
```

---

## 🚀 First Run

### Option 1: Automated (Recommended)

**Windows:**
```bash
start_vision_app.bat
```

**macOS/Linux:**
```bash
./start_vision_app.sh
```

### Option 2: Manual

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements_vision.txt

# 4. Start Ollama (if not running)
ollama serve &

# 5. Run application
python app_vision.py
```

### Access the Application

1. **Open browser:** http://localhost:5000
2. **Upload a test PDF** (try one with diagrams)
3. **Ask a question:** "What diagrams are in this document?"

---

## ⚙️ Advanced Configuration

### GPU Acceleration (NVIDIA)

**Install CUDA PyTorch:**
```bash
# Uninstall CPU version
pip uninstall torch torchvision

# Install CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Use Larger Model (Better Quality)

```bash
# Download 90B model (~55GB)
ollama pull llama3.2-vision:90b

# Update app_vision.py:
model_name='llama3.2-vision:90b'
```

**Requires:** 64GB+ RAM

### Custom Ollama Configuration

```bash
# Create Modelfile
cat > Modelfile << EOF
FROM llama3.2-vision:11b
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER num_predict 1000
EOF

# Create custom model
ollama create my-vision-model -f Modelfile

# Use in app_vision.py
model_name='my-vision-model'
```

---

## 📦 Offline Installation

If you need to install on a machine without internet:

### 1. Download on Connected Machine

```bash
# Download Ollama model
ollama pull llama3.2-vision:11b

# Export model
ollama save llama3.2-vision:11b > llama-vision.tar

# Download Python packages
pip download -r requirements_vision.txt -d packages/
```

### 2. Transfer to Offline Machine

Copy:
- `llama-vision.tar`
- `packages/` directory
- Project files

### 3. Install on Offline Machine

```bash
# Install Ollama from downloaded installer

# Import model
ollama load < llama-vision.tar

# Install Python packages
pip install --no-index --find-links=packages/ -r requirements_vision.txt
```

---

## 🐛 Common Issues

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
pip install -r requirements_vision.txt
```

### Issue: "Connection refused to Ollama"

**Solution:**
```bash
# Start Ollama
ollama serve

# Check if running
curl http://localhost:11434/api/tags
```

### Issue: "Model not found"

**Solution:**
```bash
# Pull the model
ollama pull llama3.2-vision:11b

# Verify
ollama list | grep vision
```

### Issue: "Out of memory"

**Solution:**
```python
# Reduce DPI in vision_pdf_processor.py
processor = VisionPDFProcessor(dpi=100)  # Instead of 150

# Or use smaller batches
processor = VisionPDFProcessor(batch_size=5)
```

### Issue: "Port 5000 already in use"

**Solution:**
```python
# Change port in app_vision.py
app.run(host='0.0.0.0', port=5001)  # Use different port
```

---

## 📚 Post-Installation

### Recommended Next Steps

1. **Read the guides:**
   - [QUICK_START.md](QUICK_START.md) - Get started in 5 minutes
   - [README_VISION.md](README_VISION.md) - Full features and usage
   - [ARCHITECTURE.md](ARCHITECTURE.md) - Technical details

2. **Run verification:**
   ```bash
   python check_setup.py
   ```

3. **Test with sample PDF:**
   - Upload a PDF with diagrams
   - Ask: "What diagrams are in this document?"
   - Ask: "Explain the chart on page X"

4. **Check performance:**
   - Review: `logs/vision_performance.txt`
   - Monitor response times
   - Adjust DPI if needed

### Performance Tuning

**For Speed:**
- Lower DPI (100 instead of 150)
- Smaller batches (5 instead of 10)
- Use 11B model (not 90B)
- Enable GPU acceleration

**For Quality:**
- Higher DPI (200)
- Larger batches (20)
- Use 90B model
- More context (top_k=10)

---

## 🎓 Training & Examples

### Example PDFs to Try

1. **Technical Documentation**
   - Software architecture docs
   - API documentation
   - System design specs

2. **Financial Reports**
   - Annual reports
   - Financial statements
   - Investment analyses

3. **Research Papers**
   - Academic papers
   - Conference proceedings
   - Technical journals

4. **Legal Documents**
   - Contracts
   - Court filings
   - Patent documents

### Example Questions

```
# For diagrams
"Explain the architecture diagram on page 12"
"What does the flowchart show?"
"Describe the system topology"

# For charts
"What trends are shown in the revenue chart?"
"Analyze the Q4 performance graph"
"What does the data visualization indicate?"

# For tables
"What are the values in the comparison table?"
"Extract the pricing information from the table"
"What does the feature matrix show?"

# For images
"Describe the photo on page 15"
"What does the screenshot show?"
"Explain the product image"
```

---

## 🆘 Getting Help

### Resources

1. **Documentation:** Check all `.md` files in project
2. **Logs:** Review `logs/vision_performance.txt`
3. **Verification:** Run `python check_setup.py`

### Support Checklist

Before asking for help:
- [ ] Ran `check_setup.py`
- [ ] Verified Ollama is running
- [ ] Checked model is downloaded
- [ ] Reviewed error logs
- [ ] Tried troubleshooting steps above

---

## ✅ Installation Complete!

If all checks pass, you're ready to use the Vision PDF QA System!

**Start the application:**
```bash
# Windows
start_vision_app.bat

# macOS/Linux
./start_vision_app.sh
```

**Open browser:**
```
http://localhost:5000
```

**Upload a PDF and start asking questions!**

---

**Questions? Check:**
- [QUICK_START.md](QUICK_START.md)
- [README_VISION.md](README_VISION.md)
- [VISION_SETUP.md](VISION_SETUP.md)
