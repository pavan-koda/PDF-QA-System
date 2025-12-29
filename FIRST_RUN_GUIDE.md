# First Run Guide - What Happens on the Server

## 📥 What Gets Downloaded Automatically

When you run `./start_app.sh` on the server for the first time, here's what happens:

---

## Step-by-Step First Run

### 1️⃣ **Git Clone** (Your Code)
```bash
git clone https://github.com/YOUR_USERNAME/pdf-qa-system.git
cd pdf-qa-system
```

**Downloads**: ~100KB (application code only)
**Time**: ~5 seconds

---

### 2️⃣ **Run Startup Script**
```bash
./start_app.sh
```

This script does the following:

#### A. Creates Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```
**Time**: ~10 seconds

#### B. Installs Python Packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Downloads from PyPI:**
- Flask, Werkzeug (web framework)
- PyPDF (PDF processing)
- **PyTorch** (~800MB) ⚠️ Largest download
- **Transformers** (~500MB)
- **sentence-transformers** (~100MB)
- faiss-cpu (vector search)
- numpy, other utilities

**Total Download**: ~1.5GB
**Time**: 2-5 minutes (depends on server speed)

#### C. Downloads AI Models (Automatic)

When the app first starts, **sentence-transformers** automatically downloads:

**Embedding Model**: `all-MiniLM-L6-v2`
- **Size**: ~90MB
- **Location**: `~/.cache/huggingface/hub/`
- **Time**: ~30 seconds
- **One-time**: Cached for future runs

**Optional QA Model** (if `use_advanced_qa: True`):
- **Model**: `distilbert-base-cased-distilled-squad`
- **Size**: ~260MB
- **Location**: `~/.cache/huggingface/hub/`
- **Time**: ~1 minute

---

## 📊 Total First Run Downloads

| Component | Size | Download From | Cached? |
|-----------|------|---------------|---------|
| **Code (Git)** | ~100KB | GitHub | ✅ Yes |
| **Python Packages** | ~1.5GB | PyPI | ✅ Yes (venv) |
| **Embedding Model** | ~90MB | HuggingFace | ✅ Yes (cache) |
| **QA Model (optional)** | ~260MB | HuggingFace | ✅ Yes (cache) |
| **Total First Run** | **~1.7GB** | Multiple sources | ✅ All cached |

**Time**: 3-7 minutes (one-time only)

---

## 🔄 Subsequent Runs (After Updates)

When you update code and run `git pull`:

```bash
cd ~/pdf-qa-system
git pull                          # ~5 seconds (code only)
./start_app.sh                    # ~10 seconds (uses cache)
```

**Downloads**: ~0 bytes (everything cached!)
**Time**: ~15 seconds total ⚡

---

## 💾 Where Everything Gets Stored

### On Server After First Run:

```
~/pdf-qa-system/                  # Your project
├── app.py, config.py, etc.       # Code (~100KB)
├── venv/                         # Python packages (~1.5GB)
├── uploads/                      # User PDFs (created at runtime)
├── data/                         # Processed data (created at runtime)
└── logs/                         # Application logs

~/.cache/huggingface/             # HuggingFace models cache
├── hub/
│   ├── all-MiniLM-L6-v2/        # Embedding model (~90MB)
│   └── distilbert-*/             # QA model (~260MB, if enabled)

Total Disk Space: ~2GB
```

---

## 🚀 Optimization Tips

### To Reduce Download Size:

#### 1. **Use CPU-only PyTorch** (Smaller)
Current `requirements.txt` uses:
```
torch>=2.2.0  # ~800MB (includes CUDA support)
```

For CPU-only server (smaller):
```bash
# Edit requirements.txt, replace torch line with:
torch==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```
**Saves**: ~400MB

#### 2. **Disable Optional Packages**
Comment out in `requirements.txt`:
```
# langchain==0.1.0           # Save ~100MB
# langchain-community==0.0.10
# tiktoken==0.5.2
```

#### 3. **Use Smaller Embedding Model**
In `config.py`:
```python
EMBEDDING_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',  # Current: 90MB
    # Alternative smaller option:
    # 'model_name': 'paraphrase-MiniLM-L3-v2',  # Only 60MB
}
```

---

## 🔍 How to Monitor Downloads

### Watch First Run Progress:

```bash
# Run with verbose output
./start_app.sh

# Or manually to see each step:
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --verbose
python3 app.py
```

### Check Downloaded Models:

```bash
# See HuggingFace cache
ls -lh ~/.cache/huggingface/hub/

# See model sizes
du -sh ~/.cache/huggingface/hub/*

# Check total project size
du -sh ~/pdf-qa-system/
```

---

## ⚠️ Important Notes

### 1. **Models Are NOT in Git**
- ✅ Models excluded by `.gitignore`
- ✅ Downloaded from HuggingFace (not GitHub)
- ✅ Git only stores code (~100KB)

### 2. **First Run Takes Longer**
- ✅ First run: 3-7 minutes (downloads everything)
- ✅ Subsequent runs: ~15 seconds (uses cache)
- ✅ After `git pull`: ~15 seconds (no re-download)

### 3. **Cache is Shared**
- ✅ Models stored in `~/.cache/huggingface/`
- ✅ Shared across all HuggingFace projects
- ✅ Download once, use everywhere

### 4. **Internet Required for First Run**
- ✅ Need internet to download packages & models
- ✅ After setup, can run offline (if no new models)

---

## 🐛 Troubleshooting First Run

### Download Fails or Times Out:

```bash
# Increase timeout and retry
pip install -r requirements.txt --timeout 300

# Install packages one by one to isolate issue
pip install Flask==3.0.0
pip install torch
pip install transformers
pip install sentence-transformers
```

### Model Download Fails:

```bash
# Manually download embedding model
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Check HuggingFace cache
ls -la ~/.cache/huggingface/hub/
```

### Out of Disk Space:

```bash
# Check available space
df -h

# Clean pip cache if needed
pip cache purge

# Clean HuggingFace cache (careful!)
rm -rf ~/.cache/huggingface/hub/*
```

---

## ✅ Verification After First Run

```bash
# Check if virtual environment exists
ls venv/

# Check if packages installed
source venv/bin/activate
pip list | grep -E "torch|transformers|sentence"

# Check if models downloaded
ls ~/.cache/huggingface/hub/

# Test the app
./start_app.sh
# Should see: "Server will be accessible at: http://172.16.20.12:5000"
```

---

## 📝 Summary

### What the `.gitignore` Excludes:
```
❌ venv/                 # Not in Git, pip installs it
❌ models/               # Not in Git, auto-downloaded
❌ *.bin, *.safetensors  # Not in Git, auto-downloaded
❌ .cache/               # Not in Git, auto-downloaded
```

### What Happens on First Run:
```
1. git clone          → Downloads code (~100KB)
2. pip install        → Downloads packages (~1.5GB)
3. App starts         → Downloads models (~90MB)

Total: ~1.7GB one-time download
```

### What Happens on Updates:
```
1. git pull           → Updates code (~KB)
2. ./start_app.sh     → Uses cache (no download)

Total: ~15 seconds ⚡
```

---

## 🎯 Key Takeaway

**The `.gitignore` is working perfectly!**

- ✅ Code goes to Git (~100KB)
- ✅ Models download from HuggingFace (~90MB)
- ✅ Packages download from PyPI (~1.5GB)
- ✅ Everything cached after first run
- ✅ Updates are fast (no re-download)

**First run takes 5 minutes. Every update after that takes 15 seconds!** ⚡

---

For deployment, just follow:
1. **[QUICK_START.md](QUICK_START.md)** - Commands to run
2. **This file** - Understanding what downloads
3. **[MODEL_HANDLING.md](MODEL_HANDLING.md)** - Model details
