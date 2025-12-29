# Model Handling & Storage Guide

## ✅ Models are EXCLUDED from Git

Your `.gitignore` is configured to **automatically exclude all models** from being uploaded to GitHub. This saves storage and speeds up deployment.

---

## 🚫 What Gets EXCLUDED (Not Uploaded)

These are automatically ignored by Git:

### Large Model Files:
- ✅ `models/` - Model directory
- ✅ `gpt2/`, `gpt2-medium/`, `gpt2-large/`, `gpt2-xl/`
- ✅ `distilbert*/`, `roberta*/`, `bert*/`
- ✅ `t5*/`, `flan*/`, `gemma*/`, `llama*/`, `mistral*/`
- ✅ `.cache/` - HuggingFace cache

### Model File Types:
- ✅ `*.bin` - PyTorch binary files (largest files!)
- ✅ `*.safetensors` - SafeTensors format
- ✅ `*.gguf` - GGUF format (llama.cpp)
- ✅ `*.h5` - Keras/TensorFlow weights
- ✅ `*.onnx` - ONNX format
- ✅ `*.pb` - TensorFlow protobuf
- ✅ `*.msgpack` - MessagePack files

### Runtime Data:
- ✅ `uploads/` - User PDFs
- ✅ `data/` - Processed chunks & embeddings
- ✅ `logs/` - Log files
- ✅ `*.pkl`, `*.faiss` - Index files
- ✅ `venv/` - Virtual environment

---

## ✅ What Gets UPLOADED (Small Files Only)

Only your **code** goes to GitHub:

```
✅ app.py                    - Flask application
✅ config.py                 - Configuration
✅ qa_engine.py              - QA logic
✅ pdf_processor.py          - PDF processing
✅ requirements.txt          - Dependencies list
✅ start_app.sh              - Startup script
✅ templates/                - HTML templates
✅ static/                   - CSS/JS files
✅ README.md                 - Documentation
```

**Total size**: ~100KB (very small!)

---

## 📥 How Models Get Downloaded

Models are downloaded **automatically on the server** when the app first runs:

### On First Run:

```bash
# When you run on the server
./start_app.sh

# The app automatically downloads models from HuggingFace:
# 1. Sentence transformer: "all-MiniLM-L6-v2" (~90MB)
# 2. QA model (if enabled): "distilbert-base-cased-distilled-squad" (~260MB)
# 3. Generator (if enabled): "gpt2" or custom model
```

### Where Models are Stored on Server:

```
~/.cache/huggingface/        # HuggingFace cache (auto-created)
~/pdf-qa-system/models/      # Local models (if any)
```

---

## 🔍 Verify What Will Be Uploaded

Before pushing to GitHub, check what Git will upload:

```bash
# See what will be committed
git status

# See ignored files (should include models)
git status --ignored

# Check file sizes that would be uploaded
git ls-files | xargs ls -lh
```

If you see large files (>10MB), they should NOT be uploaded!

---

## 🛡️ Model Download Strategy

### Current Setup (Recommended):

**Your config.py:**
```python
GENERATOR_CONFIG = {
    'model_name': 'none',      # No large generator model
    'use_generator': False,
}

EMBEDDING_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',  # Small embedding model (~90MB)
}
```

**Download on server:**
- ✅ Embedding model: Auto-downloaded (~90MB, one-time)
- ✅ Fast download: ~30 seconds on good connection
- ✅ Cached: Only downloads once

---

## 🚀 Deployment Workflow

### Step 1: Push Code Only (Windows)

```bash
git add .
git commit -m "Update application"
git push
```

**What's uploaded**: Only code files (~100KB)
**What's NOT uploaded**: Models, data, logs (excluded by .gitignore)

### Step 2: Models Download on Server

```bash
# On server
git pull
./start_app.sh
```

**First time:**
- ✅ Creates virtual environment
- ✅ Installs Python packages
- ✅ Downloads models from HuggingFace (~90MB)
- ⏱️ Takes 2-5 minutes (one-time setup)

**Subsequent updates:**
- ✅ Just pulls code changes
- ✅ Models already cached
- ⏱️ Takes 5-10 seconds

---

## 📊 Storage Requirements

### GitHub Repository:
```
Code only: ~100KB
✅ Very small, fast uploads
```

### Server Storage:
```
Application code:        ~100KB
Dependencies (venv):     ~500MB
Models (cache):          ~90MB  (embedding model)
Runtime data:            Variable (depends on usage)
---
Total initial:           ~600MB
```

### With Optional Large Models:
```
GPT-2 Medium:            +1.5GB
GPT-2 Large:             +3GB
T5-Large:                +3GB
---
Only install if needed!
```

---

## ⚠️ Troubleshooting

### If Models Were Accidentally Added:

```bash
# Remove from Git tracking (keeps local file)
git rm --cached -r models/
git rm --cached -r gpt2-medium/
git rm --cached *.bin
git rm --cached *.safetensors

# Commit the removal
git commit -m "Remove large model files from Git"

# Push
git push
```

### If .gitignore Wasn't Working:

```bash
# Clear Git cache and re-add files
git rm -r --cached .
git add .
git commit -m "Apply .gitignore rules"
git push
```

### If Download Fails on Server:

```bash
# Manually download specific model
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Or set cache directory
export HF_HOME=/path/to/cache
./start_app.sh
```

---

## 🎯 Best Practices

### ✅ DO:
- Use small, efficient models (all-MiniLM-L6-v2)
- Let HuggingFace handle model downloads
- Keep models in cache directory
- Use `.gitignore` to exclude models

### ❌ DON'T:
- Commit `.bin`, `.safetensors` files to Git
- Upload models to GitHub
- Store models in project directory
- Use huge models unless necessary

---

## 📝 Model Configuration Options

### Option 1: No Generator (Fastest, Smallest)
```python
GENERATOR_CONFIG = {
    'model_name': 'none',
    'use_generator': False,
}
```
**Size**: ~90MB total
**Speed**: Fast
**Accuracy**: Good (extractive answers)

### Option 2: With Small Generator
```python
GENERATOR_CONFIG = {
    'model_name': 'gpt2',  # 500MB
    'use_generator': True,
}
```
**Size**: ~600MB total
**Speed**: Medium
**Accuracy**: Better (generative answers)

### Option 3: With Advanced QA
```python
QA_CONFIG = {
    'use_advanced_qa': True,
    'advanced_qa_model': 'distilbert-base-cased-distilled-squad',  # 260MB
}
```
**Size**: ~350MB total
**Speed**: Medium-Fast
**Accuracy**: Better (BERT-based QA)

---

## 🔄 Summary

| Item | Uploaded to Git? | Downloaded on Server? |
|------|------------------|----------------------|
| **Code files** | ✅ Yes (~100KB) | ✅ Yes (via git pull) |
| **Models** | ❌ No (excluded) | ✅ Yes (auto-download) |
| **Dependencies** | 📝 List only (requirements.txt) | ✅ Yes (pip install) |
| **User data** | ❌ No (excluded) | 🔄 Created at runtime |

**Result**: Fast uploads, efficient deployment, no storage waste! ✨

---

## 🎉 You're All Set!

Your `.gitignore` is properly configured to:
- ✅ Exclude all model files
- ✅ Exclude runtime data
- ✅ Keep repository small
- ✅ Enable fast deployments

Just use `git push` and models will download automatically on the server!
