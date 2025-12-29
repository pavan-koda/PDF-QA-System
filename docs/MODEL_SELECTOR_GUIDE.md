# Model Selector Guide

## What's New?

When you run `start_app.bat`, you'll now see an **interactive model selector** that lets you:

1. ✅ **Choose which AI models to use**
2. ✅ **See what's already downloaded**
3. ✅ **Automatically download models you select**
4. ✅ **Save your preferences** for future use
5. ✅ **Work completely OFFLINE** after first download

---

## How It Works

### First Time Startup

When you run `start_app.bat` for the first time:

```
========================================================================
                    PDF Q&A SYSTEM - STARTUP
========================================================================

[SETUP] Virtual environment not found. Creating one...
[SETUP] Installing dependencies...

========================================================================
                      MODEL CONFIGURATION
========================================================================

  PDF Q&A SYSTEM - MODEL CONFIGURATION

──────────────────────────────────────────────────────────────────────
SELECT EMBEDDING MODEL:
──────────────────────────────────────────────────────────────────────

[1] MiniLM (Recommended) (DEFAULT)
    Status: ⬇ NEEDS DOWNLOAD
    Size: ~80MB | Speed: Fast | Quality: Good
    Lightweight and fast, perfect for most use cases

[2] MPNet
    Status: ⬇ NEEDS DOWNLOAD
    Size: ~420MB | Speed: Medium | Quality: Excellent
    Higher quality embeddings, better semantic understanding

──────────────────────────────────────────────────────────────────────
Enter choice [1-2] (default: 1): _
```

### What You'll See

You'll be asked to select **3 types of models**:

#### 1️⃣ Embedding Model
- Converts text to numbers for semantic search
- **Recommended: MiniLM** (fast, small, good quality)
- Alternative: MPNet (larger, better quality)

#### 2️⃣ Question Answering Mode
- How the system answers your questions
- **Recommended: Smart Extractive** (0MB, no download, best for invoices)
- Alternatives:
  - DistilBERT (~250MB) - AI-powered, good for complex questions
  - RoBERTa (~500MB) - Most accurate

#### 3️⃣ Text Generator (Optional)
- For generating text (not recommended for QA)
- **Recommended: None** (more accurate for factual questions)
- Alternatives:
  - GPT-2 Small (~500MB)
  - GPT-2 Medium (~1.5GB)

---

## Recommended Configuration

### For Invoices/Bills/Structured Documents (Most Users):
```
Embedding Model: [1] MiniLM (Recommended)
QA Mode: [1] Smart Extractive (Recommended)
Generator: [1] None (Recommended)

Total Download: ~80MB
Works offline: ✅ Yes
Best for: Invoices, bills, receipts, forms
```

### For Research Papers/Complex Documents:
```
Embedding Model: [2] MPNet
QA Mode: [2] DistilBERT
Generator: [1] None

Total Download: ~670MB
Works offline: ✅ Yes
Best for: Research papers, articles, books
```

### For Maximum Accuracy:
```
Embedding Model: [2] MPNet
QA Mode: [3] RoBERTa
Generator: [1] None

Total Download: ~920MB
Works offline: ✅ Yes
Best for: When accuracy is critical
```

---

## Download Process

After you select models, the system will:

1. **Check** if models are already downloaded
2. **Download** any missing models (with progress)
3. **Save** your configuration to `model_config.json`
4. **Update** `config.py` with your selections
5. **Start** the application

Example:
```
========================================================================
DOWNLOADING REQUIRED MODELS
========================================================================

Downloading embeddings: all-MiniLM-L6-v2
Downloading embedding model...
✓ Successfully downloaded: all-MiniLM-L6-v2

========================================================================
✓ CONFIGURATION COMPLETE!
========================================================================

Your app is now ready to work OFFLINE!
All models are cached locally.

Configuration Summary:
  Embedding Model: MiniLM (Recommended)
  QA Mode: Smart Extractive (Recommended)
  Generator: No Generator (Recommended)

========================================================================
```

---

## Subsequent Startups

On future runs:

```
========================================================================
                      MODEL CONFIGURATION
========================================================================

Found existing configuration:
  Embedding: all-MiniLM-L6-v2
  QA Model: extractive
  Generator: none

Use this configuration? [Y/n]: _
```

**Options:**
- Press **Enter** or type **Y** → Uses saved config (fast)
- Type **n** → Opens model selector to choose different models

---

## Where Models Are Stored

Models are downloaded to:

1. **Local directory** (project folder):
   - GPT-2 models: `./gpt2/`, `./gpt2-medium/`

2. **HuggingFace cache** (`C:\Users\<YourName>\.cache\huggingface\`):
   - Embedding models
   - DistilBERT/RoBERTa models

**After download, everything works OFFLINE!** ✅

---

## Configuration Files

The system creates/updates these files:

- **`model_config.json`** - Your model selections (persistent)
- **`config.py`** - Application configuration (auto-generated)

---

## FAQ

### Q: Can I change models later?
**A:** Yes! Just run `start_app.bat` and choose **n** when asked to use existing config.

### Q: Will it re-download models?
**A:** No! Once downloaded, models are cached. The system checks if they exist before downloading.

### Q: What if I want to delete models?
**A:** Delete the model folders:
- Local: `./gpt2/`, `./gpt2-medium/`, etc.
- Cache: `C:\Users\<YourName>\.cache\huggingface\`

### Q: Which config should I use?
**A:** For invoices/bills:
- **Embedding:** MiniLM (fast, small)
- **QA Mode:** Smart Extractive (accurate for structured docs)
- **Generator:** None (better accuracy)

### Q: Does it work offline?
**A:** Yes! After the first download, everything runs locally without internet.

### Q: What if download fails?
**A:** The system will warn you but continue. You can:
- Check your internet connection
- Run the batch file again
- Choose different models

### Q: How much disk space do I need?
**A:** Depends on models:
- Minimal: ~80MB (MiniLM + Extractive)
- Recommended: ~330MB (MiniLM + DistilBERT)
- Maximum: ~2GB (MPNet + RoBERTa + GPT-2 Medium)

---

## Troubleshooting

### "Model download failed"
- Check internet connection
- Ensure enough disk space
- Try again or choose a smaller model

### "Configuration file error"
- Delete `model_config.json` and `config.py`
- Run `start_app.bat` again

### "Model not found" during runtime
- Re-run `start_app.bat`
- Choose **n** to reconfigure
- Select models again

---

## Advanced: Manual Configuration

If you want to skip the interactive selector, edit `config.py` directly:

```python
QA_CONFIG = {
    'use_advanced_qa': False,  # True for DistilBERT/RoBERTa
    'advanced_qa_model': 'distilbert-base-cased-distilled-squad',
}

EMBEDDING_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',  # or 'all-mpnet-base-v2'
}

GENERATOR_CONFIG = {
    'model_name': 'none',  # or 'gpt2', 'gpt2-medium'
    'use_generator': False,
}
```

---

**Ready to start?** Run `start_app.bat` and follow the prompts! 🚀
