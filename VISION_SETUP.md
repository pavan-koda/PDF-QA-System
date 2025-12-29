# Vision-Based PDF QA System Setup Guide

Complete guide for setting up the advanced vision-based PDF QA system with Llama 3.2-Vision and ColPali for handling 500+ page documents with images, diagrams, and complex layouts.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
- [Model Setup](#model-setup)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

## Overview

This system uses:

1. **Llama 3.2-Vision (11B)** via Ollama - Multimodal AI that understands images AND text
2. **ColPali** - Visual document retriever that "sees" your PDF pages
3. **ChromaDB** - Multimodal vector database for fast search
4. **PyMuPDF** - Advanced PDF processing with image extraction

### Why This Approach?

Traditional PDF QA systems only read text. This system:
- ✅ Sees diagrams, charts, and images
- ✅ Understands table layouts and visual structure
- ✅ Handles 500+ page documents efficiently
- ✅ Works with unstructured, complex documents
- ✅ Retrieves exact pages in seconds using visual similarity

## Prerequisites

### System Requirements

**Minimum:**
- Python 3.9 or higher
- 16GB RAM
- 20GB free disk space
- CPU with AVX2 support

**Recommended:**
- Python 3.10+
- 32GB RAM
- NVIDIA GPU with 8GB+ VRAM (optional but faster)
- 50GB free disk space

### Operating Systems

- ✅ Windows 10/11
- ✅ macOS 10.15+
- ✅ Linux (Ubuntu 20.04+, Debian, etc.)

## Installation Steps

### 1. Install Ollama

Ollama runs Llama 3.2-Vision locally on your machine.

**Windows:**
```bash
# Download from: https://ollama.ai/download
# Run the installer

# Verify installation
ollama --version
```

**macOS:**
```bash
# Using Homebrew
brew install ollama

# Or download from: https://ollama.ai/download
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Start Ollama Server

**Windows/macOS:**
Ollama starts automatically after installation. Check the system tray.

**Linux:**
```bash
# Start Ollama server in background
ollama serve &

# Or start as systemd service
systemctl start ollama
```

### 3. Download Llama 3.2-Vision Model

```bash
# Download 11B model (recommended - ~7GB)
ollama pull llama3.2-vision:11b

# Or download 90B model for best quality (~55GB)
ollama pull llama3.2-vision:90b

# Verify model is downloaded
ollama list
```

**Expected output:**
```
NAME                    ID              SIZE      MODIFIED
llama3.2-vision:11b    abc123def456    7.0 GB    2 minutes ago
```

### 4. Set Up Python Environment

**Windows:**
```bash
# Navigate to project directory
cd "d:\A\LLM\GPT2(M)\PDF-QA-System"

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

**macOS/Linux:**
```bash
cd /path/to/PDF-QA-System

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 5. Install Python Dependencies

```bash
# Install core requirements
pip install -r requirements_vision.txt

# This will install:
# - PyMuPDF (PDF processing)
# - ChromaDB (vector database)
# - Transformers (AI models)
# - ColPali (visual retrieval)
# - Ollama Python client
# - And more...
```

**For GPU Support (Optional but Recommended):**

```bash
# Uninstall CPU version
pip uninstall torch torchvision

# Install CUDA version (NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 6. Install ColPali (Advanced)

ColPali may need special installation:

```bash
# Try standard installation
pip install colpali-engine

# If that fails, install from GitHub
pip install git+https://github.com/illuin-tech/colpali.git

# If both fail, the system will use CLIP as fallback
```

## Model Setup

### Configure Your Models

Run the interactive model selector:

```bash
python model_selector.py
```

**Example Configuration:**

```
========================================================================
                      MODEL CONFIGURATION
========================================================================

SELECT VISION MODEL:
[1] Llama 3.2-Vision 11B (Recommended) - ~7GB, Exceptional quality ✓
[2] Llama 3.2-Vision 90B - ~55GB, Best-in-class
[3] ColPali Visual Retriever Only - ~1GB, Good for retrieval

Enter choice [1-3] (default: 1): 1

✓ Selected: Llama 3.2-Vision 11B

SELECT EMBEDDING MODEL:
[1] MiniLM (Recommended) - ~80MB, Fast ✓
[2] BGE Base - ~420MB, State-of-the-art
...

✓ Configuration complete!
```

## Usage

### Start the Application

**Easy way (Windows):**
```bash
start_vision_app.bat
```

**Manual way:**
```bash
# Make sure Ollama is running
ollama serve

# Start Flask app
python app_vision.py
```

**Open in browser:**
```
http://localhost:5000
```

### Using the System

1. **Upload PDF** (up to 500+ pages)
   - Drag & drop or browse for PDF
   - System will:
     - Convert each page to an image
     - Extract text and embedded images
     - Create visual index with ColPali
     - Store in ChromaDB

2. **Ask Questions**
   - Type your question
   - System will:
     - Find relevant pages visually using ColPali
     - Send page images to Llama 3.2-Vision
     - Understand diagrams, charts, tables
     - Generate comprehensive answer

### Example Questions

**For Technical Documents:**
```
"What does the architecture diagram on page 15 show?"
"Explain the flowchart in section 3"
"What are the components in the system diagram?"
```

**For Financial Documents:**
```
"What is the total revenue shown in the graph?"
"Summarize the chart on Q4 performance"
"What trends are shown in the financial data visualization?"
```

**For Academic Papers:**
```
"Explain the results shown in Figure 2"
"What does the comparison table in section 4 indicate?"
"Describe the experimental setup diagram"
```

## Performance Optimization

### For Large PDFs (500+ pages)

1. **Adjust Processing DPI:**
```python
# In vision_pdf_processor.py
processor = VisionPDFProcessor(
    dpi=100,  # Lower DPI for faster processing (default: 150)
    batch_size=20  # Process more pages at once
)
```

2. **Use GPU Acceleration:**
```bash
# Install CUDA PyTorch (if you have NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. **Increase Ollama Context:**
```bash
# Edit Ollama modelfile for larger context
ollama show llama3.2-vision:11b --modelfile > Modelfile
# Edit Modelfile: PARAMETER num_ctx 8192
ollama create llama3.2-vision:11b-extended -f Modelfile
```

### Memory Management

**For systems with limited RAM:**

```python
# Process PDF in chunks
CHUNK_SIZE = 50  # Process 50 pages at a time

# Use lower DPI
DPI = 100  # Instead of 150

# Disable ColPali if needed
USE_COLPALI = False  # Falls back to text-based retrieval
```

## Troubleshooting

### Ollama Issues

**Problem: "Connection refused to Ollama"**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Check Ollama logs (Linux/macOS)
journalctl -u ollama -f
```

**Problem: "Model not found"**
```bash
# List available models
ollama list

# Pull the model
ollama pull llama3.2-vision:11b
```

### ColPali Issues

**Problem: "Failed to load ColPali"**
```bash
# System will automatically fall back to CLIP
# To force ColPali installation:
pip install git+https://github.com/illuin-tech/colpali.git

# Or disable ColPali in config
USE_COLPALI = False
```

### Memory Issues

**Problem: "Out of memory" when processing large PDFs**
```python
# Reduce batch size
processor = VisionPDFProcessor(batch_size=5)

# Reduce DPI
processor = VisionPDFProcessor(dpi=100)

# Process in smaller chunks
# Split 500-page PDF into 10x50 page chunks
```

### GPU Issues

**Problem: "CUDA out of memory"**
```bash
# Use CPU instead
export CUDA_VISIBLE_DEVICES=""

# Or reduce batch size
BATCH_SIZE = 1
```

## Advanced Configuration

### Custom Ollama Model

```bash
# Create custom modelfile
cat > Modelfile << EOF
FROM llama3.2-vision:11b
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER num_predict 1000
EOF

# Create custom model
ollama create my-vision-model -f Modelfile

# Use in config
MODEL_NAME = "my-vision-model"
```

### Multiple GPUs

```python
# In vision_qa_engine.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use GPU 0 and 1
```

## Comparison: Traditional vs Vision System

| Feature | Traditional System | Vision System |
|---------|-------------------|---------------|
| Text extraction | ✅ Good | ✅ Excellent |
| Image understanding | ❌ No | ✅ Yes |
| Diagram analysis | ❌ No | ✅ Yes |
| Chart interpretation | ❌ No | ✅ Yes |
| Table extraction | ⚠️ Limited | ✅ Excellent |
| Layout awareness | ❌ No | ✅ Yes |
| Speed (text-only) | ⚡ Very fast | 🐢 Slower |
| Speed (with images) | N/A | ⚡ Fast with ColPali |
| Large PDFs (500+ pages) | ⚠️ Struggles | ✅ Optimized |

## Next Steps

1. ✅ Install Ollama and Llama 3.2-Vision
2. ✅ Install Python dependencies
3. ✅ Configure models with `model_selector.py`
4. ✅ Start application with `python app_vision.py`
5. ✅ Upload your first PDF and ask questions!

## Support

For issues:
1. Check this guide's troubleshooting section
2. Verify Ollama is running: `ollama list`
3. Check logs in `logs/` directory
4. Review ChromaDB data in `chroma_db/` directory

## Resources

- Ollama: https://ollama.ai
- Llama 3.2 Vision: https://ollama.ai/library/llama3.2-vision
- ColPali: https://github.com/illuin-tech/colpali
- ChromaDB: https://docs.trychroma.com

---

**Ready to process your PDFs? Start with:**
```bash
python app_vision.py
```
