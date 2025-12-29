# PDF Question & Answer System

A robust web-based application that allows you to upload PDF documents and ask questions about their content using AI. The system uses FAISS for semantic search and intelligent extraction for accurate answers.

## Features

- **Web-based Interface**: Upload PDFs and ask questions through a clean, modern web interface
- **Smart Extractive QA**: Intelligently extracts specific information like amounts, dates, and names
- **Semantic Search**: Uses sentence transformers to find relevant content in your PDF
- **Multiple QA Modes**: Choose between fast extractive or advanced AI-powered answering
- **Session Management**: Multiple users can use the system simultaneously with isolated sessions
- **Robust Error Handling**: Comprehensive error handling and validation
- **Drag & Drop Upload**: Easy PDF upload with drag-and-drop support
- **Real-time Chat Interface**: Ask multiple questions in a conversational format

### New Smart Features

✨ **Automatic Information Extraction**:
- **Amounts & Prices**: Automatically finds and extracts monetary values (₹, $, Rs.)
- **Dates**: Detects various date formats (DD/MM/YYYY, YYYY-MM-DD, etc.)
- **Names & Companies**: Identifies capitalized names and company entities
- **Context-Aware**: Shows relevant context around extracted information

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **AI Models** (Choose from multiple options):
  - **Embeddings**: MiniLM, MPNet, BGE, DistilRoBERTa, and more
  - **QA Models**: BERT, RoBERTa, DistilBERT, ELECTRA
  - **Generators**: GPT-2 (all sizes), FLAN-T5, OPT, Phi-2, TinyLlama, Gemma 1 & 2, Llama 2, StableLM
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **PDF Processing**: pypdf

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- 4GB+ RAM recommended
- GPU optional (will use CPU if not available)

### Step 1: Clone or Download the Repository

```bash
cd GPT2(M)
```

### Step 2: Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including Flask, PyTorch, Transformers, FAISS, and more.

### Step 4: Start the Application (Automatic Setup)

**Easy way (Windows):**
```bash
start_app.bat
```

**Manual way:**
```bash
python app.py
```

### First-Time Setup - Model Selector

On first run, you'll see an **interactive model selector** with many options:

```
========================================================================
                      MODEL CONFIGURATION
========================================================================

SELECT EMBEDDING MODEL:
[1] MiniLM (Recommended) - ~80MB, Fast
[2] MPNet - ~420MB, Better quality
[3] DistilRoBERTa - ~290MB, Balanced
[4] Multilingual MiniLM - ~420MB, 50+ languages
[5] Multi-QA MPNet - ~420MB, Optimized for QA
[6] BGE Small - ~130MB, High performance
[7] BGE Base - ~420MB, State-of-the-art

SELECT QUESTION ANSWERING MODEL:
[1] Smart Extractive (Recommended) - 0MB, Best for invoices
[2] DistilBERT - ~250MB, AI-powered
[3] RoBERTa - ~500MB, Most accurate
[4] BERT Large - ~1.3GB, High accuracy
[5] ELECTRA Base - ~400MB, Optimized QA

SELECT TEXT GENERATOR (Optional):
[1] None (Recommended) - 0MB, Most accurate
[2] GPT-2 Small - ~500MB, 117M params
[3] GPT-2 Medium - ~1.5GB, 345M params
[4] GPT-2 Large - ~3GB, 774M params
[5] GPT-2 XL - ~6GB, 1.5B params (requires GPU)
[6] OPT-350M - ~700MB, Meta's efficient model
[7] OPT-1.3B - ~2.5GB, Better generation
[8] FLAN-T5 Small - ~300MB, Good for QA
[9] FLAN-T5 Base - ~900MB, Instruction-tuned
[10] FLAN-T5 Large - ~3GB, Best for QA tasks
[11] Phi-2 - ~5.5GB, Microsoft's 2.7B model
[12] TinyLlama Chat - ~2.2GB, Llama-based 1.1B
[13] StableLM Zephyr - ~3GB, Stability AI 1.6B
[14] Gemma 2B - ~5GB, Google's lightweight model
[15] Gemma 7B - ~14GB, Full Gemma (requires GPU)
[16] Gemma 2B Instruct - ~5GB, Best for QA
[17] Gemma 7B Instruct - ~14GB, Premium quality
[18] Gemma 2 2B - ~5GB, Improved 2nd generation
[19] Gemma 2 9B - ~18GB, State-of-the-art
[20] Gemma 2 2B Instruct - ~5GB, Excellent for QA
[21] Gemma 2 9B Instruct - ~18GB, Best quality
[22] Gemma 2 27B Instruct - ~54GB, Exceptional (requires 48GB VRAM)
[23] Llama 2 7B Chat - ~13GB, Meta's chat model
[24] Llama 2 13B Chat - ~26GB, Larger Llama 2
```

**Recommended for most users (invoices/bills):**
- Embedding: **MiniLM** (80MB download)
- QA Mode: **Smart Extractive** (no download needed)
- Generator: **None** (no download needed)

**Total: ~80MB one-time download, then works OFFLINE!**

**For advanced QA with AI models:**
- Embedding: **Multi-QA MPNet** or **BGE Base**
- QA Mode: **RoBERTa** or **BERT Large**
- Generator: **FLAN-T5 Base** (excellent for question answering)

## Usage

### Starting the Application

**Windows:**
```bash
start_app.bat
```

The script will:
1. Check/create virtual environment
2. Install dependencies if needed
3. Show model selector (first time)
4. Download selected models
5. Start the Flask server

**Linux/Mac:**
```bash
chmod +x start_app.sh
./start_app.sh
```

**Open your browser:**
```
http://localhost:5000
```

### Using the Application

1. **Upload a PDF**:
   - Click "Browse Files" or drag & drop your PDF onto the upload area
   - Maximum file size: 16MB
   - Click "Upload & Process" to process the PDF

2. **Ask Questions**:
   - Once the PDF is processed, the Q&A interface will appear
   - Type your question in the text area
   - Press "Ask Question" or hit Enter to submit
   - The system will extract relevant information from your PDF

   **Example Questions:**
   - "What is the total amount?" - Extracts monetary values
   - "What is the date?" - Finds dates in the document
   - "Who is the vendor?" - Identifies company names
   - "What is this about?" - Returns relevant sections

3. **Upload a New PDF**:
   - Click "Upload New PDF" to reset and upload a different document

## Advanced Features & Model Selection

Want to explore more models or optimize performance? Check out our comprehensive guides:

- **[MODEL_GUIDE.md](MODEL_GUIDE.md)** - Complete guide to all 26+ available models with:
  - Detailed comparisons of all models
  - Performance benchmarks
  - Recommended combinations for different use cases
  - Memory and speed requirements

- **[NEW_MODELS_SUMMARY.md](NEW_MODELS_SUMMARY.md)** - Summary of newly added models:
  - 18 new models including GPT-2 Large/XL, FLAN-T5, Phi-2, TinyLlama
  - BGE state-of-the-art embeddings
  - Multilingual support
  - Migration guide

## Project Structure

```
GPT2(M)/
├── app.py                  # Flask web application (main entry point)
├── pdf_processor.py        # PDF text extraction and chunking
├── qa_engine.py           # Question-answering engine with smart extraction
├── config.py              # Configuration file for models and settings
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── ADVANCED_MODELS.md    # Guide for using advanced QA models
├── start_app.bat         # Windows quick start script
├── start_app.sh          # Linux/Mac quick start script
├── templates/
│   └── index.html        # Web interface HTML
├── static/
│   ├── style.css         # CSS styling
│   └── script.js         # Frontend JavaScript
├── uploads/              # Temporary PDF storage (created on first run)
└── data/                # Session data storage (created on first run)
```

## Configuration

You can modify the following parameters in the respective files:

### PDF Processing ([pdf_processor.py](pdf_processor.py))
- `chunk_size`: Maximum words per chunk (default: 400)
- `chunk_overlap`: Words to overlap between chunks (default: 50)

### QA Engine ([qa_engine.py](qa_engine.py))
- `max_new_tokens`: Maximum length of generated answers (default: 150)
- `temperature`: Randomness in generation (default: 0.7)
- `top_k`: Top-k sampling parameter (default: 50)
- `top_p`: Nucleus sampling parameter (default: 0.92)

### Flask App ([app.py](app.py))
- `MAX_CONTENT_LENGTH`: Maximum file size (default: 16MB)
- `server_port`: Port to run the server (default: 5000)

## API Endpoints

The application exposes the following endpoints:

- `GET /` - Main web interface
- `POST /upload` - Upload and process PDF
- `POST /ask` - Ask a question about the uploaded PDF
- `POST /reset` - Reset the session and clear data
- `GET /health` - Health check endpoint

## Troubleshooting

### Models Not Loading
- Ensure you have a stable internet connection for first-time model download
- Check that you have enough disk space (models are ~1GB total)
- Try manually downloading models as shown in Step 4

### PDF Processing Fails
- Ensure the PDF contains selectable text (not scanned images)
- Try a smaller PDF or increase `MAX_CONTENT_LENGTH`
- Check that the PDF is not password-protected

### Out of Memory Errors
- Reduce `chunk_size` in [pdf_processor.py](pdf_processor.py)
- Reduce `max_new_tokens` in [qa_engine.py](qa_engine.py)
- Close other applications to free up RAM

### Port Already in Use
Change the port in [app.py](app.py):
```python
app.run(host='0.0.0.0', port=5001, debug=True)  # Use different port
```

## Performance Tips

1. **Use GPU**: If you have an NVIDIA GPU, install the GPU version of PyTorch:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Reduce Model Size**: For faster performance on CPU, you can use `gpt2` instead of `gpt2-medium` in [qa_engine.py](qa_engine.py)

3. **Optimize Chunk Size**: Smaller chunks improve retrieval accuracy but may lose context

## Security Considerations

- This is a development server; use a production WSGI server (like Gunicorn) for deployment
- The application stores uploaded PDFs temporarily; ensure proper cleanup
- Consider adding authentication for production use
- Validate and sanitize all user inputs

## License

This project is for educational purposes. Please ensure you have the right to use the PDFs you upload.

## Acknowledgments

- Hugging Face for the Transformers library
- Facebook AI for FAISS
- Sentence Transformers for embedding models

## Support

For issues or questions, please check:
- Ensure all dependencies are installed correctly
- Check Python version compatibility
- Review error logs for specific issues

---

**Note**: The first time you run the application, it may take a few minutes to download the AI models. Subsequent runs will be much faster.
