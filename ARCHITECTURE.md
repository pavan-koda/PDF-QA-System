# 🏗️ System Architecture - Vision PDF QA

Detailed technical architecture of the vision-based PDF QA system.

## 📊 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│                     (Flask Web App)                              │
└────────────┬────────────────────────────────┬────────────────────┘
             │                                 │
             │ Upload PDF                      │ Ask Question
             ↓                                 ↓
┌────────────────────────────┐    ┌──────────────────────────────┐
│   VISION PDF PROCESSOR     │    │    VISION QA ENGINE          │
│   (vision_pdf_processor)   │    │   (vision_qa_engine)         │
└────────────┬───────────────┘    └───────────┬──────────────────┘
             │                                 │
             │ Process PDF                     │ Retrieve + Answer
             ↓                                 ↓
┌────────────────────────────┐    ┌──────────────────────────────┐
│   PyMuPDF (fitz)          │    │   COLPALI RETRIEVER          │
│   - Render pages → images  │    │   (colpali_retriever)        │
│   - Extract text           │    │   - Visual similarity search │
│   - Extract embedded imgs  │    │   - FAISS index              │
└────────────┬───────────────┘    └───────────┬──────────────────┘
             │                                 │
             │ Store                           │ Query
             ↓                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│                         CHROMADB                                  │
│              (Multimodal Vector Database)                         │
│   - Page images metadata                                          │
│   - Text embeddings                                               │
│   - Visual embeddings                                             │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                │ Page images + context
                                ↓
┌──────────────────────────────────────────────────────────────────┐
│                    LLAMA 3.2-VISION (via Ollama)                  │
│              - Multimodal understanding                           │
│              - Image + Text analysis                              │
│              - Answer generation                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 🔧 Component Details

### 1. Vision PDF Processor (`vision_pdf_processor.py`)

**Purpose:** Convert PDFs to processable format for vision models

**Key Features:**
- Renders PDF pages to images (150 DPI default)
- Extracts text using PyMuPDF's `get_text()`
- Extracts embedded images separately
- Processes in batches for memory efficiency

**Technology:**
- **PyMuPDF (fitz):** Superior PDF rendering and extraction
- **Pillow:** Image format conversion
- **Base64:** Encoding for API transmission

**Process Flow:**
```python
PDF File
    ↓
Open with PyMuPDF
    ↓
For each page:
    ├─→ Render to PNG (150 DPI)
    ├─→ Extract text content
    └─→ Extract embedded images
    ↓
Save to session directory:
    ├─→ page_0001.png, page_0002.png, ...
    ├─→ embedded_images/page_0001_img_001.png, ...
    └─→ Metadata (pages, text, images)
```

**Performance:**
- **Speed:** 2-5 seconds per page
- **Memory:** ~50MB per page at 150 DPI
- **Optimization:** Batch processing reduces overhead

### 2. ColPali Retriever (`colpali_retriever.py`)

**Purpose:** Visual similarity search across document pages

**How It Works:**
1. **Indexing:**
   - Encodes each page image into vector embedding
   - Creates FAISS index for fast similarity search
   - Stores embeddings + metadata

2. **Retrieval:**
   - Encodes user query to vector
   - Searches FAISS index for similar pages
   - Returns top-k most relevant pages

**Technology:**
- **ColPali:** Vision-language retrieval model
- **CLIP (fallback):** OpenAI's vision-text model
- **FAISS:** Facebook's similarity search library

**Key Advantages:**
- ✅ Understands visual layout (diagrams, charts)
- ✅ Finds pages even without text
- ✅ Fast: Searches 500 pages in <1 second
- ✅ Works with unstructured documents

**Architecture:**
```
Page Images → ColPali Encoder → Visual Embeddings → FAISS Index

Query Text → ColPali Encoder → Query Embedding → Search FAISS
                                                      ↓
                                              Top-k Page Indices
```

### 3. Vision QA Engine (`vision_qa_engine.py`)

**Purpose:** Answer questions using multimodal understanding

**Components:**

#### A. ChromaDB Integration
```python
Collection per session:
├─→ Documents: Page text content
├─→ Metadatas: {page, image_path, has_text, ...}
├─→ Embeddings: Text embeddings (auto-generated)
└─→ IDs: page_1, page_2, ..., page_N
```

#### B. Ollama Client
```python
Query Process:
1. Receive question
2. Retrieve relevant pages (ColPali or ChromaDB)
3. Load page image
4. Encode image to base64
5. Send to Ollama API:
   {
     "model": "llama3.2-vision:11b",
     "prompt": question + context,
     "images": [base64_image]
   }
6. Parse response
7. Return answer
```

#### C. Multi-Strategy Retrieval
```python
Strategy 1: ColPali Visual Search (preferred)
    - Best for diagrams, charts, visual content
    - Fast and accurate

Strategy 2: ChromaDB Text Search (fallback)
    - Good for text-heavy documents
    - Used when ColPali unavailable
```

**Answer Generation Flow:**
```
User Question
    ↓
Retrieve Pages (ColPali or ChromaDB)
    ↓
Get top 5 relevant pages
    ↓
For best page:
    ├─→ Load page image
    ├─→ Get text context from top 3 pages
    └─→ Send to Llama 3.2-Vision
    ↓
Llama analyzes:
    ├─→ Visual content (diagrams, charts, images)
    ├─→ Text content (words, paragraphs)
    └─→ Layout (tables, structure)
    ↓
Generate comprehensive answer
    ↓
Return to user (5-15 seconds total)
```

### 4. Llama 3.2-Vision (via Ollama)

**Purpose:** Multimodal AI for vision + text understanding

**Model Specs:**
- **Parameters:** 11B (recommended) or 90B (best quality)
- **Context Window:** 8,192 tokens
- **Vision Capabilities:**
  - Object detection
  - OCR (text in images)
  - Chart/diagram understanding
  - Scene description
  - Visual reasoning

**Ollama Integration:**
```bash
# Local inference server
ollama serve → localhost:11434

# API endpoint
POST /api/generate
{
  "model": "llama3.2-vision:11b",
  "prompt": "Explain this diagram",
  "images": ["base64_encoded_image"],
  "options": {
    "num_predict": 800,
    "temperature": 0.7
  }
}

# Response
{
  "response": "The diagram shows...",
  "total_duration": 12345678900
}
```

**Advantages of Ollama:**
- ✅ 100% local (no cloud)
- ✅ No API keys needed
- ✅ Fast inference (GPU optional)
- ✅ Easy model management
- ✅ Cross-platform (Windows, macOS, Linux)

## 💾 Data Flow

### Upload Process

```
1. User uploads PDF
   └─→ Save to uploads/{session_id}_{filename}.pdf

2. Vision PDF Processor
   ├─→ Open PDF with PyMuPDF
   ├─→ Process each page:
   │   ├─→ Render to PNG (150 DPI)
   │   ├─→ Extract text
   │   └─→ Extract images
   └─→ Save to processed_pdfs/{session_id}/
       ├─→ page_0001.png
       ├─→ page_0002.png
       └─→ embedded_images/

3. ColPali Indexing
   ├─→ Encode all page images
   ├─→ Create FAISS index
   └─→ Save to data/{session_id}_colpali.faiss

4. ChromaDB Storage
   ├─→ Create collection pdf_{session_id}
   ├─→ Add documents (page texts)
   ├─→ Add metadatas (image paths, page numbers)
   └─→ Auto-generate embeddings

5. Ready for queries! ✅
```

### Query Process

```
1. User asks question
   └─→ POST /ask {question: "..."}

2. Retrieval Phase
   ├─→ ColPali search (visual)
   │   ├─→ Encode query
   │   ├─→ Search FAISS index
   │   └─→ Get top-k pages
   │
   └─→ ChromaDB search (text fallback)
       ├─→ Embed query
       ├─→ Search collection
       └─→ Get top-k documents

3. Context Building
   ├─→ Best page image
   ├─→ Text from top 3 pages
   └─→ Combine into context

4. Vision AI Analysis
   ├─→ Encode page image to base64
   ├─→ Build prompt (question + context)
   ├─→ Send to Ollama
   └─→ Receive answer

5. Response
   └─→ Return {answer, response_time, page_used}
```

## 🗄️ Storage Structure

```
PDF-QA-System/
├── uploads/
│   └── {session_id}_{filename}.pdf
│
├── processed_pdfs/
│   └── {session_id}/
│       ├── page_0001.png
│       ├── page_0002.png
│       ├── page_0003.png
│       └── embedded_images/
│           ├── page_0001_img_001.png
│           └── page_0002_img_001.png
│
├── data/
│   ├── {session_id}_colpali.faiss       # FAISS index
│   └── {session_id}_colpali_meta.pkl    # Metadata
│
├── chroma_db/
│   └── {chroma_internal_structure}      # ChromaDB data
│
└── logs/
    └── vision_performance.txt           # Query logs
```

## ⚙️ Configuration Options

### PDF Processing

```python
# vision_pdf_processor.py
VisionPDFProcessor(
    dpi=150,              # Image quality (72-300)
    extract_images=True,  # Extract embedded images
    extract_text=True,    # Extract text content
    batch_size=10         # Pages per batch
)
```

### ColPali Retrieval

```python
# colpali_retriever.py
ColPaliRetriever(
    model_name="vidore/colpali",  # Model to use
    device="cuda",                # GPU/CPU
    use_half_precision=True       # FP16 for speed
)
```

### Vision QA

```python
# vision_qa_engine.py
VisionQAEngine(
    ollama_url="http://localhost:11434",
    model_name="llama3.2-vision:11b",
    chroma_persist_dir="chroma_db",
    use_colpali=True
)
```

## 🔐 Security Considerations

### Input Validation
- PDF file type checking
- File size limits (500MB)
- Question length limits (1000 chars)
- Session ID validation

### Session Isolation
- Unique session ID per upload
- Separate storage directories
- Collection namespacing in ChromaDB

### Data Cleanup
- Automatic cleanup on session reset
- Temporary file removal
- Database collection deletion

### Production Deployment
- Use Gunicorn instead of Flask dev server
- Add authentication (Flask-Login)
- Enable HTTPS (nginx reverse proxy)
- Rate limiting
- Input sanitization

## 📈 Performance Optimization

### CPU-Bound Operations
- PDF rendering (PyMuPDF)
- Image encoding (Pillow)
- FAISS indexing

**Optimization:**
- Batch processing
- Lower DPI for faster rendering
- Lazy loading

### GPU-Bound Operations
- ColPali encoding
- Llama 3.2-Vision inference

**Optimization:**
- Use CUDA PyTorch
- FP16 precision
- Batch inference

### Memory Management
- Process PDFs in chunks
- Release unused page data
- Clear embeddings after indexing

### Caching Strategy
- Cache embeddings (FAISS + ChromaDB)
- Cache page images on disk
- No caching of answers (always fresh)

## 🔮 Future Enhancements

1. **OCR Integration**
   - Add Tesseract for scanned PDFs
   - Combine OCR + native text

2. **Multi-PDF Support**
   - Search across multiple documents
   - Cross-document queries

3. **Advanced Retrieval**
   - Hybrid search (dense + sparse)
   - Re-ranking models
   - Query expansion

4. **Better Vision Models**
   - LLaVA integration
   - GPT-4V API option
   - CogVLM support

5. **Export Features**
   - Save Q&A to PDF
   - Export highlights
   - Bookmark important pages

---

**Architecture designed for:**
- ✅ Scalability (500+ page PDFs)
- ✅ Accuracy (vision + text understanding)
- ✅ Speed (ColPali fast retrieval)
- ✅ Privacy (100% local processing)
- ✅ Flexibility (modular components)
