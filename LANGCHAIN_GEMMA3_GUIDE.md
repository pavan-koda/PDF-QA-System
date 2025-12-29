# LangChain RAG & Gemma 3 Integration Guide

## 🎉 New Features Added

### ✅ 1. Gemma 3 Models (4 variants)
Google's latest third-generation Gemma models are now available!

### ✅ 2. LangChain RAG Pipeline
Advanced RAG (Retrieval-Augmented Generation) using LangChain framework!

---

## 🌟 Gemma 3 Models

### What is Gemma 3?

**Gemma 3** is Google's third-generation open model family (released late 2024), offering:
- **Improved architecture** over Gemma 2
- **Better instruction-following**
- **More efficient inference**
- **Enhanced reasoning capabilities**

### Available Gemma 3 Models

#### 1. **Gemma 3 4B**
```
Model ID: google/gemma-3-4b
Size: ~8GB
Speed: Fast
Quality: Excellent
Best For: Compact deployment, fast inference
Requirements: 10GB RAM or 8GB VRAM
```

#### 2. **Gemma 3 12B**
```
Model ID: google/gemma-3-12b
Size: ~24GB
Speed: Medium
Quality: Exceptional
Best For: High-quality responses
Requirements: 24GB+ VRAM (GPU required)
```

#### 3. **Gemma 3 4B Instruct** ⭐ **RECOMMENDED**
```
Model ID: google/gemma-3-4b-it
Size: ~8GB
Speed: Fast
Quality: Exceptional
Best For: Question-answering, instruction tasks
Requirements: 10GB RAM or 8GB VRAM
Highlights: Best-in-class for QA at this size!
```

#### 4. **Gemma 3 12B Instruct**
```
Model ID: google/gemma-3-12b-it
Size: ~24GB
Speed: Medium
Quality: Exceptional
Best For: Premium quality, production use
Requirements: 24GB+ VRAM (GPU required)
```

---

## 📊 Complete Gemma Family Overview

| Generation | Models | Sizes | Status |
|------------|--------|-------|--------|
| **Gemma 1** | 4 | 2B, 7B (base + instruct) | ✅ Available |
| **Gemma 2** | 5 | 2B, 9B, 27B (base + instruct) | ✅ Available |
| **Gemma 3** | 4 | 4B, 12B (base + instruct) | ✅ NEW |
| **TOTAL** | **13 models** | 2B → 27B | All ready! |

---

## 🔗 LangChain RAG Integration

### What is LangChain RAG?

**LangChain** is a framework for building applications with LLMs. The RAG (Retrieval-Augmented Generation) pipeline:
- **Retrieves** relevant document chunks
- **Augments** the query with context
- **Generates** answers using an LLM

### Benefits Over Custom FAISS:

✅ **Better chunking** - RecursiveCharacterTextSplitter
✅ **Advanced retrieval** - Multiple search algorithms
✅ **Prompt templates** - Customizable prompts
✅ **Chain composition** - Combine multiple steps
✅ **Source tracking** - Know which chunks were used
✅ **Framework ecosystem** - Integrates with many tools

### Architecture

```
PDF Text
   ↓
RecursiveCharacterTextSplitter (LangChain)
   ↓
Text Chunks
   ↓
HuggingFaceEmbeddings
   ↓
FAISS Vector Store (LangChain)
   ↓
Question → Retriever → Relevant Chunks
   ↓
Prompt Template + LLM (optional)
   ↓
Answer
```

---

## 🚀 How to Use

### Option 1: Enable LangChain RAG

1. **Install LangChain:**
```bash
pip install langchain langchain-community tiktoken
```

2. **Edit config.py:**
```python
QA_CONFIG = {
    'use_langchain': True,  # Change to True
    ...
}
```

3. **Restart app:**
```bash
python app.py
```

Now the system uses LangChain RAG instead of custom FAISS!

### Option 2: Use Gemma 3 Models

```bash
# Run model selector
python model_selector.py

# Navigate to generator selection
# Choose one of:
[25] Gemma 3 4B
[26] Gemma 3 12B
[27] Gemma 3 4B Instruct ← RECOMMENDED
[28] Gemma 3 12B Instruct

# Select 27 for best balance
Enter choice: 27

# Confirm download
✓ Selected: Gemma 3 4B Instruct
```

---

## 📝 Usage Examples

### Example 1: LangChain RAG with Extractive Mode

**Config:**
```python
QA_CONFIG = {
    'use_langchain': True,
    'use_advanced_qa': False,
}
EMBEDDING_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2'
}
GENERATOR_CONFIG = {
    'model_name': 'none'
}
```

**Features:**
- Uses LangChain for better chunking
- Extractive answers (exact text from PDF)
- Fast and accurate

### Example 2: LangChain RAG with Gemma 3

**Config:**
```python
QA_CONFIG = {
    'use_langchain': True,
}
EMBEDDING_CONFIG = {
    'model_name': 'BAAI/bge-base-en-v1.5'
}
GENERATOR_CONFIG = {
    'model_name': 'google/gemma-3-4b-it',
    'use_generator': True
}
```

**Features:**
- Advanced RAG pipeline
- Gemma 3 for generation
- Best quality answers

### Example 3: Premium Setup

**Config:**
```python
QA_CONFIG = {
    'use_langchain': True,
    'use_advanced_qa': True,
    'advanced_qa_model': 'deepset/roberta-base-squad2'
}
EMBEDDING_CONFIG = {
    'model_name': 'BAAI/bge-base-en-v1.5'
}
GENERATOR_CONFIG = {
    'model_name': 'google/gemma-3-12b-it',
    'use_generator': True
}
```

**Features:**
- LangChain RAG
- RoBERTa for QA
- Gemma 3 12B for generation
- Top-tier quality

---

## 🎯 Recommended Setups

### 1. Fast & Accurate (Recommended)
```
✅ LangChain: Enabled
✅ Embedding: MiniLM (80MB)
✅ QA: Extractive
✅ Generator: None
━━━━━━━━━━━━━━━━━━━━━━━━
Total: 80MB
Speed: ⚡⚡⚡ Very Fast
Quality: ⭐⭐⭐⭐ Excellent
```

### 2. Balanced with Gemma 3
```
✅ LangChain: Enabled
✅ Embedding: BGE Base (420MB)
✅ QA: Extractive
✅ Generator: Gemma 3 4B Instruct (8GB)
━━━━━━━━━━━━━━━━━━━━━━━━
Total: 8.4GB
Speed: ⚡⚡ Fast
Quality: ⭐⭐⭐⭐⭐ Exceptional
```

### 3. Premium Quality
```
✅ LangChain: Enabled
✅ Embedding: BGE Base (420MB)
✅ QA: BERT Large (1.3GB)
✅ Generator: Gemma 3 12B Instruct (24GB)
━━━━━━━━━━━━━━━━━━━━━━━━
Total: 25.7GB
Speed: ⚡ Medium
Quality: ⭐⭐⭐⭐⭐ Exceptional
Requirements: 24GB+ VRAM
```

---

## 📈 Performance Comparison

### Chunking Quality

| Method | Chunk Quality | Semantic Preservation |
|--------|---------------|----------------------|
| Custom Split | Good | ⭐⭐⭐ |
| LangChain | Excellent | ⭐⭐⭐⭐⭐ |

**Winner: LangChain** (Better chunk boundaries)

### Generation Quality (for QA)

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| GPT-2 Medium | 1.5GB | Medium | ⭐⭐⭐ | General |
| FLAN-T5 Base | 900MB | Fast | ⭐⭐⭐⭐ | QA |
| Gemma 2 2B-it | 5GB | Fast | ⭐⭐⭐⭐ | QA |
| **Gemma 3 4B-it** | **8GB** | **Fast** | **⭐⭐⭐⭐⭐** | **QA** |
| Gemma 3 12B-it | 24GB | Medium | ⭐⭐⭐⭐⭐ | Premium |

**Winner: Gemma 3 4B Instruct** (Best balance)

---

## 🔧 Advanced Features

### LangChain RAG Features

1. **Better Text Splitting:**
   - Preserves sentence boundaries
   - Handles markdown/code better
   - Configurable separators

2. **Advanced Retrieval:**
   - Similarity search
   - MMR (Maximum Marginal Relevance)
   - Custom scoring

3. **Prompt Engineering:**
   - Custom prompt templates
   - Few-shot examples
   - Chain of thought

4. **Source Attribution:**
   - Track which chunks were used
   - Show confidence scores
   - Return metadata

### Using LangChain Directly

```python
from langchain_rag import LangChainRAG

# Initialize
rag = LangChainRAG(
    embedder_model="BAAI/bge-base-en-v1.5",
    llm_model="google/gemma-3-4b-it",
    chunk_size=500,
    chunk_overlap=50
)

# Create vector store
rag.create_vector_store(pdf_text, "session123")

# Ask question (extractive)
answer = rag.answer_question(
    "What is the total amount?",
    "session123",
    use_llm=False
)

# Ask question (generative with Gemma 3)
answer = rag.answer_question(
    "Summarize the key points",
    "session123",
    use_llm=True
)
```

---

## 🆚 LangChain vs Custom FAISS

| Feature | Custom FAISS | LangChain RAG |
|---------|-------------|---------------|
| Chunking | Word-based | Recursive char-based ✅ |
| Retrieval | Basic similarity | Advanced algorithms ✅ |
| Prompts | Hardcoded | Customizable ✅ |
| Framework | Standalone | Ecosystem ✅ |
| Speed | Very Fast | Fast |
| Complexity | Simple | Moderate |
| Features | Basic | Advanced ✅ |

**Recommendation:**
- **Custom FAISS:** For speed and simplicity
- **LangChain RAG:** For advanced features and flexibility

---

## 📦 Total Model Count

### All Generators (29 models):

**Lightweight (0-1GB):**
1. None
2. FLAN-T5 Small (300MB)
3. GPT-2 Small (500MB)

**Medium (1-3GB):**
4-10. [7 models]

**Large (3-8GB):**
11-17. [7 models]

**Extra Large (10GB+):**
18-25. [8 models]

**Gemma 3 Family (NEW):**
26. Gemma 3 4B (8GB)
27. Gemma 3 12B (24GB)
28. **Gemma 3 4B Instruct** (8GB) ⭐ **BEST FOR QA**
29. Gemma 3 12B Instruct (24GB)

### Complete System:
- **Embeddings:** 7 models
- **QA Models:** 5 models
- **Generators:** 29 models
- **TOTAL:** **41 models** 🎉

---

## 🚀 Quick Start

### Step 1: Install LangChain (Optional)
```bash
pip install langchain langchain-community tiktoken
```

### Step 2: Enable LangChain
Edit `config.py`:
```python
QA_CONFIG = {
    'use_langchain': True,
}
```

### Step 3: Select Gemma 3
```bash
python model_selector.py
# Choose: Gemma 3 4B Instruct (option 28)
```

### Step 4: Start App
```bash
python app.py
```

### Step 5: Upload & Ask!
```
http://localhost:5000
→ Upload PDF
→ Ask questions
→ Get Gemma 3 powered answers!
```

---

## 💡 Tips

1. **For best QA:** Use Gemma 3 4B Instruct
2. **For speed:** Keep LangChain disabled
3. **For features:** Enable LangChain
4. **For quality:** Gemma 3 + LangChain + BGE embeddings

---

## 🎯 Summary

✅ **4 new Gemma 3 models** added
✅ **LangChain RAG** integration complete
✅ **29 generator models** available
✅ **41 total models** in system
✅ **Best-in-class QA** with Gemma 3 4B Instruct

**Everything ready to use!** 🚀
