# Using Advanced Models for Better Accuracy

The PDF Q&A system now supports **two modes** for answering questions:

## Mode 1: Smart Extractive (DEFAULT - Recommended)
✅ **Currently Active**
✅ Fast and accurate for most questions
✅ Works on low-end hardware (CPU only)
✅ Uses regex patterns to find specific info (amounts, dates, names)
✅ Returns actual text from your PDF

**Perfect for:**
- Invoice amounts, totals, prices
- Dates and deadlines
- Company names and addresses
- Any factual information extraction

## Mode 2: Advanced QA with DistilBERT (Optional)
⚡ More powerful but slower
⚡ Requires more memory (~2GB)
⚡ Better at understanding complex questions
⚡ Provides confidence scores

**Best for:**
- Complex reasoning questions
- Questions requiring context understanding
- When you need confidence scores
- Multi-hop reasoning

---

## How to Enable Advanced QA Model

### Step 1: Edit `config.py`

Open [config.py](config.py) and change:

```python
QA_CONFIG = {
    'mode': 'extractive',
    'use_advanced_qa': True,  # ← Change this from False to True
    'advanced_qa_model': 'distilbert-base-cased-distilled-squad',
    'top_k_chunks': 5,
    'max_answer_length': 800,
}
```

### Step 2: Restart the Server

Stop the server (Ctrl+C) and restart:
```bash
python app.py
```

### Step 3: Wait for Model Download

The first time you enable this, it will download the DistilBERT model (~250MB). This happens automatically.

---

## Available Advanced Models

You can choose different models in `config.py`:

### 1. DistilBERT (Default - Recommended)
```python
'advanced_qa_model': 'distilbert-base-cased-distilled-squad'
```
- **Size:** ~250MB
- **Speed:** Fast
- **Accuracy:** Good
- **Best for:** General purpose QA

### 2. RoBERTa (More Accurate)
```python
'advanced_qa_model': 'deepset/roberta-base-squad2'
```
- **Size:** ~500MB
- **Speed:** Medium
- **Accuracy:** Better
- **Best for:** Complex questions

### 3. BERT Large (Most Accurate)
```python
'advanced_qa_model': 'bert-large-uncased-whole-word-masking-finetuned-squad'
```
- **Size:** ~1.3GB
- **Speed:** Slower
- **Accuracy:** Best
- **Best for:** When accuracy is critical
- **Requires:** 4GB+ RAM

---

## Current Smart Extractive Features

Even without the advanced model, the system now has **smart extraction** for:

### ✅ Monetary Amounts
Questions like:
- "What is the total amount?"
- "How much is the bill?"
- "What is the price?"

**Detects:** ₹1,234.56, Rs. 1000, $99.99, etc.
**Returns:** The largest amount found (usually the total)

### ✅ Dates
Questions like:
- "What is the date?"
- "When was this issued?"
- "What is the deadline?"

**Detects:** 12/31/2023, 2023-12-31, 31 Dec 2023, etc.

### ✅ Names & Companies
Questions like:
- "What is the company name?"
- "Who is the vendor?"
- "What is the seller name?"

**Detects:** Capitalized names, companies with Pvt. Ltd., Inc., etc.

---

## Performance Comparison

| Feature | Smart Extractive | Advanced QA |
|---------|-----------------|-------------|
| **Speed** | Very Fast (<1s) | Medium (2-5s) |
| **Memory** | Low (~500MB) | High (2-4GB) |
| **Accuracy (Facts)** | Excellent | Good |
| **Accuracy (Reasoning)** | Good | Excellent |
| **Works Offline** | ✅ Yes | ✅ Yes |
| **GPU Support** | Not needed | Recommended |

---

## Troubleshooting

### Advanced model fails to load
- **Issue:** Not enough memory
- **Solution:** Stick with Smart Extractive mode or use a smaller model

### Slow performance
- **Issue:** Running on CPU
- **Solution:** Install PyTorch with CUDA support for GPU acceleration

### Wrong answers still
- **Issue:** PDF text extraction quality
- **Solution:** Ensure your PDF has selectable text (not scanned images)

---

## Recommendation

**For your use case (invoices/bills):**
- ✅ **Keep Smart Extractive mode (default)** - it's perfect for amounts, dates, and names
- The regex-based extraction is actually MORE accurate for structured documents like invoices
- Advanced QA is better for unstructured text like articles or reports

**Try Advanced QA if:**
- You're querying research papers, books, or articles
- Questions require understanding context and relationships
- You need the system to "reason" about the content

---

## Example Outputs

### Smart Extractive (Current):
**Question:** "What is the total amount?"
**Answer:**
```
Total Amount: ₹27,085.22

Context: Grand Total ₹27,085.22 inclusive of all taxes
```

### Advanced QA:
**Question:** "What is the total amount?"
**Answer:**
```
₹27,085.22

[Confidence: High (0.92)]
```

Both work well, but Smart Extractive gives more context!

---

**Need help?** Check the main [README.md](README.md) for general setup instructions.
