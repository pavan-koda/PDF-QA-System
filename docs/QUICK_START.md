# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Restart the Server

**Stop the current server:**
- Go to your command prompt window
- Press `Ctrl + C`

**Start the new version:**
```bash
start_app.bat
```

**Wait for this message:**
```
INFO:qa_engine:Sentence transformer loaded successfully
INFO:qa_engine:GPT-2 model loaded successfully
* Running on http://127.0.0.1:5000
```

### Step 2: Open in Browser

Navigate to: **http://localhost:5000** or **http://127.0.0.1:5000**

### Step 3: Upload and Ask

1. **Upload your PDF** (drag & drop or browse)
2. **Wait for processing**
3. **Ask questions!**

---

## ✨ Example Questions to Try

### For Invoices/Bills:

```
✅ "What is the total amount?"
✅ "What is the bill date?"
✅ "Who is the vendor?"
✅ "What is the GST number?"
✅ "What items are listed?"
```

### For General Documents:

```
✅ "What is this about?"
✅ "Summarize the main points"
✅ "What are the key dates?"
✅ "Who are the parties involved?"
```

---

## 🎯 What Changed

| Before | After |
|--------|-------|
| Random gibberish | ✅ Actual text from PDF |
| "J2Z4Billing..." | ✅ "Total Amount: ₹27,085.22" |
| No amounts found | ✅ Extracts all monetary values |
| No dates found | ✅ Detects date formats |
| No names found | ✅ Identifies companies/people |

---

## ⚡ Current Settings

**Mode:** Smart Extractive (Fast & Accurate)
- ✅ Best for invoices, bills, receipts
- ✅ Extracts amounts, dates, names
- ✅ Returns actual PDF text
- ✅ Works on low-end hardware

**Want More Power?**
See [ADVANCED_MODELS.md](ADVANCED_MODELS.md) to enable DistilBERT

---

## 🔧 Troubleshooting

### "Connection Refused" Error
- Make sure the server is running
- Check for "Running on http://127.0.0.1:5000" message
- Try http://127.0.0.1:5000 instead of localhost

### Still Getting Wrong Answers
- Make sure you **restarted** the server
- Try more specific questions ("total amount" not just "amount")
- Check if PDF has selectable text (not scanned image)

### Server Won't Start
- Make sure you're in the correct directory
- Check if virtual environment is activated
- Run: `pip install -r requirements.txt`

---

## 📚 Documentation

- **[README.md](README.md)** - Full documentation
- **[ADVANCED_MODELS.md](ADVANCED_MODELS.md)** - Advanced features
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** - What's new
- **[config.py](config.py)** - Settings to customize

---

## 🆘 Need Help?

1. Check error messages in the terminal
2. Review the documentation files above
3. Ensure PDF has selectable text (not images)

---

**Ready? Restart your server and try it now!** 🎉
