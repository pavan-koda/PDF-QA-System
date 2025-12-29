# How to Download Gemma 3 Models

Gemma 3 models are **gated models** from Google that require HuggingFace authentication.

## Step-by-Step Setup Guide

### 1. Create HuggingFace Account (if you don't have one)
- Go to: https://huggingface.co/join
- Sign up for a free account

### 2. Request Access to Gemma 3
- Visit: https://huggingface.co/google/gemma-3-4b-it
- Click the **"Agree and access repository"** button
- You may need to accept Google's terms and conditions
- Access is usually granted immediately

### 3. Get Your HuggingFace Access Token
- Go to: https://huggingface.co/settings/tokens
- Click **"New token"** or **"Create new token"**
- Name it: `gemma-access` (or any name you like)
- Token type: Select **"Read"** (this is sufficient for downloading models)
- Click **"Generate token"**
- **IMPORTANT:** Copy the token immediately - you won't see it again!

### 4. Login with HuggingFace CLI

Open your terminal/command prompt in the project directory and run:

```bash
# Activate virtual environment first
venv\Scripts\activate

# Login to HuggingFace
huggingface-cli login
```

When prompted:
- Paste your token (it won't show as you type - this is normal)
- Press Enter
- When asked "Add token as git credential?", type `n` (not needed)

You should see: `✓ Login successful`

### 5. Download Gemma 3 via Model Selector

Now you can download Gemma 3:

```bash
# Run the batch file
start_app.bat

# Answer 'y' to reconfigure models
# Select Gemma 3 4B Instruct (option 28)
```

The model will now download successfully!

---

## Alternative: Set Token as Environment Variable

Instead of using `huggingface-cli login`, you can set the token as an environment variable:

### Windows (Command Prompt):
```cmd
set HF_TOKEN=your_token_here
```

### Windows (PowerShell):
```powershell
$env:HF_TOKEN="your_token_here"
```

### Permanent (Windows):
1. Search for "Environment Variables" in Windows
2. Click "Edit the system environment variables"
3. Click "Environment Variables..." button
4. Under "User variables", click "New"
5. Variable name: `HF_TOKEN`
6. Variable value: `your_token_here`
7. Click OK

---

## Troubleshooting

### Error: "401 Client Error: Unauthorized"
- You haven't logged in with HuggingFace CLI
- Solution: Run `huggingface-cli login` and enter your token

### Error: "Access to model google/gemma-3-4b-it is restricted"
- You haven't requested access to the model
- Solution: Visit https://huggingface.co/google/gemma-3-4b-it and click "Agree and access repository"

### Error: "Cannot access gated repo"
- Your token doesn't have the right permissions
- Solution:
  1. Go to https://huggingface.co/settings/tokens
  2. Create a new token with "Read" permission
  3. Re-run `huggingface-cli login` with the new token

### Model downloads to cache but not to local folder
- This is normal! The pipeline method downloads to HuggingFace cache first
- The model_selector will then copy it to the local folder

---

## Quick Test

After logging in, test if authentication works:

```bash
# Activate venv
venv\Scripts\activate

# Test login
python -c "from huggingface_hub import whoami; print(whoami())"
```

If you see your username, authentication is working!

---

## Current App Status

✅ **Your app is working right now!**
- Embedding: all-MiniLM-L6-v2 ✓
- QA Model: deepset/electra-base-squad2 ✓
- Generator: None (extractive mode)

**To enable Gemma 3:**
1. Follow the steps above to authenticate
2. Run `start_app.bat` and answer 'y' to reconfigure
3. Select Gemma 3 4B Instruct
4. Wait for download (8GB, may take 10-30 minutes depending on internet speed)
5. Restart the app - Gemma 3 will be active!

---

## Alternative Models (No Authentication Required)

If you don't want to set up HuggingFace authentication, these models work without it:

**Recommended alternatives:**
- **google/flan-t5-base** (900MB) - Excellent for QA tasks
- **google/flan-t5-large** (3GB) - Even better quality
- **gpt2-medium** (1.5GB) - Good general-purpose model
- **gpt2-large** (3GB) - Better quality

To use these:
1. Run `start_app.bat`
2. Answer 'y' to reconfigure
3. Select one of the models above
4. Download will work immediately (no authentication needed)

---

**Need help?** Check the error messages - they now provide helpful guidance!
