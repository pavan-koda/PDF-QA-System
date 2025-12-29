# Git-Based Deployment Guide

This guide shows you how to deploy the PDF QA System to your Ubuntu server using Git (GitHub, GitLab, or Bitbucket).

## Why Use Git?

✅ **Easy Updates** - Just `git pull` to get latest changes
✅ **Version Control** - Track all changes and rollback if needed
✅ **Clean Deployment** - No manual file transfers
✅ **Best Practice** - Standard industry approach

---

## Step 1: Setup Git Repository (One Time)

### Option A: GitHub (Recommended)

1. **Create a new repository on GitHub**:
   - Go to https://github.com/new
   - Repository name: `pdf-qa-system`
   - Visibility: Private (recommended) or Public
   - Don't initialize with README (we already have files)
   - Click "Create repository"

2. **On your Windows machine**, open Git Bash or PowerShell in your project directory:

```bash
cd d:/A/LLM/GPT2\(M\)/PDF-QA-System

# Initialize git (if not already done)
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: PDF QA System with improved accuracy"

# Add your GitHub repository as remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/pdf-qa-system.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Option B: GitLab or Bitbucket

Same steps as above, but use your GitLab/Bitbucket repository URL instead.

---

## Step 2: Deploy to Ubuntu Server

### Initial Deployment

SSH into your Ubuntu server:

```bash
ssh YOUR_USERNAME@172.16.20.12
```

Then run these commands:

```bash
# Navigate to your home directory
cd ~

# Clone your repository (replace with your actual repo URL)
git clone https://github.com/YOUR_USERNAME/pdf-qa-system.git

# Navigate into the project
cd pdf-qa-system

# Run the setup script (it will create venv and install dependencies)
chmod +x start_app.sh
./start_app.sh
```

That's it! The app will:
- Automatically create a virtual environment
- Install all dependencies from `requirements.txt`
- Start the Flask server on port 5000

Access at: **http://172.16.20.12:5000**

---

## Step 3: Running in Production (Recommended)

### Using systemd service (Auto-start on boot)

Create a service file:

```bash
sudo nano /etc/systemd/system/pdf-qa.service
```

Paste this content (replace YOUR_USERNAME with your actual username):

```ini
[Unit]
Description=PDF QA System
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/pdf-qa-system
Environment="PATH=/home/YOUR_USERNAME/pdf-qa-system/venv/bin"
ExecStart=/home/YOUR_USERNAME/pdf-qa-system/venv/bin/python3 app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Save and exit (Ctrl+X, then Y, then Enter).

Enable and start the service:

```bash
# Reload systemd
sudo systemctl daemon-reload

# Start the service
sudo systemctl start pdf-qa

# Enable on boot
sudo systemctl enable pdf-qa

# Check status
sudo systemctl status pdf-qa
```

---

## Step 4: Updating Your Application

When you make changes on your Windows machine:

### On Windows (Push changes):

```bash
cd d:/A/LLM/GPT2\(M\)/PDF-QA-System

# Add changed files
git add .

# Commit changes
git commit -m "Describe your changes here"

# Push to GitHub
git push
```

### On Ubuntu Server (Pull changes):

```bash
# SSH to server
ssh YOUR_USERNAME@172.16.20.12

# Navigate to project
cd ~/pdf-qa-system

# Pull latest changes
git pull

# Restart the service
sudo systemctl restart pdf-qa

# Check if running
sudo systemctl status pdf-qa
```

That's it! Your changes are deployed.

---

## Common Git Commands

### On Windows (Development):

```bash
# Check status
git status

# See changes
git diff

# Add all changes
git add .

# Commit with message
git commit -m "Your message"

# Push to remote
git push

# Pull latest from remote
git pull

# View commit history
git log --oneline
```

### On Server (Deployment):

```bash
# Pull latest changes
git pull

# Check current branch
git branch

# Switch branch (if needed)
git checkout branch-name

# Discard local changes (careful!)
git reset --hard origin/main

# View what changed
git log -p
```

---

## Troubleshooting

### Authentication Issues

If GitHub asks for password repeatedly, use SSH or Personal Access Token:

**Option 1: Personal Access Token**
1. GitHub → Settings → Developer settings → Personal access tokens → Generate new token
2. Give it `repo` permissions
3. Use token as password when pushing

**Option 2: SSH Key (Better)**
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings → SSH and GPG keys → New SSH key

# Change remote to SSH
git remote set-url origin git@github.com:YOUR_USERNAME/pdf-qa-system.git
```

### Permission Issues on Server

```bash
# Make sure you own the files
sudo chown -R YOUR_USERNAME:YOUR_USERNAME ~/pdf-qa-system

# Make start script executable
chmod +x ~/pdf-qa-system/start_app.sh
```

### Service Not Starting

```bash
# Check logs
sudo journalctl -u pdf-qa -n 50 --no-pager

# Check if port 5000 is in use
sudo netstat -tulpn | grep 5000

# Kill process using port 5000 (if needed)
sudo kill $(sudo lsof -t -i:5000)
```

---

## Quick Reference Card

### First Time Setup:
```bash
# Windows
git init
git add .
git commit -m "Initial commit"
git remote add origin YOUR_REPO_URL
git push -u origin main

# Server
git clone YOUR_REPO_URL
cd pdf-qa-system
./start_app.sh
```

### Update Workflow:
```bash
# Windows - After making changes
git add .
git commit -m "Description"
git push

# Server - Deploy updates
git pull
sudo systemctl restart pdf-qa
```

### Check Status:
```bash
# Server
sudo systemctl status pdf-qa      # Service status
tail -f logs/performance.txt       # Application logs
sudo journalctl -u pdf-qa -f       # System logs
```

---

## Advanced: Using Git Branches

For safe deployments, use branches:

### Development Branch Workflow:

```bash
# Windows - Create dev branch
git checkout -b development
# Make changes, test locally
git add .
git commit -m "New feature"
git push -u origin development

# When ready for production
git checkout main
git merge development
git push

# Server - Deploy from main
git pull origin main
sudo systemctl restart pdf-qa
```

---

## Security Best Practices

1. **Use Private Repository** for production code
2. **Never commit**:
   - `.env` files with secrets
   - API keys or passwords
   - Database credentials
   - User uploaded files
3. **Use SSH keys** instead of passwords
4. **Review changes** before deploying: `git diff`

---

## Benefits Summary

| Method | Pros | Cons |
|--------|------|------|
| **Git** | ✅ Version control<br>✅ Easy updates<br>✅ Rollback capability<br>✅ Industry standard | ⚠️ Need Git knowledge<br>⚠️ Initial setup |
| **SCP/FTP** | ✅ Simple<br>✅ Direct transfer | ❌ No version history<br>❌ Manual process<br>❌ Error-prone |

**Recommendation:** Use Git for professional, maintainable deployments.

---

## Next Steps

1. ✅ Push code to GitHub/GitLab
2. ✅ Clone on server
3. ✅ Setup systemd service
4. ✅ Test the deployment
5. ✅ Make changes and use `git pull` to update

Your deployment is now **production-ready** and **maintainable**! 🚀
