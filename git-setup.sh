#!/bin/bash

# Git Setup Script for PDF QA System
# This script helps you initialize and push your project to GitHub

echo "================================================"
echo "  PDF QA System - Git Setup Wizard"
echo "================================================"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first:"
    echo "   Download from: https://git-scm.com/downloads"
    exit 1
fi

echo "✅ Git is installed"
echo ""

# Check if already a git repository
if [ -d ".git" ]; then
    echo "⚠️  This is already a Git repository."
    echo ""
    read -p "Do you want to reinitialize? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 0
    fi
    rm -rf .git
fi

# Initialize git
echo "📁 Initializing Git repository..."
git init
echo ""

# Set default branch to main
git branch -M main
echo "✅ Set default branch to 'main'"
echo ""

# Add all files
echo "📝 Adding files to Git..."
git add .
echo ""

# Show what will be committed
echo "📋 Files to be committed:"
git status --short
echo ""

# Get commit message
read -p "Enter commit message (or press Enter for default): " COMMIT_MSG
if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="Initial commit: PDF QA System with improved accuracy and full context support"
fi

# Create first commit
echo ""
echo "💾 Creating initial commit..."
git commit -m "$COMMIT_MSG"
echo ""

# Get repository URL
echo "================================================"
echo "  GitHub Repository Setup"
echo "================================================"
echo ""
echo "Steps to create GitHub repository:"
echo "1. Go to: https://github.com/new"
echo "2. Create a new repository (private recommended)"
echo "3. Don't initialize with README"
echo "4. Copy the repository URL"
echo ""
read -p "Enter your GitHub repository URL (e.g., https://github.com/username/repo.git): " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo ""
    echo "⚠️  No URL provided. You can add it later with:"
    echo "   git remote add origin YOUR_REPO_URL"
    echo "   git push -u origin main"
    exit 0
fi

# Add remote
echo ""
echo "🔗 Adding remote repository..."
git remote add origin "$REPO_URL"
echo ""

# Push to GitHub
echo "🚀 Pushing to GitHub..."
read -p "Ready to push? (Y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    git push -u origin main
    echo ""
    echo "================================================"
    echo "  ✅ SUCCESS!"
    echo "================================================"
    echo ""
    echo "Your code is now on GitHub!"
    echo "Repository: $REPO_URL"
    echo ""
    echo "Next steps:"
    echo "1. SSH to your server: ssh user@172.16.20.12"
    echo "2. Clone: git clone $REPO_URL"
    echo "3. Run: cd pdf-qa-system && ./start_app.sh"
    echo ""
else
    echo ""
    echo "Skipped push. You can push later with:"
    echo "  git push -u origin main"
fi

echo ""
echo "For deployment instructions, see:"
echo "  - QUICK_START.md (Quick reference)"
echo "  - GIT_DEPLOYMENT.md (Detailed guide)"
echo ""
