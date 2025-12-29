#!/bin/bash

# Prepare deployment package for PDF QA System

echo "Preparing deployment package..."

# Create deployment directory
DEPLOY_DIR="pdf-qa-deploy"
rm -rf $DEPLOY_DIR
mkdir -p $DEPLOY_DIR

# Copy essential files
echo "Copying application files..."
cp app.py $DEPLOY_DIR/
cp config.py $DEPLOY_DIR/
cp pdf_processor.py $DEPLOY_DIR/
cp qa_engine.py $DEPLOY_DIR/
cp requirements.txt $DEPLOY_DIR/
cp start_app.sh $DEPLOY_DIR/
cp DEPLOYMENT_GUIDE.md $DEPLOY_DIR/
cp README.md $DEPLOY_DIR/
cp .gitignore $DEPLOY_DIR/

# Copy directories
echo "Copying directories..."
cp -r static $DEPLOY_DIR/
cp -r templates $DEPLOY_DIR/

# Create necessary directories
mkdir -p $DEPLOY_DIR/uploads
mkdir -p $DEPLOY_DIR/data
mkdir -p $DEPLOY_DIR/logs

# Create .gitignore for deployment
cat > $DEPLOY_DIR/.gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
*.egg-info/
dist/
build/

# Virtual Environments
venv/
ENV/
env/

# Application specific
uploads/
data/
*.pkl
*.faiss

# Logs
logs/
*.log

# Environment variables
.env

# Model cache
.cache/
EOF

# Make scripts executable
chmod +x $DEPLOY_DIR/start_app.sh

# Create archive
echo "Creating archive..."
tar -czf pdf-qa-system.tar.gz $DEPLOY_DIR/

echo ""
echo "Deployment package created: pdf-qa-system.tar.gz"
echo ""
echo "To deploy to server:"
echo "  1. Transfer: scp pdf-qa-system.tar.gz user@172.16.20.12:~/"
echo "  2. SSH:      ssh user@172.16.20.12"
echo "  3. Extract:  tar -xzf pdf-qa-system.tar.gz"
echo "  4. Run:      cd pdf-qa-deploy && ./start_app.sh"
echo ""
