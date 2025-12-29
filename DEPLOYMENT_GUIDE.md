# PDF QA System - Server Deployment Guide

## Server Information
- **Server IP**: 172.16.20.12
- **OS**: Ubuntu
- **User**: Your username

## Deployment Steps

### 1. Transfer Files to Server

#### Option A: Using SCP (Secure Copy)
From your Windows machine (Git Bash, WSL, or PowerShell):

```bash
# Create a zip file of your project (exclude venv, data, uploads)
cd d:\A\LLM\GPT2(M)\PDF-QA-System
tar -czf pdf-qa-system.tar.gz \
  --exclude=venv \
  --exclude=__pycache__ \
  --exclude=data \
  --exclude=uploads \
  --exclude=logs \
  --exclude=.claude \
  app.py config.py pdf_processor.py qa_engine.py \
  requirements.txt start_app.sh \
  static/ templates/

# Transfer to server (replace YOUR_USERNAME with your username)
scp pdf-qa-system.tar.gz YOUR_USERNAME@172.16.20.12:~/
```

#### Option B: Using Git (if you have a repository)
```bash
# On server
git clone YOUR_REPO_URL
cd pdf-qa-system
```

#### Option C: Using FileZilla or WinSCP
- Install FileZilla or WinSCP
- Connect to 172.16.20.12
- Upload the project folder

### 2. Setup on Server

SSH into your server:
```bash
ssh YOUR_USERNAME@172.16.20.12
```

Then run these commands:

```bash
# Extract files (if using Option A)
cd ~
tar -xzf pdf-qa-system.tar.gz

# Navigate to project directory
cd PDF-QA-System  # or your extracted folder name

# Check Python version (need Python 3.8+)
python3 --version

# Install pip if not available
sudo apt update
sudo apt install python3-pip python3-venv -y

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads data logs static templates

# Make start script executable
chmod +x start_app.sh
```

### 3. Configure the Application

```bash
# Edit config if needed
nano config.py

# Verify the configuration
python3 -c "from config import *; print('Config loaded successfully')"
```

### 4. Run the Application

#### Option A: Run Directly (Testing)
```bash
# Activate venv if not already
source venv/bin/activate

# Run the app
python3 app.py
```

The app will be available at: `http://172.16.20.12:5000`

#### Option B: Run in Background (Production)
```bash
# Using nohup
nohup python3 app.py > logs/app.log 2>&1 &

# Check if running
ps aux | grep app.py

# View logs
tail -f logs/app.log
```

#### Option C: Using systemd (Best for Production)
Create a service file:

```bash
sudo nano /etc/systemd/system/pdf-qa.service
```

Add this content:
```ini
[Unit]
Description=PDF QA System
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/PDF-QA-System
Environment="PATH=/home/YOUR_USERNAME/PDF-QA-System/venv/bin"
ExecStart=/home/YOUR_USERNAME/PDF-QA-System/venv/bin/python3 app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:
```bash
# Reload systemd
sudo systemctl daemon-reload

# Start service
sudo systemctl start pdf-qa

# Enable on boot
sudo systemctl enable pdf-qa

# Check status
sudo systemctl status pdf-qa

# View logs
sudo journalctl -u pdf-qa -f
```

### 5. Configure Firewall (if needed)

```bash
# Allow port 5000
sudo ufw allow 5000/tcp

# Check firewall status
sudo ufw status
```

### 6. Access the Application

From any browser on your network:
```
http://172.16.20.12:5000
```

## Troubleshooting

### Check if app is running:
```bash
ps aux | grep app.py
netstat -tulpn | grep 5000
```

### View logs:
```bash
tail -f logs/app.log
tail -f logs/performance.txt
```

### Restart the service:
```bash
sudo systemctl restart pdf-qa
```

### Stop the service:
```bash
sudo systemctl stop pdf-qa
```

### Kill the process if stuck:
```bash
pkill -f app.py
```

## Production Recommendations

### 1. Use Gunicorn (Production WSGI Server)

Install:
```bash
pip install gunicorn
```

Run:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Update systemd service:
```ini
ExecStart=/home/YOUR_USERNAME/PDF-QA-System/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 2. Use Nginx as Reverse Proxy

Install Nginx:
```bash
sudo apt install nginx -y
```

Configure:
```bash
sudo nano /etc/nginx/sites-available/pdf-qa
```

Add:
```nginx
server {
    listen 80;
    server_name 172.16.20.12;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    client_max_body_size 16M;
}
```

Enable:
```bash
sudo ln -s /etc/nginx/sites-available/pdf-qa /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

Now accessible at: `http://172.16.20.12` (port 80)

## Quick Commands Reference

```bash
# Start
sudo systemctl start pdf-qa

# Stop
sudo systemctl stop pdf-qa

# Restart
sudo systemctl restart pdf-qa

# Status
sudo systemctl status pdf-qa

# Logs
sudo journalctl -u pdf-qa -f

# Update code
cd ~/PDF-QA-System
git pull  # if using git
sudo systemctl restart pdf-qa
```

## Notes

- Default port: 5000
- Max file size: 16MB
- Supported format: PDF only
- Log files: `logs/performance.txt`
