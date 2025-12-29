# Quick Start - Git Deployment

## TL;DR - 5 Minute Deployment

### On Windows (One Time):

```bash
# Open Git Bash in your project folder
cd d:/A/LLM/GPT2\(M\)/PDF-QA-System

# Initialize and push to GitHub
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/pdf-qa-system.git
git push -u origin main
```

### On Ubuntu Server (172.16.20.12):

```bash
# Clone and run
git clone https://github.com/YOUR_USERNAME/pdf-qa-system.git
cd pdf-qa-system
./start_app.sh
```

**Done!** Access at: http://172.16.20.12:5000

---

## Update Application (After Changes)

### Windows:
```bash
git add .
git commit -m "Your changes"
git push
```

### Server:
```bash
cd ~/pdf-qa-system
git pull
sudo systemctl restart pdf-qa  # if using service
```

---

## Setup as Service (Production)

```bash
# Create service file
sudo nano /etc/systemd/system/pdf-qa.service
```

Paste (replace YOUR_USERNAME):
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

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now pdf-qa
sudo systemctl status pdf-qa
```

---

## Useful Commands

```bash
# Check service
sudo systemctl status pdf-qa

# View logs
sudo journalctl -u pdf-qa -f

# Restart after updates
sudo systemctl restart pdf-qa

# Stop service
sudo systemctl stop pdf-qa
```

---

For detailed guide, see [GIT_DEPLOYMENT.md](GIT_DEPLOYMENT.md)
