"""
Setup Verification Script
Checks if all components are properly installed and configured
Run this before using the vision PDF QA system
"""

import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def print_status(component, status, details=""):
    """Print status with icon."""
    icon = "✓" if status else "✗"
    status_text = "OK" if status else "FAILED"
    print(f"[{icon}] {component:30s} {status_text}")
    if details:
        print(f"    {details}")

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    required = (3, 9)
    is_ok = version >= required
    details = f"Version: {version.major}.{version.minor}.{version.micro}"
    if not is_ok:
        details += f" (Required: {required[0]}.{required[1]}+)"
    print_status("Python", is_ok, details)
    return is_ok

def check_module(module_name, import_name=None):
    """Check if Python module is installed."""
    if import_name is None:
        import_name = module_name

    try:
        __import__(import_name)
        print_status(f"Module: {module_name}", True)
        return True
    except ImportError:
        print_status(f"Module: {module_name}", False, f"Run: pip install {module_name}")
        return False

def check_ollama():
    """Check if Ollama is installed and running."""
    # Check installation
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            print_status("Ollama Installed", True, version)
        else:
            print_status("Ollama Installed", False, "Install from: https://ollama.ai/download")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print_status("Ollama Installed", False, "Install from: https://ollama.ai/download")
        return False

    # Check if running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print_status("Ollama Running", True, "Server is active")
            return True
        else:
            print_status("Ollama Running", False, "Run: ollama serve")
            return False
    except Exception:
        print_status("Ollama Running", False, "Run: ollama serve")
        return False

def check_ollama_models():
    """Check if required Ollama models are installed."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            output = result.stdout

            # Check for Llama 3.2-Vision
            has_vision = 'llama3.2-vision' in output
            if has_vision:
                print_status("Llama 3.2-Vision Model", True, "Model found")
            else:
                print_status("Llama 3.2-Vision Model", False, "Run: ollama pull llama3.2-vision:11b")

            return has_vision
        else:
            print_status("Ollama Models", False, "Cannot list models")
            return False
    except Exception as e:
        print_status("Ollama Models", False, str(e))
        return False

def check_files():
    """Check if required project files exist."""
    required_files = [
        'vision_pdf_processor.py',
        'colpali_retriever.py',
        'vision_qa_engine.py',
        'app_vision.py',
        'requirements_vision.txt'
    ]

    all_exist = True
    for file in required_files:
        exists = Path(file).exists()
        print_status(f"File: {file}", exists)
        all_exist = all_exist and exists

    return all_exist

def check_directories():
    """Check if required directories exist (create if needed)."""
    dirs = ['uploads', 'data', 'logs', 'processed_pdfs', 'chroma_db']

    for dir_name in dirs:
        path = Path(dir_name)
        if not path.exists():
            path.mkdir(exist_ok=True)
            print_status(f"Directory: {dir_name}", True, "Created")
        else:
            print_status(f"Directory: {dir_name}", True, "Exists")

    return True

def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            print_status("GPU (CUDA)", True, f"Available: {gpu_name}")
        else:
            print_status("GPU (CUDA)", False, "Using CPU (slower but works)")
        return True  # GPU is optional
    except ImportError:
        print_status("GPU (CUDA)", False, "PyTorch not installed")
        return False

def main():
    """Run all checks."""
    print_header("VISION PDF QA SYSTEM - SETUP VERIFICATION")

    print("Checking Python environment...")
    checks = []

    # Python version
    checks.append(("Python Version", check_python_version()))

    # Required Python modules
    print("\nChecking Python modules...")
    modules = [
        ('flask', 'flask'),
        ('PyMuPDF', 'fitz'),
        ('chromadb', 'chromadb'),
        ('transformers', 'transformers'),
        ('sentence-transformers', 'sentence_transformers'),
        ('torch', 'torch'),
        ('faiss-cpu', 'faiss'),
        ('Pillow', 'PIL'),
        ('requests', 'requests'),
    ]

    for module_name, import_name in modules:
        checks.append((f"Module: {module_name}", check_module(module_name, import_name)))

    # Ollama
    print("\nChecking Ollama...")
    checks.append(("Ollama", check_ollama()))
    checks.append(("Ollama Models", check_ollama_models()))

    # Project files
    print("\nChecking project files...")
    checks.append(("Project Files", check_files()))

    # Directories
    print("\nChecking directories...")
    checks.append(("Directories", check_directories()))

    # GPU (optional)
    print("\nChecking GPU (optional)...")
    checks.append(("GPU", check_gpu()))

    # Summary
    print_header("SUMMARY")

    passed = sum(1 for _, status in checks if status)
    total = len(checks)
    critical_checks = checks[:10]  # First 10 are critical
    critical_passed = sum(1 for _, status in critical_checks if status)

    print(f"Checks Passed: {passed}/{total}")
    print(f"Critical Checks: {critical_passed}/{len(critical_checks)}")

    if critical_passed == len(critical_checks):
        print("\n✅ ALL CRITICAL CHECKS PASSED!")
        print("   You're ready to use the Vision PDF QA System!")
        print("\n   Start the application:")
        print("   - Windows: start_vision_app.bat")
        print("   - Linux/macOS: ./start_vision_app.sh")
        print("   - Manual: python app_vision.py")
        return 0
    else:
        print("\n❌ SOME CRITICAL CHECKS FAILED")
        print("   Please fix the issues above before starting the application.")
        print("\n   Common fixes:")
        print("   - Install missing modules: pip install -r requirements_vision.txt")
        print("   - Install Ollama: https://ollama.ai/download")
        print("   - Download model: ollama pull llama3.2-vision:11b")
        print("   - Start Ollama: ollama serve")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
