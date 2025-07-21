#!/usr/bin/env python3
"""
Installation guide for Visual Designer dependencies
Specifically CLIP and related packages
"""

import subprocess
import sys
import os

def print_banner():
    print("🔧 Visual Designer Dependencies Installation")
    print("=" * 60)
    print("📦 Installing CLIP and required packages...")
    print()

def install_package(package_name, pip_command=None):
    """Install a package using pip"""
    if pip_command is None:
        pip_command = f"pip install {package_name}"
    
    print(f"📦 Installing {package_name}...")
    try:
        subprocess.check_call(pip_command, shell=True)
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False

def check_installation():
    """Check if packages are properly installed"""
    print("\n🔍 Checking installations...")
    
    packages_to_check = [
        ("torch", "import torch; print(f'PyTorch: {torch.__version__}')"),
        ("torchvision", "import torchvision; print(f'torchvision: {torchvision.__version__}')"),
        ("PIL", "from PIL import Image; print('Pillow: OK')"),
        ("clip", "import clip; print('CLIP: OK')"),
        ("ftfy", "import ftfy; print('ftfy: OK')"),
        ("regex", "import regex; print('regex: OK')"),
        ("tqdm", "import tqdm; print('tqdm: OK')")
    ]
    
    all_good = True
    for package, test_code in packages_to_check:
        try:
            exec(test_code)
        except ImportError as e:
            print(f"❌ {package} not available: {e}")
            all_good = False
        except Exception as e:
            print(f"⚠️  {package} import warning: {e}")
    
    return all_good

def main():
    print_banner()
    
    # List of packages to install
    packages = [
        ("PyTorch & torchvision", "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"),
        ("Pillow (PIL)", "pip install Pillow"),
        ("ftfy", "pip install ftfy"),
        ("regex", "pip install regex"),
        ("tqdm", "pip install tqdm"),
        ("CLIP", "pip install git+https://github.com/openai/CLIP.git")
    ]
    
    print("📋 Packages to install:")
    for name, _ in packages:
        print(f"  • {name}")
    print()
    
    # Ask for confirmation
    response = input("🤔 Do you want to proceed with installation? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Installation cancelled.")
        return
    
    print("\n🚀 Starting installation...")
    
    # Install packages
    success_count = 0
    for name, command in packages:
        if install_package(name, command):
            success_count += 1
        print()
    
    print(f"📊 Installation Summary: {success_count}/{len(packages)} packages installed")
    
    # Check installations
    if check_installation():
        print("\n🎉 All dependencies installed successfully!")
        print("✅ Visual Designer is ready to use!")
    else:
        print("\n⚠️  Some packages may not be properly installed.")
        print("💡 Try running the failed installations manually.")
    
    print("\n📖 Next steps:")
    print("1. Run: python run_designer.py")
    print("2. Visit: http://localhost:8004/docs")
    print("3. Test visual search with /visual-search endpoint")

if __name__ == "__main__":
    main()
