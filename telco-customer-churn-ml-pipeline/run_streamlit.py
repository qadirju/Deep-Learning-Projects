#!/usr/bin/env python
"""
Streamlit App Launcher
Installs dependencies and runs the Telco Churn Prediction Streamlit app
"""

import subprocess
import sys
import os

def install_package(package_name):
    """Install a package using pip"""
    print(f"📦 Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Could not install {package_name}: {e}")
        return False

def main():
    """Main launcher function"""
    print("=" * 80)
    print("🚀 Telco Customer Churn Prediction - Streamlit App Launcher")
    print("=" * 80)
    
    # Change to app directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Check and install dependencies
    required_packages = ["streamlit"]
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is already installed")
        except ImportError:
            install_package(package)
    
    print("\n" + "=" * 80)
    print("📊 Starting Streamlit Application...")
    print("=" * 80)
    print("\n🌐 Open your browser and go to: http://localhost:8501")
    print("📌 Press Ctrl+C to stop the server\n")
    
    # Run streamlit
    try:
        subprocess.call([
            sys.executable, "-m", "streamlit", "run", 
            "app/streamlit_app.py",
            "--logger.level=info"
        ])
    except KeyboardInterrupt:
        print("\n\n✋ Stopping Streamlit app...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
