#!/usr/bin/env python3
"""
Setup script for ClickSentry - Phishing URL Detection
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up ClickSentry - Phishing URL Detection")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Install requirements
    print("\n📦 Installing dependencies...")
    if not run_command("pip install -r requirements.txt"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Train the model
    print("\n🧠 Training machine learning model...")
    if not run_command("python train_model.py"):
        print("❌ Failed to train model")
        sys.exit(1)
    
    # Check if model files exist
    if os.path.exists("phishing_model.pkl") and os.path.exists("feature_names.pkl"):
        print("✓ Model files created successfully")
    else:
        print("❌ Model files not found")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo start the application:")
    print("  python app.py")
    print("\nThen open your browser and go to:")
    print("  http://localhost:5000")

if __name__ == "__main__":
    main()