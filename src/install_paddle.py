import subprocess
import sys
import os
import time

def install_dependencies():
    print("Installing PaddlePaddle (this may take a few minutes)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "paddlepaddle"])
    print("PaddlePaddle installation complete.")
    
    # Verify PaddleOCR installation
    print("Verifying PaddleOCR installation...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "paddleocr"])
        print("PaddleOCR installation complete.")
    except:
        print("PaddleOCR already installed.")

if __name__ == "__main__":
    install_dependencies()
    print("\nAll dependencies installed. Now you can run process_test4.py")
    print("Run with: python src/process_test4.py")