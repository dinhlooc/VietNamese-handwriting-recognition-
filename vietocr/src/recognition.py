import yaml
import os
import torch
from PIL import Image
import sys

# Check if vietocr is installed
try:
    from vietocr.tool.predictor import Predictor
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "vietocr"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pillow"])
    from vietocr.tool.predictor import Predictor

# Get project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define paths relative to project root
config_path = os.path.join(project_root, 'config', 'config.yml')
weights_path = os.path.join(project_root, 'model', 'transformerocr.pth')
image_dir = os.path.join(project_root, 'image')

def load_config():
    """Load and modify config to use CPU instead of GPU"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    # Modify config to use CPU instead of GPU
    config['device'] = 'cpu'
    print(f"Using device: {config['device']}")
    
    return config

def predict_image(image_path, config):
    """Predict text from image using VietOCR"""
    # Initialize predictor with config
    predictor = Predictor(config)
    
    # Load weights
    print(f"Loading weights from: {weights_path}")
    predictor.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    
    # Open image and predict
    image = Image.open(image_path)
    result = predictor.predict(image)
    
    return result

def main():
    """Main function to run prediction on all images in the image directory"""
    # Load config
    config = load_config()
    
    # Get all image files in image directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} image(s) to process")
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"\nProcessing image: {image_file}")
        
        try:
            result = predict_image(image_path, config)
            print(f"Recognized text: {result}")
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")

if __name__ == "__main__":
    main()