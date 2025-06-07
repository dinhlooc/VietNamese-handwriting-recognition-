import yaml
import os
import torch
import time
import cv2
import numpy as np
from PIL import Image
from vietocr.tool.predictor import Predictor
import sys
from paddleocr import PaddleOCR

# Get project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# File paths
config_path = os.path.join(project_root, 'config', 'config.yml')
weights_path = os.path.join(project_root, 'model', 'transformerocr.pth')
test_image_path = os.path.join(project_root, 'image', 'test4.png')
output_dir = os.path.join(project_root, 'output')
os.makedirs(output_dir, exist_ok=True)

def load_config():
    """Load and modify config to use CPU instead of GPU"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    # Modify config to use CPU instead of GPU
    config['device'] = 'cpu'
    
    # Ensure the weights path is set correctly
    config['weights'] = weights_path
    
    print(f"Using device: {config['device']}")
    print(f"Using model weights from: {weights_path}")
    
    return config

def detect_lines_paddleocr(image_path):
    """Detect text lines in an image using PaddleOCR"""
    print("Initializing PaddleOCR for line detection...")
    # Use Vietnamese language model for better detection of Vietnamese text
    ocr = PaddleOCR(use_angle_cls=True, lang='vi', use_gpu=False)
    
    # Read the image for visualization
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Create a copy for visualization
    visual_img = img.copy()
    
    print("Running PaddleOCR detection...")
    # Get detection results
    result = ocr.ocr(image_path, cls=True)
    
    # Extract bounding boxes
    line_boxes = []
    for idx, line in enumerate(result[0]):
        if line:
            points = line[0]  # Get the detection box coordinates
            # Convert to [x1, y1, x2, y2] format for easy processing
            x1 = min(int(points[0][0]), int(points[3][0]))
            y1 = min(int(points[0][1]), int(points[1][1]))
            x2 = max(int(points[1][0]), int(points[2][0]))
            y2 = max(int(points[2][1]), int(points[3][1]))
            
            # Add padding to ensure entire text is captured
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img.shape[1], x2 + padding)
            y2 = min(img.shape[0], y2 + padding)
            
            line_boxes.append((x1, y1, x2, y2))
    
    # Sort by vertical position (top to bottom)
    line_boxes.sort(key=lambda box: box[1])
    
    return line_boxes, img, visual_img

def enhance_line_image(line_img):
    """Apply image enhancement for better recognition"""
    # Convert to grayscale if not already
    if len(line_img.shape) == 3:
        gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = line_img
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Convert back to RGB for VietOCR
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return rgb

def main():
    print("Starting line detection and recognition for test4.png...")
    config = load_config()
    
    print("Initializing VietOCR...")
    try:
        # Create predictor with custom config that includes correct weights path
        predictor = Predictor(config)
        
        # Additional verification that the weights file exists
        if not os.path.exists(weights_path):
            print(f"WARNING: Model file does not exist at: {weights_path}")
            print("Current directory:", os.getcwd())
            print("Checking if file exists with different case...")
            # Check directory contents to help debug
            model_dir = os.path.dirname(weights_path)
            if os.path.exists(model_dir):
                print("Files in model directory:", os.listdir(model_dir))
        
        print(f"Loading model weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu')
        predictor.model.load_state_dict(state_dict)
        
        # Detect lines using PaddleOCR (without fallback)
        print("Using PaddleOCR for text line detection")
        line_boxes, img, visual_img = detect_lines_paddleocr(test_image_path)
        print(f"Detected {len(line_boxes)} text lines")
        
        # Process each line
        results = []
        for i, (x1, y1, x2, y2) in enumerate(line_boxes):
            # Crop the line from the image
            line_img = img[y1:y2, x1:x2]
            
            # Skip empty or invalid crops
            if line_img.size == 0 or line_img.shape[0] == 0 or line_img.shape[1] == 0:
                continue
                
            # Apply enhancement
            enhanced_line = enhance_line_image(line_img)
            
            # Convert to PIL Image for VietOCR
            pil_line = Image.fromarray(enhanced_line)
            
            # Recognize text
            print(f"Recognizing line {i+1}...")
            text = predictor.predict(pil_line)
            results.append((text, (x1, y1, x2, y2)))
            
            # Draw bounding box and line number on visualization image
            cv2.rectangle(visual_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(visual_img, f"{i+1}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 0, 255), 2)
        
        # Save the visualization
        output_path = os.path.join(output_dir, 'test4_paddle_detection.jpg')
        cv2.imwrite(output_path, visual_img)
        print(f"Visualization saved to {output_path}")
        
        # Print recognition results
        print("\nRecognition Results:")
        print("-" * 50)
        for i, (text, _) in enumerate(results):
            print(f"Line {i+1}: {text}")
        print("-" * 50)
        
        # Write results to a text file
        result_text_path = os.path.join(output_dir, 'test4_paddle_recognition_result.txt')
        with open(result_text_path, 'w', encoding='utf-8') as f:
            for i, (text, _) in enumerate(results):
                f.write(f"Line {i+1}: {text}\n")
        
        print(f"Results saved to {result_text_path}")
    
    except Exception as e:
        print(f"Error initializing VietOCR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()