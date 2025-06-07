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
import argparse
from PIL import ImageFont, ImageDraw

def draw_vietnamese_text(image, text, position, font_path=None, font_size=24, color=(255, 0, 0), bg_color=None):
    """Cải thiện hàm vẽ văn bản tiếng Việt lên hình ảnh OpenCV
    
    Args:
        image: Hình ảnh OpenCV (BGR)
        text: Văn bản tiếng Việt cần hiển thị
        position: Vị trí (x, y) để hiển thị văn bản
        font_path: Đường dẫn đến font chữ (nếu None, sẽ tự động tìm font)
        font_size: Kích thước font
        color: Màu chữ theo định dạng (R, G, B)
        bg_color: Màu nền (R, G, B), None nếu không có nền
    
    Returns:
        Hình ảnh sau khi thêm văn bản
    """
    # Convert OpenCV image (BGR) to PIL image (RGB)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    try:
        # Load a font that supports Vietnamese
        if font_path is None:
            # Tìm font mặc định trong hệ thống Windows
            possible_fonts = [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/calibri.ttf",
                "C:/Windows/Fonts/tahoma.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
                "C:/Windows/Fonts/times.ttf"
            ]
            for path in possible_fonts:
                if os.path.exists(path):
                    font_path = path
                    break
            
            if font_path is None:
                # Nếu không tìm thấy font, sử dụng font mặc định
                font = ImageFont.load_default()
                print("Không tìm thấy font hỗ trợ tiếng Việt, sử dụng font mặc định")
            else:
                font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.truetype(font_path, font_size)

        # Vẽ nền nếu được yêu cầu
        if bg_color is not None:
            # Ước tính kích thước văn bản
            text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
            # Vẽ hình chữ nhật làm nền
            draw.rectangle(
                [position[0], position[1], position[0] + text_width, position[1] + text_height],
                fill=bg_color
            )

        # Draw text
        draw.text(position, text, font=font, fill=color)
    except Exception as e:
        print(f"Lỗi khi vẽ văn bản: {e}")
        # Sử dụng font mặc định nếu có lỗi
        draw.text(position, text, fill=color)

    # Convert back to OpenCV image
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# Get project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# File paths
config_path = os.path.join(project_root, 'config', 'config.yml')
weights_path = os.path.join(project_root, 'model', 'transformerocr.pth')
output_dir = os.path.join(project_root, 'output')
os.makedirs(output_dir, exist_ok=True)

# Cố định tên file đầu ra
FIXED_OUTPUT_IMAGE = os.path.join(output_dir, 'webcam_ocr_result.jpg')
FIXED_OUTPUT_TEXT = os.path.join(output_dir, 'webcam_ocr_result.txt')

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

def detect_lines_paddleocr(image):
    """Detect text lines in an image using PaddleOCR"""
    # Use Vietnamese language model for better detection of Vietnamese text
    ocr = PaddleOCR(use_angle_cls=True, lang='vi', use_gpu=False)
    
    # Create a copy for visualization
    visual_img = image.copy()
    
    # Get detection results - convert to BGR format if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        result = ocr.ocr(image)
    else:
        # Convert to 3 channels if grayscale
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        result = ocr.ocr(image_bgr)
    
    # Extract bounding boxes
    line_boxes = []
    
    if result[0] is not None:
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
                x2 = min(image.shape[1], x2 + padding)
                y2 = min(image.shape[0], y2 + padding)
                
                line_boxes.append((x1, y1, x2, y2))
    
    # Sort by vertical position (top to bottom)
    line_boxes.sort(key=lambda box: box[1])
    
    return line_boxes, image, visual_img

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

def process_frame(frame, predictor, ocr):
    """Process a single frame for text recognition"""
    if frame is None or frame.size == 0:
        return None, []
    
    # Detect lines
    line_boxes, img, visual_img = detect_lines_paddleocr(frame)
    
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
        text = predictor.predict(pil_line)
        results.append((text, (x1, y1, x2, y2)))
        
        # Draw bounding box and line number on visualization image
        cv2.rectangle(visual_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Sử dụng hàm draw_vietnamese_text thay vì cv2.putText
        line_label = f"{i+1}: {text}"
        visual_img = draw_vietnamese_text(
            visual_img,
            line_label,
            (x1, max(0, y1-30)),
            font_size=20,
            color=(0, 0, 255),  # Màu đỏ
            bg_color=(255, 255, 255)  # Nền trắng
        )
    
    return visual_img, results

def list_available_cameras():
    """Liệt kê tất cả các camera có sẵn trên hệ thống"""
    print("\nKiểm tra các camera có sẵn:")
    available_cameras = []
    
    # Kiểm tra 10 camera đầu tiên (0-9)
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Lấy thông tin camera
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"  Camera {i}: Có thể sử dụng ({width}x{height}, {fps:.1f} FPS)")
            available_cameras.append(i)
            # Chụp một khung hình để xác nhận camera hoạt động
            ret, frame = cap.read()
            if ret:
                # Lưu khung hình để kiểm tra
                preview_path = os.path.join(output_dir, f'camera_{i}_preview.jpg')
                cv2.imwrite(preview_path, frame)
                print(f"    Đã lưu ảnh xem trước từ camera {i} tại: {preview_path}")
        else:
            print(f"  Camera {i}: Không tìm thấy hoặc đang bị sử dụng")
        cap.release()
    
    return available_cameras

def main():
    parser = argparse.ArgumentParser(description='Webcam USB OCR')
    parser.add_argument('-c', '--camera', type=int, default=None, 
                        help='Camera index (default: auto-detect webcam USB)')
    parser.add_argument('-s', '--size', type=str, default='1280x720',
                        help='Camera resolution (default: 1280x720)')
    parser.add_argument('-l', '--list', action='store_true',
                        help='List all available cameras and exit')
    args = parser.parse_args()
    
    # Liệt kê camera nếu được yêu cầu
    if args.list:
        list_available_cameras()
        return
    
    # Parse camera resolution
    try:
        width, height = map(int, args.size.split('x'))
    except:
        width, height = 1280, 720
        print(f"Invalid size format. Using default: {width}x{height}")
    
    print("Initializing webcam USB OCR system...")
    config = load_config()
    
    try:
        # Initialize VietOCR
        print("Loading VietOCR model...")
        predictor = Predictor(config)
        
        # Verify model file exists
        if not os.path.exists(weights_path):
            print(f"ERROR: Model file not found at: {weights_path}")
            sys.exit(1)
            
        # Initialize PaddleOCR
        print("Initializing PaddleOCR for text detection...")
        ocr = PaddleOCR(use_angle_cls=True, lang='vi', use_gpu=False)
        
        # Auto-detect USB webcam if camera not specified
        if args.camera is None:
            print("Tự động tìm USB webcam...")
            available_cameras = list_available_cameras()
            
            if not available_cameras:
                print("Không tìm thấy camera nào. Đang thoát...")
                sys.exit(1)
                
            # Ưu tiên camera không phải camera 0 (thường là webcam USB ngoài)
            if len(available_cameras) > 1 and 0 in available_cameras:
                camera_index = [cam for cam in available_cameras if cam != 0][0]
                print(f"Nhiều camera được tìm thấy. Đang sử dụng camera {camera_index} (có vẻ là webcam USB)")
            else:
                camera_index = available_cameras[0]
                print(f"Đang sử dụng camera {camera_index}")
        else:
            camera_index = args.camera
            print(f"Đang sử dụng camera đã chỉ định: {camera_index}")
            
        # Initialize camera
        print(f"Đang mở camera {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not cap.isOpened():
            print(f"Lỗi: Không thể mở camera {camera_index}")
            print("Vui lòng kiểm tra xem webcam USB đã được kết nối đúng cách chưa")
            print("Hãy thử chạy lại với tùy chọn --list để xem các camera có sẵn")
            sys.exit(1)
        
        # Hiển thị thông tin thực tế của camera
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Webcam USB đã sẵn sàng: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
        
        # Variables for state
        processing = False
        last_result_img = None
        last_results = []
        
        print("\nWebcam OCR sẵn sàng!")
        print("Nhấn 'SPACE' để chụp và nhận dạng văn bản")
        print("Nhấn 'q' để thoát")
        print(f"Kết quả sẽ được lưu vào các file cố định:")
        print(f"  - Ảnh: {FIXED_OUTPUT_IMAGE}")
        print(f"  - Văn bản: {FIXED_OUTPUT_TEXT}")
        
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Lỗi: Không thể lấy khung hình từ camera")
                break
                
            # Create display frame
            display = frame.copy()
            
            # Show status on frame using Vietnamese text
            status_text = "TRỰC TIẾP: Nhấn SPACE để nhận dạng văn bản"
            display = draw_vietnamese_text(
                display,
                status_text,
                (10, 30),
                font_size=24,
                color=(0, 0, 255),  # Màu đỏ
                bg_color=(255, 255, 255)  # Nền trắng
            )
            
            # If we have results, show them in a separate window
            if last_result_img is not None:
                cv2.imshow('Kết quả OCR', last_result_img)
                
            # Display the frame
            cv2.imshow('Webcam USB Feed', display)
            
            # Wait for key press
            key = cv2.waitKey(1)
            
            # If 'q' is pressed, break the loop
            if key == ord('q'):
                break
                
            # If space is pressed, process the current frame
            if key == 32:  # Space key
                print("\nĐang xử lý khung hình để nhận dạng văn bản...")
                
                # Process the frame
                result_img, results = process_frame(frame, predictor, ocr)
                
                if result_img is not None:
                    last_result_img = result_img
                    last_results = results
                    
                    # Print recognition results
                    print("\nKết quả nhận dạng:")
                    print("-" * 50)
                    for i, (text, _) in enumerate(results):
                        print(f"Dòng {i+1}: {text}")
                    print("-" * 50)
                    
                    # Lưu kết quả vào các file cố định
                    cv2.imwrite(FIXED_OUTPUT_IMAGE, result_img)
                    
                    with open(FIXED_OUTPUT_TEXT, 'w', encoding='utf-8') as f:
                        for i, (text, _) in enumerate(results):
                            f.write(f"Dòng {i+1}: {text}\n")
                    
                    print(f"Kết quả đã được lưu vào:")
                    print(f"  - Ảnh: {FIXED_OUTPUT_IMAGE}")
                    print(f"  - Văn bản: {FIXED_OUTPUT_TEXT}")
                else:
                    print("Không tìm thấy văn bản hoặc lỗi khi xử lý khung hình")
        
        # Release the camera and close windows
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()