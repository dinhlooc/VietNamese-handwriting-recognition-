import yaml
import os
import torch
import time
import cv2
import numpy as np
import argparse
from PIL import Image
from vietocr.tool.predictor import Predictor
import sys
from paddleocr import PaddleOCR
from PIL import ImageFont, ImageDraw
import glob

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

def correct_skew(image, angle_threshold=1.0):
    # Chuyển ảnh sang xám và làm mịn
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Phát hiện cạnh
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Tìm các đường thẳng
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        print("Không tìm thấy đường thẳng nào trong ảnh.")
        return image  # Trả về ảnh gốc nếu không có đường thẳng

    # Tính góc xoay trung bình
    angles = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            angles.append(angle)
    
    median_angle = np.median(angles)
    
    # Nếu góc xoay nhỏ hơn ngưỡng, trả về ảnh gốc
    if abs(median_angle) < angle_threshold:
        print(f"Góc nghiêng nhỏ ({median_angle:.2f}°), không cần xoay.")
        return image

    # Xoay ảnh để chỉnh thẳng
    print(f"Xoay ảnh với góc {median_angle:.2f}°")

    height, width = image.shape[:2]
    M = cv2.getRotationMatrix2D((width / 2, height / 2), median_angle, 1)

    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)

    M[0, 2] += (new_width / 2) - width / 2
    M[1, 2] += (new_height / 2) - height / 2

    rotated_image = cv2.warpAffine(image, M, (new_width, new_height))
    return rotated_image

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

def process_image(image_path, predictor, ocr):
    """Process an image file for text recognition"""
    print(f"\nXử lý ảnh: {image_path}")
    
    # Đọc ảnh
    frame = cv2.imread(image_path)
    
    if frame is None or frame.size == 0:
        print(f"Lỗi: Không thể đọc ảnh từ {image_path}")
        return None, []
    
    # Lấy tên file không có phần mở rộng
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Áp dụng correct_skew để chỉnh ảnh về đúng khung ngang trước khi detect
    print("Đang điều chỉnh khung hình để chuẩn hóa góc nghiêng...")
    corrected_frame = correct_skew(frame)
    
    # Detect lines
    line_boxes, img, visual_img = detect_lines_paddleocr(corrected_frame)
    
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
    
    # Tạo đường dẫn đầu ra cụ thể cho từng ảnh
    output_image_path = os.path.join(output_dir, f"{file_name}_ocr_result.jpg")
    output_text_path = os.path.join(output_dir, f"{file_name}_ocr_result.txt")
    
    # Lưu kết quả ảnh
    cv2.imwrite(output_image_path, visual_img)
    
    # Lưu kết quả text
    with open(output_text_path, 'w', encoding='utf-8') as f:
        f.write(f"Kết quả OCR cho ảnh: {image_path}\n")
        f.write("-" * 50 + "\n")
        for i, (text, _) in enumerate(results):
            f.write(f"Dòng {i+1}: {text}\n")
    
    print(f"Đã lưu kết quả:")
    print(f"  - Ảnh: {output_image_path}")
    print(f"  - Văn bản: {output_text_path}")
    
    return visual_img, results

def main():
    parser = argparse.ArgumentParser(description='OCR từ thư mục ảnh')
    parser.add_argument('-i', '--input', type=str, default=None, 
                        help='Đường dẫn đến ảnh hoặc thư mục chứa ảnh')
    parser.add_argument('-e', '--ext', type=str, default='jpg,jpeg,png', 
                        help='Định dạng ảnh cần xử lý (ngăn cách bởi dấu phẩy, mặc định: jpg,jpeg,png)')
    args = parser.parse_args()
    
    # Xác định đường dẫn input mặc định nếu không được cung cấp
    if args.input is None:
        # Sử dụng thư mục image trong project
        input_path = os.path.join(project_root, 'image')
        if not os.path.exists(input_path):
            print(f"Thư mục mặc định {input_path} không tồn tại.")
            # Thử sử dụng thư mục samples
            samples_path = os.path.join(project_root, '..', 'PBL5_final', 'src', 'samples')
            if os.path.exists(samples_path):
                input_path = samples_path
                print(f"Sử dụng thư mục samples: {input_path}")
            else:
                print("Không tìm thấy thư mục ảnh mặc định.")
                print("Vui lòng chỉ định đường dẫn đến thư mục ảnh bằng tham số -i.")
                sys.exit(1)
    else:
        input_path = args.input
    
    # Chuyển đổi list phần mở rộng
    extensions = args.ext.lower().split(',')
    
    print("Khởi tạo hệ thống OCR...")
    config = load_config()
    
    try:
        # Initialize VietOCR
        print("Đang tải mô hình VietOCR...")
        predictor = Predictor(config)
        
        # Kiểm tra file model tồn tại
        if not os.path.exists(weights_path):
            print(f"LỖI: Không tìm thấy file model tại: {weights_path}")
            sys.exit(1)
            
        # Initialize PaddleOCR
        print("Khởi tạo PaddleOCR cho việc phát hiện văn bản...")
        ocr = PaddleOCR(use_angle_cls=True, lang='vi', use_gpu=False)
        
        # Thu thập danh sách ảnh cần xử lý
        image_files = []
        
        if os.path.isfile(input_path):
            # Nếu là file, kiểm tra phần mở rộng
            ext = os.path.splitext(input_path)[1].lower().lstrip('.')
            if ext in extensions:
                image_files.append(input_path)
            else:
                print(f"Định dạng file {ext} không được hỗ trợ.")
                sys.exit(1)
        else:
            # Nếu là thư mục, thu thập tất cả file ảnh
            for ext in extensions:
                pattern = os.path.join(input_path, f"*.{ext}")
                image_files.extend(glob.glob(pattern))
        
        if not image_files:
            print(f"Không tìm thấy ảnh nào trong {input_path} với định dạng: {args.ext}")
            sys.exit(1)
        
        print(f"\nĐã tìm thấy {len(image_files)} ảnh để xử lý.")
        
        # Xử lý từng ảnh
        all_results = []
        for img_path in image_files:
            result_img, results = process_image(img_path, predictor, ocr)
            
            if result_img is not None:
                # Hiển thị kết quả
                print(f"\nKết quả nhận dạng cho {os.path.basename(img_path)}:")
                print("-" * 50)
                for i, (text, _) in enumerate(results):
                    print(f"Dòng {i+1}: {text}")
                print("-" * 50)
                
                # Hiển thị ảnh kết quả (có thể bỏ nếu không muốn)
                cv2.imshow(f'Kết quả OCR: {os.path.basename(img_path)}', result_img)
                cv2.waitKey(0)
                
                all_results.append((img_path, results))
            else:
                print(f"Không tìm thấy văn bản hoặc có lỗi khi xử lý ảnh {img_path}")
        
        # Tạo báo cáo tổng hợp
        summary_path = os.path.join(output_dir, "image_ocr_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"BÁO CÁO TỔNG HỢP OCR\n")
            f.write(f"Ngày: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 80 + "\n\n")
            
            for img_path, results in all_results:
                f.write(f"Ảnh: {img_path}\n")
                f.write("-" * 50 + "\n")
                for i, (text, _) in enumerate(results):
                    f.write(f"Dòng {i+1}: {text}\n")
                f.write("\n" + "-" * 50 + "\n\n")
        
        print(f"\nĐã hoàn thành xử lý {len(image_files)} ảnh.")
        print(f"Báo cáo tổng hợp đã được lưu tại: {summary_path}")
        
        # Đóng các cửa sổ hiển thị
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()