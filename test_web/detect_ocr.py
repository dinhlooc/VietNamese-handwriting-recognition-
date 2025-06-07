import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from paddleocr import PaddleOCR
import os
import yaml

# Đường dẫn model/config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config_path = os.path.join(project_root, 'config', 'config.yml')
weights_path = os.path.join(project_root, 'model', 'transformerocr.pth')

# Hàm load config VietOCR
def load_config():
    # Đảm bảo lấy đúng đường dẫn config.yml trong test_web/config/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    config_path = os.path.join(project_root, 'config', 'config.yml')
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    config['device'] = 'cpu'
    # Đảm bảo lấy đúng đường dẫn weights trong test_web/weights/
    config['weights'] = os.path.join(project_root, 'weights', 'transformerocr.pth')
    return config

def enhance_line_image(line_img):
    if len(line_img.shape) == 3:
        gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = line_img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return rgb

def detect_lines_paddleocr(image, ocr):
    if len(image.shape) == 3 and image.shape[2] == 3:
        result = ocr.ocr(image)
    else:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        result = ocr.ocr(image_bgr)
    line_boxes = []
    if result[0] is not None:
        for idx, line in enumerate(result[0]):
            if line:
                points = line[0]
                x1 = min(int(points[0][0]), int(points[3][0]))
                y1 = min(int(points[0][1]), int(points[1][1]))
                x2 = max(int(points[1][0]), int(points[2][0]))
                y2 = max(int(points[2][1]), int(points[3][1]))
                padding = 5
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.shape[1], x2 + padding)
                y2 = min(image.shape[0], y2 + padding)
                line_boxes.append((x1, y1, x2, y2))
    line_boxes.sort(key=lambda box: box[1])
    return line_boxes, image

def process_frame(image, predictor, ocr):
    if image is None or image.size == 0:
        return []
    line_boxes, img = detect_lines_paddleocr(image, ocr)
    results = []
    for i, (x1, y1, x2, y2) in enumerate(line_boxes):
        line_img = img[y1:y2, x1:x2]
        if line_img.size == 0 or line_img.shape[0] == 0 or line_img.shape[1] == 0:
            continue
        enhanced_line = enhance_line_image(line_img)
        pil_line = Image.fromarray(enhanced_line)
        text = predictor.predict(pil_line)
        results.append(text)
    return results

def draw_vietnamese_text(image, text, position, font_path=None, font_size=24, color=(255, 0, 0), bg_color=None):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    try:
        if font_path is None:
            possible_fonts = [
                os.path.join(project_root, 'font', 'ARIAL.TTF'),
                os.path.join(project_root, 'font', 'CALIBRIB.TTF'),
                os.path.join(project_root, 'font', 'TAHOMA.TTF'),
                os.path.join(project_root, 'font', 'TIMESBD.TTF'),
            ]
            for path in possible_fonts:
                if os.path.exists(path):
                    font_path = path
                    break
            if font_path is None:
                font = ImageFont.load_default()
            else:
                font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.truetype(font_path, font_size)
        if bg_color is not None:
            text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
            draw.rectangle(
                [position[0], position[1], position[0] + text_width, position[1] + text_height],
                fill=bg_color
            )
        draw.text(position, text, font=font, fill=color)
    except Exception as e:
        draw.text(position, text, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def process_frame_with_visual(image, predictor, ocr):
    if image is None or image.size == 0:
        return None, []
    line_boxes, img = detect_lines_paddleocr(image, ocr)
    results = []
    visual_img = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(line_boxes):
        line_img = img[y1:y2, x1:x2]
        if line_img.size == 0 or line_img.shape[0] == 0 or line_img.shape[1] == 0:
            continue
        enhanced_line = enhance_line_image(line_img)
        pil_line = Image.fromarray(enhanced_line)
        text = predictor.predict(pil_line)
        results.append((text, (x1, y1, x2, y2)))
        cv2.rectangle(visual_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        line_label = f"{i+1}: {text}"
        visual_img = draw_vietnamese_text(
            visual_img,
            line_label,
            (x1, max(0, y1-30)),
            font_size=20,
            color=(0, 0, 255),
            bg_color=(255, 255, 255)
        )
    return visual_img, results
