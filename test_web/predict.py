import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

def split_lines(image, padding=30):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hist = np.sum(thresh, axis=1)

    threshold_hist = np.max(hist) * 0.1
    lines = []
    start = None
    for y, value in enumerate(hist):
        if value > threshold_hist and start is None:
            start = y
        elif value <= threshold_hist and start is not None:
            end = y
            if end - start > 10:
                lines.append((start, end))
            start = None

    lines_img = []
    for (startY, endY) in lines:
        startY = max(0, startY - padding)
        endY = min(image.shape[0], endY + padding)
        line_img = image[startY:endY, :]
        lines_img.append(line_img)

    return lines_img

def predict_image(image, config_name='vgg_transformer', weights_path='weights/transformerocr.pth', device='cpu'):
    """
    Nhận diện văn bản từ ảnh và trả về kết quả dưới dạng một chuỗi.
    """
    # Cấu hình vietocr
    config = Cfg.load_config_from_name(config_name)
    config['weights'] = weights_path
    config['device'] = device
    detector = Predictor(config)

    # Tách dòng từ ảnh
    lines = split_lines(image)

    # Nhận diện từng dòng và ghép thành chuỗi
    results = []
    for line_img in lines:
        line_pil = Image.fromarray(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
        text = detector.predict(line_pil)
        results.append(text)

    # Ghép các dòng thành một chuỗi, mỗi dòng cách nhau bởi ký tự xuống dòng
    full_text = '\n'.join(results)
    return full_text

if __name__ == '__main__':
    # Đường dẫn ảnh
    image_path = 'uploads/don_nghi_hoc.png'

    # Đọc ảnh
    img = cv2.imread(image_path)

    # Gọi hàm predict_image
    full_text = predict_image(img)

    # Hiển thị kết quả
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # Bên trái: Ảnh input
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Input Image', fontsize=16)
    ax1.axis('off')

    # Bên phải: Text kết quả
    ax2.axis('off')
    ax2.set_title('Recognized Text', fontsize=16, loc='left')

    ax2.text(0, 1, full_text, fontsize=12, verticalalignment='top')

    plt.tight_layout()
    plt.show()
