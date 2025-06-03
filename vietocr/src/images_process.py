import cv2
import numpy as np
from PIL import Image, ImageEnhance

class ImageProcessor:
    def __init__(self):
        """Khởi tạo bộ xử lý hình ảnh"""
        pass
        
    def enhance_for_ocr(self, image):
        """
        Cải thiện ảnh để nhận dạng OCR tốt hơn
        
        Args:
            image: ảnh đầu vào (định dạng BGR từ OpenCV hoặc numpy array)
            
        Returns:
            ảnh đã được cải thiện
        """
        # Chuyển về grayscale nếu là ảnh màu
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Chuyển về định dạng RGB để sử dụng với VietOCR
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return rgb
    
    def enhance_with_pil(self, image):
        """
        Cải thiện ảnh sử dụng PIL thay vì OpenCV
        
        Args:
            image: ảnh đầu vào (định dạng BGR từ OpenCV)
            
        Returns:
            ảnh đã được cải thiện (định dạng PIL Image)
        """
        # Chuyển sang định dạng PIL Image
        if isinstance(image, np.ndarray):
            # Chuyển từ BGR (OpenCV) sang RGB (PIL)
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image
        
        # Tăng cường độ tương phản
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.5)
        
        # Tăng cường độ sắc nét
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.5)
        
        return pil_image
    
    def binarize(self, image, threshold=128, max_value=255, method=cv2.THRESH_BINARY | cv2.THRESH_OTSU):
        """
        Nhị phân hóa ảnh cho OCR
        
        Args:
            image: ảnh đầu vào (định dạng BGR từ OpenCV)
            threshold: ngưỡng nhị phân hóa
            max_value: giá trị tối đa sau nhị phân hóa
            method: phương pháp nhị phân hóa
            
        Returns:
            ảnh đã được nhị phân hóa
        """
        # Chuyển về grayscale nếu là ảnh màu
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Áp dụng nhị phân hóa
        _, binary = cv2.threshold(gray, threshold, max_value, method)
        
        return binary
    
    def deskew(self, image, angle_threshold=1.0):
        """
        Chỉnh sửa góc nghiêng của ảnh để đưa văn bản về đúng hướng ngang
        
        Args:
            image: ảnh đầu vào (định dạng BGR từ OpenCV)
            angle_threshold: ngưỡng góc (độ) để quyết định có xoay ảnh hay không
            
        Returns:
            ảnh đã được chỉnh sửa góc nghiêng
        """
        # Chuyển ảnh sang xám và làm mịn
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
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
    
    def denoise(self, image):
        """
        Khử nhiễu ảnh
        
        Args:
            image: ảnh đầu vào (định dạng BGR từ OpenCV)
            
        Returns:
            ảnh đã được khử nhiễu
        """
        # Chuyển về grayscale nếu là ảnh màu
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Áp dụng phương pháp khử nhiễu Gaussian
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return denoised
    
    def resize_with_aspect_ratio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        """
        Thay đổi kích thước ảnh giữ nguyên tỷ lệ
        
        Args:
            image: ảnh đầu vào (định dạng BGR từ OpenCV)
            width: chiều rộng mới (nếu None, sẽ tính dựa trên height)
            height: chiều cao mới (nếu None, sẽ tính dựa trên width)
            inter: phương pháp nội suy
            
        Returns:
            ảnh đã thay đổi kích thước
        """
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        resized = cv2.resize(image, dim, interpolation=inter)
        return resized
        
    def correct_brightness(self, image, alpha=1.0, beta=0):
        """
        Điều chỉnh độ sáng và độ tương phản
        
        Args:
            image: ảnh đầu vào (định dạng BGR từ OpenCV)
            alpha: hệ số tương phản
            beta: hệ số độ sáng
            
        Returns:
            ảnh đã điều chỉnh
        """
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted