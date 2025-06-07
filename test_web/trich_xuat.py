import re
from rapidfuzz import fuzz
import cv2
from predict import predict_image

# Khai báo các key cần trích xuất (chỉ key chuẩn, không cần thêm biến thể lỗi)
target_keys = {
        "Họ và tên sinh viên": [],
    "Số thẻ sinh viên": [],
    "Ngày sinh": [],
    "Lớp": [],
    "Khoa": [],
    "Quê quán": [],
}

# Hàm tìm các chuỗi có định dạng ngày
def find_date_patterns(text):
    pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(\d{1,2}\s*[-]\s*\d{1,2}\s*[-]\s*\d{2,4})|(\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})'
    return re.findall(pattern, text)

# Lấy chuỗi ngày từ tuple match
def clean_date_tuple(match):
    return [x for x in match if x][0]

# Hàm trích xuất thông tin
def extract_info_smart(text, threshold=70):
    info = {}
    lines = text.split('\n')
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        lower_line = line_clean.lower()
        # Họ và tên sinh viên
        if not info.get('Họ và tên sinh viên') and re.search(r'(tên em là|họ và tên sinh viên|họ và tên|em tên là|tên sinh viên)', lower_line):
            match = re.search(r'(?:tên em là|họ và tên sinh viên|họ và tên|em tên là|tên sinh viên)\s*[:]?\s*([A-ZÀ-Ỹa-zà-ỹ\s]+)', line_clean, re.IGNORECASE)
            if not match:
                # Nếu không có dấu :, lấy phần sau từ khóa
                match = re.search(r'(?:tên em là|họ và tên sinh viên|họ và tên|em tên là|tên sinh viên)\s*([A-ZÀ-Ỹa-zà-ỹ\s]+)', line_clean, re.IGNORECASE)
            if match:
                info['Họ và tên sinh viên'] = match.group(1).strip()
        # Số thẻ sinh viên (chấp nhận "thể" và "thẻ", "SV", "sinh viên")
        if not info.get('Số thẻ sinh viên') and re.search(r'(số thẻ|số thể|mssv|msv|số sinh viên|số sv)', lower_line):
            match = re.search(r'(?:số thẻ|số thể|mssv|msv|số sinh viên|số sv)\s*[:]?\s*(\d{6,})', line_clean, re.IGNORECASE)
            if not match:
                # Nếu không có dấu :, lấy số đầu tiên sau từ khóa
                match = re.search(r'(?:số thẻ|số thể|mssv|msv|số sinh viên|số sv)\D*(\d{6,})', line_clean, re.IGNORECASE)
            if match:
                info['Số thẻ sinh viên'] = match.group(1).strip()
        # Ngày sinh
        if not info.get('Ngày sinh') and 'ngày sinh' in lower_line:
            match = re.search(r'ngày sinh\s*[:]?\s*([\d/\-\s]+)', line_clean, re.IGNORECASE)
            if not match:
                match = re.search(r'ngày sinh\D*([\d/\-\s]+)', line_clean, re.IGNORECASE)
            if match:
                info['Ngày sinh'] = match.group(1).strip()
        # Lớp (chấp nhận "lớp", "lộp", "ước lác", "sinh viên lớp")
        if not info.get('Lớp') and re.search(r'(lớp|lộp|ước lác|sinh viên lớp)', lower_line):
            match = re.search(r'(?:lớp|lộp|ước lác|sinh viên lớp)\s*[:]?\s*([A-Za-z0-9\s]+)', line_clean, re.IGNORECASE)
            if not match:
                match = re.search(r'(?:lớp|lộp|ước lác|sinh viên lớp)\D*([A-Za-z0-9\s]+)', line_clean, re.IGNORECASE)
            if match:
                info['Lớp'] = match.group(1).strip()
        # Khoa (chấp nhận "khoa", "khoá", "thuộc khoa", "thuộC KHOA")
        if not info.get('Khoa') and re.search(r'(khoa|khoá|thuộc khoa|thuộc khoa|thuộC KHOA)', lower_line):
            match = re.search(r'(?:khoa|khoá|thuộc khoa|thuộC KHOA)\s*[:.]?\s*([A-Za-z0-9\s.]+)', line_clean, re.IGNORECASE)
            if not match:
                match = re.search(r'(?:khoa|khoá|thuộc khoa|thuộC KHOA)\D*([A-Za-z0-9\s.]+)', line_clean, re.IGNORECASE)
            if match:
                info['Khoa'] = match.group(1).strip()
        # Quê quán
        if not info.get('Quê quán') and re.search(r'(quê quán)', lower_line):
            match = re.search(r'(?:quê quán)\s*[:]?\s*([A-ZÀ-Ỹa-zà-ỹ\s]+)', line_clean, re.IGNORECASE)
            if not match:
                match = re.search(r'(?:quê quán)\D*([A-ZÀ-Ỹa-zà-ỹ\s]+)', line_clean, re.IGNORECASE)
            if match:
                info['Quê quán'] = match.group(1).strip()
    return info

def extract_student_info(text):
    info = {}
    # Họ và tên
    match = re.search(r'(?:Tên|Họ và tên|em tên là|tên em là)\s*[:]?\s*([A-ZÀ-Ỹa-zà-ỹ\s]+)', text, re.IGNORECASE)
    if match:
        info['Họ và tên'] = match.group(1).strip()
    # Số thẻ sinh viên
    match = re.search(r'(?:Số thẻ|Số thẻ SV|MSSV|MSV|Số sinh viên)\s*[:]?\s*(\d{6,})', text, re.IGNORECASE)
    if match:
        info['Số thẻ sinh viên'] = match.group(1).strip()
    # Ngày sinh
    match = re.search(r'Ngày sinh\s*[:]?\s*([\d/\-\s]+)', text, re.IGNORECASE)
    if match:
        info['Ngày sinh'] = match.group(1).strip()
    # Lớp
    match = re.search(r'(?:lớp|sinh viên lớp)\s*[:]?\s*([A-Za-z0-9\s]+)', text, re.IGNORECASE)
    if match:
        info['Lớp'] = match.group(1).strip()
    # Khoa
    match = re.search(r'(?:khoa|thuộc khoa)\s*[:]?\s*([A-Za-z0-9\s]+)', text, re.IGNORECASE)
    if match:
        info['Khoa'] = match.group(1).strip()
    # Quê quán
    match = re.search(r'(?:Quê quán|quê quán)\s*[:]?\s*([A-ZÀ-Ỹa-zà-ỹ\s]+)', text, re.IGNORECASE)
    if match:
        info['Quê quán'] = match.group(1).strip()
    return info

# Ví dụ sử dụng
if __name__ == "__main__":
    # Đường dẫn ảnh
    image_path = 'uploads/don1.png'
    # Đọc ảnh
    img = cv2.imread(image_path)
    # Gọi hàm predict_image
    full_text = predict_image(img)

    print("Full_text")
    print(full_text)
    print("===== Thông tin trích xuất =====")

    info = extract_info_smart(full_text)
    for key, value in info.items():
        print(f"{key}: {value}")

    print("===== Thông tin sinh viên =====")
    student_info = extract_student_info(full_text)
    for key, value in student_info.items():
        print(f"{key}: {value}")

