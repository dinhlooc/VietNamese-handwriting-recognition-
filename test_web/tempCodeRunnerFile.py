
import re
from rapidfuzz import fuzz
import cv2
from predict import predict_image

# Khai báo các key cần trích xuất (chỉ key chuẩn, không cần thêm biến thể lỗi)
target_keys = {
    "Họ và tên": [],
    "Lý do": [],
    "Từ ngày": [],
    "Đến ngày": [],
    "Kính gửi": [],
    "Ngày làm đơn": [],
}

# Hàm tìm các chuỗi có định dạng ngày
def find_date_patterns(text):
    pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(\d{1,2}\s*[-]\s*\d{1,2}\s*[-]\s*\d{2,4})|(\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})'
    return re.findall(pattern, text)

# Lấy chuỗi ngày từ tuple match
def clean_date_tuple(match):
    return [x for x in match if x][0]

# Hàm trích xuất thông tin
def extract_info_smart(text, threshold=75):
    info = {}
    lines = text.split('\n')

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        lower_line = line_clean.lower()

        # Kiểm tra xem dòng có chứa định dạng ngày không
        date_matches = find_date_patterns(line_clean)
        if date_matches:
            date_value = clean_date_tuple(date_matches[0])
            if "đến" in lower_line or "den ngay" in lower_line:
                if "Đến ngày" not in info:
                    info["Đến ngày"] = date_value
            elif "từ" in lower_line or "tu ngay" in lower_line or "nghỉ" in lower_line or "xin nghi" in lower_line:
                if "Từ ngày" not in info:
                    info["Từ ngày"] = date_value
            elif "năm" in lower_line and "ngày" in lower_line:
                if "Ngày làm đơn" not in info:
                    info["Ngày làm đơn"] = date_value

        # So khớp fuzzy với từng key để tìm thông tin
        for key in target_keys:
            score = fuzz.partial_ratio(key.lower(), lower_line)
            if score >= threshold:
                if ':' in line_clean:
                    parts = line_clean.split(':', 1)
                    value = parts[1].strip()
                else:
                    value = line_clean.strip()
                if key not in info and value:
                    info[key] = value
                break

    return info

# Ví dụ sử dụng
if __name__ == "__main__":
    # Đường dẫn ảnh
    image_path = 'uploads/don.png'
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
