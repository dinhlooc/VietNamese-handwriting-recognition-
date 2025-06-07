import re

def extract_information(text):
    result = {}

    # Gửi tới trường nào
    to_school = re.search(r'Kính gửi\s*:?\s*-?\s*(?:Ban giám hiệu|BGH)[^\n]*Trường\s*(.*?)\s*\n', text)
    if to_school:
        result['truong_gui_den'] = to_school.group(1).strip()

    # Giáo viên chủ nhiệm
    gvcn = re.search(r'Giáo viên chủ nhiệm lớp\s*[:\-]?\s*(\S+)', text)
    if gvcn:
        result['lop'] = gvcn.group(1).strip()

    # Họ tên phụ huynh
    ten_ph = re.search(r'Tôi tên\s*[:\-]?\s*(.*?)\s*-\s*SDT', text)
    if ten_ph:
        result['phu_huynh'] = ten_ph.group(1).strip()

    # Số điện thoại
    sdt = re.search(r'SDT\s*\(?\s*[:\-]?\s*(\(?\d{3,4}\)?[\s.-]?\d{3}[\s.-]?\d{3,4})', text)
    if sdt:
        result['sdt'] = sdt.group(1).strip()

    # Địa chỉ
    diachi = re.search(r'Địa chỉ\s*[:\-]?\s*(.*?)\s*\n', text)
    if diachi:
        result['dia_chi'] = diachi.group(1).strip()

    # Học sinh
    hs = re.search(r'là phụ huynh của em[:\-]?\s*(.*?)\s*\n', text)
    if hs:
        result['hoc_sinh'] = hs.group(1).strip()

    # Lớp học sinh
    lop_hs = re.search(r'hiện là học sinh lớp[:\-]?\s*(.*?)\s*trường', text)
    if lop_hs:
        result['lop_hoc_sinh'] = lop_hs.group(1).strip()

    # Tên trường học sinh
    truong_hs = re.search(r'trường\s*(.*?)\s*\n', text)
    if truong_hs:
        result['truong_hoc'] = truong_hs.group(1).strip()

    # Số ngày nghỉ + ngày bắt đầu / kết thúc
    so_ngay = re.search(r'trong\s*\(?\s*(\d+)\s*\)?\s*ngày', text)
    if so_ngay:
        result['so_ngay_nghi'] = so_ngay.group(1)

    tu_ngay = re.search(r'ngày thứ.*?ngày\s*(\d{1,2})[./](\d{1,2})', text)
    if tu_ngay:
        result['ngay_bat_dau'] = f"{tu_ngay.group(1).zfill(2)}/{tu_ngay.group(2).zfill(2)}"

    den_ngay = re.search(r'đến ngày\s*(\d{1,2})[./](\d{1,2})', text)
    if den_ngay:
        result['ngay_ket_thuc'] = f"{den_ngay.group(1).zfill(2)}/{den_ngay.group(2).zfill(2)}"

    # Lý do
    lydo = re.search(r'Lý do[:\-]?\s*(.*?)\n', text)
    if lydo:
        result['ly_do'] = lydo.group(1).strip()

    # Ngày viết đơn
    ngay_viet = re.search(r'TP\.HCM.*?ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})', text)
    if ngay_viet:
        result['ngay_viet_don'] = f"{ngay_viet.group(1).zfill(2)}/{ngay_viet.group(2).zfill(2)}/{ngay_viet.group(3)}"

    result['toan_bo_van_ban'] = text  # lưu lại để debug nếu cần
    return result
