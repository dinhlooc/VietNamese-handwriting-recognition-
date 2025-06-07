import os
from flask import Flask, request, send_from_directory, render_template_string
from werkzeug.utils import secure_filename
from PIL import Image
import trich_xuat  
import predict    
from cv2 import imread

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Giao diện HTML đã loại bỏ phần hiển thị văn bản OCR
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Trích xuất thông tin từ ảnh</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 40px; background-color: #f8f9fa; }
        .container { max-width: 900px; margin: auto; }
        img { max-width: 100%; height: auto; border: 1px solid #dee2e6; padding: 5px; border-radius: 5px; }
        .card { margin-top: 20px; }
    </style>
</head>
<body>
<div class="container">
    <h2 class="mb-4">Trích xuất thông tin từ ảnh đơn</h2>

    <form class="mb-4" action="/upload" method="post" enctype="multipart/form-data">
        <div class="input-group">
            <input type="file" name="file" accept="image/*" class="form-control" required>
            <button type="submit" class="btn btn-primary">Tải lên</button>
        </div>
    </form>

    <!-- Nút nhận diện chữ viết tay -->
    <form class="mb-4" id="handwritingForm" action="/handwriting" method="get">
        <button type="submit" class="btn btn-warning">
            <span id="loadingIcon" class="spinner-border spinner-border-sm d-none" role="status" aria-hidden="true"></span>
            Nhận diện chữ viết tay
        </button>
    </form>

    <script>
    // Hiện icon loading khi gửi form handwriting
    document.getElementById('handwritingForm').addEventListener('submit', function() {
        document.getElementById('loadingIcon').classList.remove('d-none');
    });
    </script>
    
    {% if image_url %}
    <div class="row">
        <div class="col-md-12">
            <div class="card shadow-sm">
                <div class="card-header bg-secondary text-white">Ảnh đã tải lên</div>
                <div class="card-body text-center">
                    <img src="{{ image_url }}" alt="Ảnh tải lên">
                </div>
            </div>
        </div>
    </div>

    <div class="card shadow-sm mt-4">
        <div class="card-header bg-success text-white">Thông tin trích xuất</div>
        <div class="card-body">
            <ul class="list-group">
            {% for key, value in extracted_info.items() %}
                <li class="list-group-item">
                    <strong>{{ key }}:</strong> {{ value }}
                </li>
            {% endfor %}
            </ul>
        </div>
    </div>
    {% endif %}
</div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'Không có file được tải lên', 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return 'File không hợp lệ', 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    img = imread(file_path)
    # Xử lý detect với PaddleOCR và lưu ảnh kết quả vào processed_images
    from detect_ocr import detect_lines_paddleocr
    import cv2
    processed_dir = os.path.join(os.path.dirname(__file__), 'processed_images')
    os.makedirs(processed_dir, exist_ok=True)
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='vi')
        line_boxes, img_with_boxes = detect_lines_paddleocr(img, ocr)
        # Vẽ các box lên ảnh gốc
        for (x1, y1, x2, y2) in line_boxes:
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        out_path = os.path.join(processed_dir, f"{os.path.splitext(filename)[0]}_detected.jpg")
        cv2.imwrite(out_path, img_with_boxes)
    except Exception as e:
        print(f"Lỗi khi detect với PaddleOCR: {e}")

    full_text = predict.predict_image(img)
    extracted_info = trich_xuat.extract_info_smart(full_text)

    return render_template_string(HTML_TEMPLATE,
                                  image_url=f"/uploads/{filename}",
                                  extracted_info=extracted_info)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/handwriting')
def handwriting():
    import requests
    api_url = 'http://127.0.0.1:5000/predict'
    try:
        response = requests.get(api_url)
        data = response.json()
        results = data.get('results', [])
        image_base64 = data.get('image_base64', None)
    except Exception as e:
        results = [f'Lỗi khi gọi API: {e}']
        image_base64 = None

    HANDWRITING_TEMPLATE = """
    <!DOCTYPE html>
    <html lang='vi'>
    <head>
        <meta charset='UTF-8'>
        <title>Nhận diện chữ viết tay</title>
        <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css' rel='stylesheet'>
        <style>
            body { padding: 40px; background-color: #f8f9fa; }
            .container { max-width: 900px; margin: auto; }
            img { max-width: 100%; height: auto; border: 1px solid #dee2e6; padding: 5px; border-radius: 5px; }
            .card { margin-top: 20px; }
        </style>
    </head>
    <body>
    <div class='container'>
        <h2 class='mb-4'>Nhận diện chữ viết tay</h2>
        <div class='card shadow-sm'>
            <div class='card-header bg-success text-white'>Kết quả nhận diện</div>
            <div class='card-body'>
                <ul class='list-group mb-3'>
                {% for line in results %}
                    <li class='list-group-item'>{{ line }}</li>
                {% endfor %}
                </ul>
                {% if image_base64 %}
                <div class='text-center'>
                    <img src='data:image/jpeg;base64,{{ image_base64 }}' alt='Ảnh kết quả'>
                </div>
                {% endif %}
            </div>
        </div>
        <a href='/' class='btn btn-secondary mt-4'>Quay lại trang trích xuất thông tin</a>
    </div>
    </body>
    </html>
    """
    return render_template_string(HANDWRITING_TEMPLATE, results=results, image_base64=image_base64)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
