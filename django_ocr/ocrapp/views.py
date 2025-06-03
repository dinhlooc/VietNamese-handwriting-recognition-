import os
import cv2
import base64
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .models import OcrResult, HandwritingResult
from datetime import datetime
from django.http import JsonResponse
from django.core.files.base import ContentFile
import uuid

# Import các hàm nhận diện và trích xuất từ test_web
import sys
sys.path.append(os.path.abspath(os.path.join(settings.BASE_DIR, '../test_web')))
import predict
import trich_xuat
from detect_ocr import load_config, process_frame_with_visual
from vietocr.tool.predictor import Predictor
from paddleocr import PaddleOCR

def home(request):
    return render(request, 'home.html')

def extract(request):
    form_type = request.GET.get('form_type') or request.POST.get('form_type') or 'chuyencap'
    context = {'form_type': form_type}
    # Lấy lịch sử đúng loại đơn
    if form_type == 'chuyencap':
        history = OcrResult.objects.exclude(image=None).order_by('-created_at')[:10]
        context['history'] = [{
            'created_at': h.created_at,
            'summary': h.text[:40],
            'detail_url': f"/extract/detail/{h.id}/?form_type=chuyencap"
        } for h in history]
    elif form_type == 'chungchi':
        from .models import CertificateApplication
        history = CertificateApplication.objects.exclude(image=None).order_by('-created_at')[:10]
        context['history'] = [{
            'created_at': h.created_at,
            'summary': h.name or h.student_id or 'Chưa có tên',
            'detail_url': f"/extract/detail/certificate/{h.id}/?form_type=chungchi"
        } for h in history]
    elif form_type == 'chuongtrinhhai':
        from .models import SecondProgramApplication
        history = SecondProgramApplication.objects.exclude(image=None).order_by('-created_at')[:10]
        context['history'] = [{
            'created_at': h.created_at,
            'summary': h.name or h.student_id or 'Chưa có tên',
            'detail_url': f"/extract/detail/second/{h.id}/?form_type=chuongtrinhhai"
        } for h in history]
    # Xử lý upload ảnh
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploads'))
        filename = fs.save(file.name, file)
        file_url = fs.url('uploads/' + filename)
        img_path = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)
        if form_type == 'chuyencap':
            img = cv2.imread(img_path)
            weights_path = os.path.join(settings.BASE_DIR, 'ocrapp', 'model', 'transformerocr.pth')
            full_text = predict.predict_image(img, weights_path=weights_path)
            extracted_info = trich_xuat.extract_info_smart(full_text)
            ocr_obj = OcrResult.objects.create(image='uploads/' + filename, text=full_text)
            context.update({'image_url': file_url, 'extracted_info': extracted_info, 'ocr_id': ocr_obj.id})
        elif form_type == 'chungchi':
            import requests
            with open(img_path, 'rb') as f:
                files = {'image': (filename, f, 'image/jpeg')}
                api_url = 'https://chithanh223.loca.lt/ocr_chungchi'
                try:
                    response = requests.post(api_url, files=files, timeout=30)
                    data = response.json()
                    from .models import CertificateApplication
                    cert_obj = CertificateApplication.objects.create(
                        image='uploads/' + filename,
                        name=data.get('name'),
                        dob=data.get('dob'),
                        student_class=data.get('class'),
                        student_id=data.get('student_id'),
                        major=data.get('major'),
                        certificate=data.get('certificate'),
                        reason=data.get('reason'),
                        attachment=data.get('attachment'),
                        application_date=data.get('application_date')
                    )
                    context.update({'image_url': file_url, 'extracted_info': data, 'ocr_id': cert_obj.id})
                except Exception as e:
                    context['error'] = f'Lỗi khi gọi API chứng chỉ: {e}'
        elif form_type == 'chuongtrinhhai':
            import requests
            with open(img_path, 'rb') as f:
                files = {'image': (filename, f, 'image/jpeg')}
                api_url = 'https://chithanh223.loca.lt/ocr_chuongtrinh'
                try:
                    response = requests.post(api_url, files=files, timeout=30)
                    data = response.json()
                    from .models import SecondProgramApplication
                    prog_obj = SecondProgramApplication.objects.create(
                        image='uploads/' + filename,
                        name=data.get('name'),
                        student_id=data.get('student_id'),
                        dob=data.get('dob'),
                        student_class=data.get('class'),
                        major=data.get('major'),
                        school_from=data.get('school_from'),
                        semester=data.get('semester'),
                        gpa=data.get('gpa'),
                        conduct_score=data.get('conduct_score'),
                        application_date=data.get('application_date')
                    )
                    context.update({'image_url': file_url, 'extracted_info': data, 'ocr_id': prog_obj.id})
                except Exception as e:
                    context['error'] = f'Lỗi khi gọi API chương trình hai: {e}'
    # Xử lý nút trích xuất từ camera
    if request.method == 'POST' and request.POST.get('from_camera') == '1':
        import requests
        api_url = 'http://192.168.129.88:5000/images'
        try:
            response = requests.get(api_url)
            data = response.json()
            image_base64 = data.get('image_base64', None)
            if image_base64:
                img_data = base64.b64decode(image_base64)
                file_name = f"camera_{uuid.uuid4().hex[:8]}.jpg"
                image_file = ContentFile(img_data, name=file_name)
                # Lưu file ảnh vào uploads
                fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploads'))
                filename = fs.save(file_name, image_file)
                file_url = fs.url('uploads/' + filename)
                img_path = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)
                # Xử lý theo loại đơn
                if form_type == 'chuyencap':
                    img = cv2.imread(img_path)
                    weights_path = os.path.join(settings.BASE_DIR, 'ocrapp', 'model', 'transformerocr.pth')
                    full_text = predict.predict_image(img, weights_path=weights_path)
                    extracted_info = trich_xuat.extract_info_smart(full_text)
                    ocr_obj = OcrResult.objects.create(image='uploads/' + filename, text=full_text)
                    context.update({'image_url': file_url, 'extracted_info': extracted_info, 'ocr_id': ocr_obj.id})
                elif form_type == 'chungchi':
                    with open(img_path, 'rb') as f:
                        files = {'image': (filename, f, 'image/jpeg')}
                        api_url = 'https://chithanh223.loca.lt/ocr_chungchi'
                        try:
                            response = requests.post(api_url, files=files, timeout=30)
                            data = response.json()
                            from .models import CertificateApplication
                            cert_obj = CertificateApplication.objects.create(
                                image='uploads/' + filename,
                                name=data.get('name'),
                                dob=data.get('dob'),
                                student_class=data.get('class'),
                                student_id=data.get('student_id'),
                                major=data.get('major'),
                                certificate=data.get('certificate'),
                                reason=data.get('reason'),
                                attachment=data.get('attachment'),
                                application_date=data.get('application_date')
                            )
                            context.update({'image_url': file_url, 'extracted_info': data, 'ocr_id': cert_obj.id})
                        except Exception as e:
                            context['error'] = f'Lỗi khi gọi API chứng chỉ: {e}'
                elif form_type == 'chuongtrinhhai':
                    with open(img_path, 'rb') as f:
                        files = {'image': (filename, f, 'image/jpeg')}
                        api_url = 'https://chithanh223.loca.lt/ocr_chuongtrinh'
                        try:
                            response = requests.post(api_url, files=files, timeout=30)
                            data = response.json()
                            from .models import SecondProgramApplication
                            prog_obj = SecondProgramApplication.objects.create(
                                image='uploads/' + filename,
                                name=data.get('name'),
                                student_id=data.get('student_id'),
                                dob=data.get('dob'),
                                student_class=data.get('class'),
                                major=data.get('major'),
                                school_from=data.get('school_from'),
                                semester=data.get('semester'),
                                gpa=data.get('gpa'),
                                conduct_score=data.get('conduct_score'),
                                application_date=data.get('application_date')
                            )
                            context.update({'image_url': file_url, 'extracted_info': data, 'ocr_id': prog_obj.id})
                        except Exception as e:
                            context['error'] = f'Lỗi khi gọi API chương trình hai: {e}'
        except Exception as e:
            context['error'] = f'Lỗi khi lấy ảnh từ camera: {e}'
    return render(request, 'extract.html', context)

def extract_detail(request, ocr_id):
    ocr_obj = get_object_or_404(OcrResult, pk=ocr_id)
    extracted_info = trich_xuat.extract_info_smart(ocr_obj.text)
    image_url = None
    if ocr_obj.image:
        image_url = settings.MEDIA_URL + str(ocr_obj.image)
    context = {
        'image_url': image_url,
        'extracted_info': extracted_info,
        'ocr_obj': ocr_obj
    }
    return render(request, 'extract_detail.html', context)

def extract_detail_certificate(request, ocr_id):
    from .models import CertificateApplication
    obj = get_object_or_404(CertificateApplication, pk=ocr_id)
    image_url = obj.image.url if obj.image else None
    extracted_info = {
        'name': obj.name,
        'dob': obj.dob,
        'class': obj.student_class,
        'student_id': obj.student_id,
        'major': obj.major,
        'certificate': obj.certificate,
        'reason': obj.reason,
        'attachment': obj.attachment,
        'application_date': obj.application_date
    }
    return render(request, 'extract_detail.html', {
        'image_url': image_url,
        'extracted_info': extracted_info,
        'ocr_obj': obj
    })

def extract_detail_second(request, ocr_id):
    from .models import SecondProgramApplication
    obj = get_object_or_404(SecondProgramApplication, pk=ocr_id)
    image_url = obj.image.url if obj.image else None
    extracted_info = {
        'name': obj.name,
        'student_id': obj.student_id,
        'dob': obj.dob,
        'class': obj.student_class,
        'major': obj.major,
        'school_from': obj.school_from,
        'semester': obj.semester,
        'gpa': obj.gpa,
        'conduct_score': obj.conduct_score,
        'application_date': obj.application_date
    }
    return render(request, 'extract_detail.html', {
        'image_url': image_url,
        'extracted_info': extracted_info,
        'ocr_obj': obj
    })

def handwriting(request):
    import requests
    history = HandwritingResult.objects.order_by('-created_at')[:10]
    results = []
    image_base64 = None
    image_file = None
    if request.method == 'GET' and request.GET.get('run') == '1':
        api_url = 'http://192.168.129.88:5000/predict'
        try:
            response = requests.get(api_url)
            data = response.json()
            results = data.get('results', [])
            image_base64 = data.get('image_base64', None)
            # Lưu ảnh base64 thành file nếu có
            if image_base64:
                import base64
                from django.core.files.base import ContentFile
                import uuid
                img_data = base64.b64decode(image_base64)
                file_name = f"handwriting_{uuid.uuid4().hex[:8]}.jpg"
                image_file = ContentFile(img_data, name=file_name)
            # Lưu vào HandwritingResult
            if results or image_file:
                HandwritingResult.objects.create(
                    text='\n'.join(results),
                    image=image_file
                )
        except Exception as e:
            results = [f'Lỗi khi gọi API: {e}']
            image_base64 = None
    context = {'results': results, 'image_base64': image_base64, 'history': history}
    return render(request, 'handwriting.html', context)

def handwriting_detail(request, pk):
    from django.shortcuts import get_object_or_404
    obj = get_object_or_404(HandwritingResult, pk=pk)
    image_url = obj.image.url if obj.image else None
    results = obj.text.split('\n') if obj.text else []
    return render(request, 'handwriting_detail.html', {
        'results': results,
        'image_url': image_url,
        'result_obj': obj
    })
