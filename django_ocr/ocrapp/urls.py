from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from ocrapp import views

urlpatterns = [
    path('', views.home, name='home'),
    path('extract/', views.extract, name='extract'),
    path('extract/detail/<int:ocr_id>/', views.extract_detail, name='extract_detail'),
    path('extract/detail/certificate/<int:ocr_id>/', views.extract_detail_certificate, name='extract_detail_certificate'),
    path('extract/detail/second/<int:ocr_id>/', views.extract_detail_second, name='extract_detail_second'),
    path('handwriting/', views.handwriting, name='handwriting'),
    path('handwriting/<int:pk>/', views.handwriting_detail, name='handwriting_detail'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
