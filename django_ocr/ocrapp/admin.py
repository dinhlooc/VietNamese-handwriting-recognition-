from django.contrib import admin
from .models import OcrResult

@admin.register(OcrResult)
class OcrResultAdmin(admin.ModelAdmin):
    list_display = ('id', 'created_at', 'image', 'text')
    search_fields = ('text',)
    list_filter = ('created_at',)

# Register your models here.
