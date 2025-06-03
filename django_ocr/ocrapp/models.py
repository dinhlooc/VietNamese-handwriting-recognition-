from django.db import models

class OcrResult(models.Model):
    image = models.ImageField(upload_to='uploads/')
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"OCR at {self.created_at}"

class HandwritingResult(models.Model):
    image = models.ImageField(upload_to='uploads/', null=True, blank=True)
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Handwriting at {self.created_at}"

class CertificateApplication(models.Model):
    image = models.ImageField(upload_to='uploads/', null=True, blank=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    dob = models.CharField(max_length=100, blank=True, null=True)
    student_class = models.CharField(max_length=100, blank=True, null=True)
    student_id = models.CharField(max_length=100, blank=True, null=True)
    major = models.CharField(max_length=255, blank=True, null=True)
    certificate = models.CharField(max_length=255, blank=True, null=True)
    reason = models.TextField(blank=True, null=True)
    attachment = models.TextField(blank=True, null=True)
    application_date = models.CharField(max_length=100, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

class SecondProgramApplication(models.Model):
    image = models.ImageField(upload_to='uploads/', null=True, blank=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    student_id = models.CharField(max_length=100, blank=True, null=True)
    dob = models.CharField(max_length=100, blank=True, null=True)
    student_class = models.CharField(max_length=100, blank=True, null=True)
    major = models.CharField(max_length=255, blank=True, null=True)
    school_from = models.CharField(max_length=255, blank=True, null=True)
    semester = models.CharField(max_length=100, blank=True, null=True)
    gpa = models.CharField(max_length=100, blank=True, null=True)
    conduct_score = models.CharField(max_length=100, blank=True, null=True)
    application_date = models.CharField(max_length=100, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
# Create your models here.
