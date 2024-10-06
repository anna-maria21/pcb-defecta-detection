from django.db import models
from django.contrib.auth.models import User


class PcbImage(models.Model):
    photo_name = models.CharField(max_length=255)
    image_hash = models.CharField(max_length=64)
    photo_location = models.CharField(max_length=255)