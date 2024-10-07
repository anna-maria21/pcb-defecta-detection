from django.db import models


class PcbImage(models.Model):
    image_hash = models.CharField(max_length=64)
    photo_location = models.CharField(max_length=255)

    class Meta:
        db_table = 'pcb_image'
