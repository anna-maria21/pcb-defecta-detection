from django.core.management.base import BaseCommand
from django.db import transaction
from main.models import Defect, Location, PcbImage, ClearConfig
import os
import glob
from datetime import datetime, timedelta
from django.db.models import Max
from django.utils import timezone


def handle():
    last_time = ClearConfig.objects.aggregate(Max('last_removing_date_time'))['last_removing_date_time__max']

    rows_num = 0
    files_num = 0

    if last_time is not None:
        now = timezone.now()
        seven_days_ago = now - timedelta(days=7)
        if last_time > seven_days_ago:
            with transaction.atomic():
                defects_count = Defect.objects.count()
                Defect.objects.all().delete()
                locations_count = Location.objects.count()
                Location.objects.all().delete()
                images_count = PcbImage.objects.count()
                PcbImage.objects.all().delete()
                rows_num = defects_count + locations_count + images_count

            folders = ['../main/uploaded_images/*', '../main/cropped_images/*']  # Adjust this path

            for folder_path in folders:
                files = glob.glob(folder_path)
                for file in files:
                    try:
                        os.remove(file)
                        files_num += 1
                    except Exception as ignored:
                        pass

    ClearConfig.objects.create(
        last_removing_date_time=timezone.now(),
        files_deleted=files_num,
        rows_deleted=rows_num
    )