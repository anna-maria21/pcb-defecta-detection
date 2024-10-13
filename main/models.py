from django.db import models


class PcbImage(models.Model):
    image_hash = models.CharField(max_length=64)
    photo_location = models.CharField(max_length=255)

    class Meta:
        db_table = 'pcb_image'


class Location(models.Model):
    x_min = models.IntegerField()
    y_min = models.IntegerField()
    x_max = models.IntegerField()
    y_max = models.IntegerField()
    image_id = models.IntegerField()

    class Meta:
        db_table = 'location'


class ModelsRating(models.Model):
    classification_model_id = models.IntegerField()
    localization_model_id = models.IntegerField()
    rating = models.DecimalField(max_digits=3, decimal_places=2)

    class Meta:
        db_table = 'models_rating'


class Defect(models.Model):
    image_id = models.IntegerField()
    user_id = models.IntegerField()
    type_id = models.IntegerField()
    localization_model_id = models.IntegerField()
    classification_model_id = models.IntegerField()
    location_id = models.IntegerField()

    class Meta:
        db_table = 'defect'


class ClearConfig(models.Model):
    last_removing_date_time = models.DateTimeField()
    files_deleted = models.IntegerField()
    rows_deleted = models.IntegerField()

    class Meta:
        db_table = 'clear_config'