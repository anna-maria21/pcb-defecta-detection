import cv2
from rest_framework import status
from rest_framework.response import Response

from django.apps import apps
from main.classify import classify
import main.utils as utils


def localize(image_path, localization_model_str, classification_model_str):
    models_dict = apps.get_app_config('main').models_dict
    localization_model = models_dict[localization_model_str]
    classification_model = models_dict[classification_model_str]

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if localization_model == models_dict['1']:
        boxes = utils.detect_with_yolo(localization_model, utils.preprocess_for_yolo(image_rgb))
    else:
        boxes = utils.detect_with_fasterrcnn(localization_model, utils.preprocess_for_fasterrcnn(image_rgb))
    cropped_image = utils.crop_image(image_path, boxes, classification_model)
    return cropped_image
