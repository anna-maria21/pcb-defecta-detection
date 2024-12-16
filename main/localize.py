from datetime import datetime

import cv2

from django.apps import apps
import main.utils as utils
from main.models import ModelPerformance


def localize(image_path, localization_model_str, classification_model_str, pcb_image, user_id):
    models_dict = apps.get_app_config('main').models_dict
    localization_model = models_dict[localization_model_str]
    classification_model = models_dict[classification_model_str]

    start_localize_time = datetime.now()
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if localization_model == models_dict['1']:
        prepared_image = utils.adjust_brightness(image_rgb)
        boxes = utils.detect_with_yolo(localization_model, utils.preprocess_for_yolo(prepared_image))
    else:
        boxes = utils.detect_with_fasterrcnn(localization_model, utils.preprocess_for_fasterrcnn(image_rgb))

    end_localize_time = datetime.now()
    localize_execution_time = end_localize_time - start_localize_time
    localize_execution_time_ms = (
            localize_execution_time.days * 24 * 60 * 60 * 1000 +  # Days to ms
            localize_execution_time.seconds * 1000 +  # Seconds to ms
            localize_execution_time.microseconds / 1000  # Microseconds to ms
    )

    start_classify_time = datetime.now()
    cropped_image = utils.crop_image(image_path, boxes, pcb_image, localization_model, classification_model, user_id)
    end_classify_time = datetime.now()
    classify_execution_time = end_classify_time - start_classify_time
    classify_execution_time_ms = (
            classify_execution_time.days * 24 * 60 * 60 * 1000 +
            classify_execution_time.seconds * 1000 +
            classify_execution_time.microseconds / 1000
    )
    ModelPerformance.objects.create(
        localization_model_id=localization_model_str,
        classification_model_id=classification_model_str,
        localization_time=localize_execution_time_ms,
        classification_time=classify_execution_time_ms
    )

    return cropped_image
