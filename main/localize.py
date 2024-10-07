import cv2
from rest_framework import status
from rest_framework.response import Response
from ultralytics import YOLO

from main.classify import classify


def localize(image, localization_model_str, classification_model_str):
    yolo_model = YOLO('D:/магістерська/pcb_defects_detection/main/ai-models/my-model.pt')
    vgg16_model = ''
    models_dict = {'1': yolo_model, '3': vgg16_model}
    localization_model = models_dict[localization_model_str]
    classification_model = models_dict[classification_model_str]

    result = localization_model(image)

    cropped_image = crop_image(image, result)


def crop_image(image, results):
    img = cv2.imread(image)

    detections = results[0].boxes

    if detections:
        for box in detections:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            cropped_img = img[y_min:y_max, x_min:x_max]

            cropped_img_path = 'D:/магістерська/pcb_defects_detection/main/cropped_images/' + image.split('/')[-1]
            cv2.imwrite(cropped_img_path, cropped_img)
            classification_result = classify(cropped_img_path)

    return classification_result
