import base64
import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch
import torchvision
import torchvision.transforms as T
from django.apps import apps
from tensorflow.keras.models import load_model
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO

from main.classify import classify
from main.models import Defect
from main.models import Location


def preprocess_for_yolo(image):
    return image


def preprocess_for_fasterrcnn(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)


def crop_image(image, boxes, pcb_image, localization_model, classification_model, userid):
    img = cv2.imread(image)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    box_num = 1
    classes = list()
    boxes = non_maximum_suppression(boxes, 0.2)
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cropped_img = img[y_min:y_max, x_min:x_max]

        detected_location = Location.objects.create(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            image_id=pcb_image.id
        )
        cropped_img_path = 'main/cropped_images/' + str(box_num) + '_' + image.split('/')[-1]

        cv2.imwrite(cropped_img_path, cropped_img)
        box_num += 1
        detected_class = classify(cropped_img_path, classification_model)
        classes.append(detected_class)

        models_dict = apps.get_app_config('main').models_dict
        Defect.objects.create(
            image_id=pcb_image.id,
            user_id=userid,
            type_id=detected_class,
            localization_model_id=next(key for key, value in models_dict.items() if value == localization_model),
            classification_model_id=next(key for key, value in models_dict.items() if value == classification_model),
            location_id=detected_location.id
        )

    return draw_bboxes(image_rgb, boxes, classes)


def draw_bboxes(image, bboxes, classes):
    color_map = apps.get_app_config('main').color_map
    for bbox, detected_class in zip(bboxes, classes):
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(image, (x_min - 5, y_min - 5), (x_max + 5, y_max + 5), color_map[detected_class[0]], 8)

    resized = resize_image(image)
    _, buffer = cv2.imencode('.jpg', resized)

    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64


def detect_with_yolo(model, image):
    results = model(image)
    bboxes = results[0].boxes.xyxy.cpu().numpy()
    return bboxes


def detect_with_fasterrcnn(model, image):
    with torch.no_grad():
        predictions = model(image)
    bboxes = predictions[0]['boxes'].cpu().numpy()
    return bboxes


def load_yolo_model():
    return YOLO('main/ai-models/new-yolo-1.pt')


def load_faster_r_cnn_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    faster_r_cnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=6)
    in_features = faster_r_cnn_model.roi_heads.box_predictor.cls_score.in_features
    faster_r_cnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=6)
    faster_r_cnn_model.load_state_dict(
        torch.load('main/ai-models/trained_faster_r_cnn_model.pt',
                   map_location=device))
    faster_r_cnn_model.eval()
    faster_r_cnn_model.to(device)
    return faster_r_cnn_model


def resize_image(image, max_size=1024):
    # Get the dimensions of the image
    h, w = image.shape[:2]

    # If the image is larger than the max_size, resize it
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        image = cv2.resize(image, new_size)

    return image


def load_vgg_model():
    return load_model('main/ai-models/vgg16_best_model.keras')


def load_resnet_model():
    return load_model('main/ai-models/resnet50_best_model.keras')


import numpy as np


def calculate_iou(boxA, boxB):
    x_min_A, y_min_A, x_max_A, y_max_A = boxA
    x_min_B, y_min_B, x_max_B, y_max_B = boxB

    # Calculate the intersection rectangle coordinates
    x_min_inter = max(x_min_A, x_min_B)
    y_min_inter = max(y_min_A, y_min_B)
    x_max_inter = min(x_max_A, x_max_B)
    y_max_inter = min(y_max_A, y_max_B)

    # Compute the width and height of the intersection rectangle
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)

    # Area of the intersection rectangle
    inter_area = inter_width * inter_height

    # Area of both the prediction and ground-truth rectangles
    area_A = (x_max_A - x_min_A) * (y_max_A - y_min_A)
    area_B = (x_max_B - x_min_B) * (y_max_B - y_min_B)

    # Union area
    union_area = area_A + area_B - inter_area

    # Compute the Intersection over Union (IoU)
    iou = inter_area / union_area if union_area != 0 else 0

    return iou


def non_maximum_suppression(boxes, iou_threshold):
    if len(boxes) == 0:
        return []

    # Initialize the list of picked indexes
    picked_boxes = []

    # Sort boxes by area (optional, if no score is available)
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)

    # Loop over the boxes
    while boxes:
        # Pick the last box and add it to the picked boxes
        current_box = boxes.pop(0)
        picked_boxes.append(current_box)

        # Remove all boxes with high IoU (overlap) compared to the current box
        boxes = [box for box in boxes if calculate_iou(current_box, box) < iou_threshold]

    return picked_boxes


def calculate_brightness(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(grayscale_image)
    return brightness


def adjust_brightness(image, target_brightness=150):
    current_brightness = calculate_brightness(image)

    if current_brightness < target_brightness:
        brightness_ratio = target_brightness / current_brightness
        brightened_image = np.clip(image * brightness_ratio, 0, 255).astype(np.uint8)
        return brightened_image
    else:
        return image
