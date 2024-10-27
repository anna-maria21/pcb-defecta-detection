import base64
import cv2
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
import io
import os
import tempfile
from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from datetime import datetime

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
    report_colors = apps.get_app_config('main').report_colors
    defect_types = apps.get_app_config('main').defect_types
    defects = list()
    for bbox, detected_class in zip(bboxes, classes):
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(image, (x_min - 5, y_min - 5), (x_max + 5, y_max + 5), color_map[detected_class[0]], 8)
        defects.append({
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "color": report_colors[detected_class[0]],
            "type": defect_types[detected_class[0]]
            })

    resized = resize_image(image)
    _, buffer = cv2.imencode('.jpg', resized)

    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64, defects


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
    h, w = image.shape[:2]

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

    x_min_inter = max(x_min_A, x_min_B)
    y_min_inter = max(y_min_A, y_min_B)
    x_max_inter = min(x_max_A, x_max_B)
    y_max_inter = min(y_max_A, y_max_B)

    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    inter_area = inter_width * inter_height

    area_A = (x_max_A - x_min_A) * (y_max_A - y_min_A)
    area_B = (x_max_B - x_min_B) * (y_max_B - y_min_B)
    union_area = area_A + area_B - inter_area

    iou = inter_area / union_area if union_area != 0 else 0
    return iou


def non_maximum_suppression(boxes, iou_threshold):
    if len(boxes) == 0:
        return []

    picked_boxes = []
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)

    while boxes:
        current_box = boxes.pop(0)
        picked_boxes.append(current_box)
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


def get_pdf(request):
    defects = request.session.get('defects')
    base64_image = request.session.get('result')
    classification_model = request.session.get('classification_model')
    localization_model = request.session.get('localization_model')

    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="pcb_defects_report.pdf"'

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    pdf = canvas.Canvas(response, pagesize=letter)
    pdf.setTitle(f"PCB Defects Report - {formatted_time}")
    pdfmetrics.registerFont(TTFont('TimesNewRoman', 'times.ttf'))
    pdf.setFont('TimesNewRoman', 14)
    pdf.drawString(40, 750, f"Звіт локалізації та класифікації - {formatted_time}")
    pdf.drawString(40, 740, "--------------------------------------------------------------------")

    pdf.drawString(40, 710, f"Використані моделі: {localization_model}, {classification_model}")

    image_data = base64.b64decode(base64_image)
    img = Image.open(io.BytesIO(image_data))
    original_width, original_height = img.size
    max_width = 400
    max_height = 350
    aspect_ratio = original_width / original_height

    if original_width > max_width or original_height > max_height:
        if original_width / max_width > original_height / max_height:
            new_width = max_width
            new_height = max_width / aspect_ratio
        else:
            new_height = max_height
            new_width = max_height * aspect_ratio
    else:
        new_width, new_height = original_width, original_height

    y = 680 - new_height
    counter = 1
    for defect in defects:
        pdf.drawString(40, y,
                       f"{counter}) {defect['type']}; координати: ({defect['x_min']}, {defect['y_min']}), ({defect['x_max']}, {defect['y_max']}); колір: {defect['color']}")
        y -= 20
        counter += 1

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    try:
        img.save(temp_file, format='JPEG')
        temp_file.seek(0)
        pdf.drawImage(temp_file.name, 40, 700-new_height, width=new_width, height=new_height)
    finally:
        temp_file.close()

    pdf.showPage()
    pdf.save()

    return response

