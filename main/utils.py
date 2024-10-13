import torch
import cv2
import base64
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from ultralytics import YOLO
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from tensorflow.keras.models import load_model

from main.classify import classify

from main.models import Location


def preprocess_for_yolo(image):
    return image


def preprocess_for_fasterrcnn(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)


def crop_image(image, boxes, pcb_image, classification_model):
    img = cv2.imread(image)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    box_num = 1
    classes = list()
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
        # classes.append(classify(cropped_img_path, classification_model))


    return draw_bboxes(image_rgb, boxes, classes)



def draw_bboxes(image, bboxes, classes):
    # todo: add the color definition logic depending on class of defect
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    resized = resize_image(image)
    _, buffer = cv2.imencode('.png', resized)

    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64


def detect_with_yolo(model, image):
    results = model(image)
    bboxes = results[0].boxes.xyxy.cpu().numpy()
    return bboxes


def detect_with_fasterrcnn(model, image):
    with torch.no_grad():
        predictions = model(image)
    bboxes = predictions[0]['boxes'].cpu().numpy()  # Get bounding boxes
    return bboxes


def load_yolo_model():
    return YOLO('main/ai-models/trained_yolo_model.pt')


def load_faster_r_cnn_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    faster_r_cnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=6)
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

