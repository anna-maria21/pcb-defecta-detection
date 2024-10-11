import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from ultralytics import YOLO
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T


def preprocess_for_yolo(image):
    return image


def preprocess_for_fasterrcnn(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)


def crop_image(image, boxes):
    img = cv2.imread(image)

    box_num = 1
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cropped_img = img[y_min:y_max, x_min:x_max]

        cropped_img_path = 'D:/магістерська/pcb_defects_detection/main/cropped_images/' + str(box_num) + '_' + image.split('/')[-1]

        cv2.imwrite(cropped_img_path, cropped_img)
        box_num += 1
        # classification_result = classify(cropped_img_path)
    return ''



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
    return YOLO('D:/магістерська/pcb_defects_detection/main/ai-models/trained_yolo_model.pt')


def load_faster_r_cnn_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    faster_r_cnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=6)
    in_features = faster_r_cnn_model.roi_heads.box_predictor.cls_score.in_features
    faster_r_cnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=6)
    faster_r_cnn_model.load_state_dict(
        torch.load('D:/магістерська/pcb_defects_detection/main/ai-models/trained_faster_r_cnn_model.pt',
                   map_location=device))
    faster_r_cnn_model.eval()
    faster_r_cnn_model.to(device)
    return faster_r_cnn_model