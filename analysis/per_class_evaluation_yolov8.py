import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

WEIGHTS = os.path.join(BASE_DIR, "weights", "yolov8_best_640.pt")
VAL_IMAGES = os.path.join(BASE_DIR, "dataset_urban", "images", "val")
VAL_LABELS = os.path.join(BASE_DIR, "dataset_urban", "labels", "val")

# CONFIG
IMG_SIZE = 640      # 320
CONF_THRES = 0.25
IOU_THRES = 0.5

CLASS_NAMES = [
    "bicycle", "bus", "car",
    "motorbike", "person", "train"
]

model = YOLO(WEIGHTS)

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])

    union = area1 + area2 - inter + 1e-6
    return inter / union

image_files = [f for f in os.listdir(VAL_IMAGES) if f.endswith(".jpg")]

metrics = {cls: {"TP":0,"FP":0,"FN":0} for cls in CLASS_NAMES}

for image_name in tqdm(image_files):

    img_path = os.path.join(VAL_IMAGES, image_name)
    label_path = os.path.join(VAL_LABELS, image_name.replace(".jpg",".txt"))

    im0 = cv2.imread(img_path)

    # Ground Truth
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f.readlines():
                cls_id, x, y, w, h = map(float, line.strip().split())
                cls_id = int(cls_id)

                x1 = (x - w/2) * im0.shape[1]
                y1 = (y - h/2) * im0.shape[0]
                x2 = (x + w/2) * im0.shape[1]
                y2 = (y + h/2) * im0.shape[0]

                gt_boxes.append({
                    "cls": CLASS_NAMES[cls_id],
                    "box":[x1,y1,x2,y2],
                    "matched":False
                })

    # Inference
    results = model(im0, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, verbose=False)

    pred_boxes = []

    for r in results:
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, cls_id in zip(boxes, classes):
                pred_boxes.append({
                    "cls": CLASS_NAMES[int(cls_id)],
                    "box": box.tolist()
                })

    # Matching
    for pred in pred_boxes:
        matched = False
        for gt in gt_boxes:
            if not gt["matched"] and pred["cls"] == gt["cls"]:
                iou = compute_iou(pred["box"], gt["box"])
                if iou >= IOU_THRES:
                    metrics[gt["cls"]]["TP"] += 1
                    gt["matched"] = True
                    matched = True
                    break
        if not matched:
            metrics[pred["cls"]]["FP"] += 1

    for gt in gt_boxes:
        if not gt["matched"]:
            metrics[gt["cls"]]["FN"] += 1

# COMPUTE
rows = []

for cls in CLASS_NAMES:
    TP = metrics[cls]["TP"]
    FP = metrics[cls]["FP"]
    FN = metrics[cls]["FN"]

    precision = TP/(TP+FP+1e-6)
    recall = TP/(TP+FN+1e-6)
    f1 = 2*precision*recall/(precision+recall+1e-6)

    rows.append({
        "class":cls,
        "TP":TP,
        "FP":FP,
        "FN":FN,
        "precision":precision,
        "recall":recall,
        "F1":f1
    })

df = pd.DataFrame(rows)

output_dir = os.path.join(BASE_DIR, "analysis_results")
os.makedirs(output_dir, exist_ok=True)

df.to_csv(os.path.join(output_dir, "per_class_metrics_yolov8_640.csv"), index=False)

print(df)
print("Per-class evaluation completed.")