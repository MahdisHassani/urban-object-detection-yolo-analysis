import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

WEIGHTS = os.path.join(BASE_DIR, "weights", "yolov8_best_320.pt")
VAL_IMAGES = os.path.join(BASE_DIR, "dataset_urban", "images", "val")
VAL_LABELS = os.path.join(BASE_DIR, "dataset_urban", "labels", "val")

# CONFIG
IMG_SIZE = 320      # 640
CONF_THRES = 0.25
IOU_THRES = 0.5

# COCO thresholds
SMALL_TH = 32 ** 2
MEDIUM_TH = 96 ** 2

# UTILS
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


def get_size_bucket(area):
    if area < SMALL_TH:
        return "small"
    elif area < MEDIUM_TH:
        return "medium"
    else:
        return "large"

model = YOLO(WEIGHTS)

metrics = {
    "small": {"TP":0,"FP":0,"FN":0},
    "medium": {"TP":0,"FP":0,"FN":0},
    "large": {"TP":0,"FP":0,"FN":0}
}

image_files = [f for f in os.listdir(VAL_IMAGES) if f.endswith(".jpg")]

for image_name in tqdm(image_files):

    img_path = os.path.join(VAL_IMAGES, image_name)
    label_path = os.path.join(VAL_LABELS, image_name.replace(".jpg",".txt"))

    im0 = cv2.imread(img_path)

    # Ground Truth
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f.readlines():
                cls, x, y, w, h = map(float, line.strip().split())

                x1 = (x - w/2) * im0.shape[1]
                y1 = (y - h/2) * im0.shape[0]
                x2 = (x + w/2) * im0.shape[1]
                y2 = (y + h/2) * im0.shape[0]

                area = (x2 - x1) * (y2 - y1)
                bucket = get_size_bucket(area)

                gt_boxes.append({
                    "box":[x1,y1,x2,y2],
                    "bucket":bucket,
                    "matched":False
                })

    # Inference
    results = model(im0, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, verbose=False)

    pred_boxes = []

    for r in results:
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                pred_boxes.append(box.tolist())

    # Matching
    for pred in pred_boxes:
        matched = False
        for gt in gt_boxes:
            if not gt["matched"]:
                iou = compute_iou(pred, gt["box"])
                if iou >= IOU_THRES:
                    metrics[gt["bucket"]]["TP"] += 1
                    gt["matched"] = True
                    matched = True
                    break

        if not matched:
            metrics["medium"]["FP"] += 1

    for gt in gt_boxes:
        if not gt["matched"]:
            metrics[gt["bucket"]]["FN"] += 1

# COMPUTE
rows = []

for bucket in metrics:
    TP = metrics[bucket]["TP"]
    FP = metrics[bucket]["FP"]
    FN = metrics[bucket]["FN"]

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    rows.append({
        "size": bucket,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "F1": f1
    })

df = pd.DataFrame(rows)

output_dir = os.path.join(BASE_DIR, "analysis_results")
os.makedirs(output_dir, exist_ok=True)

df.to_csv(os.path.join(output_dir, "size_based_metrics_yolov8_320.csv"), index=False)

print(df)
print("Size-based evaluation completed.")