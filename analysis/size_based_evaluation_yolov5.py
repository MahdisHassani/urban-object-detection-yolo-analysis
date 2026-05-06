import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
YOLO_DIR = os.path.join(BASE_DIR, "yolov5")
sys.path.append(YOLO_DIR)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.dataloaders import LoadImages

# CONFIG
WEIGHTS = os.path.join(BASE_DIR, "weights", "yolov5_best_320.pt")
VAL_IMAGES = os.path.join(BASE_DIR, "dataset_urban", "images", "val")
VAL_LABELS = os.path.join(BASE_DIR, "dataset_urban", "labels", "val")

IMG_SIZE = 320     # 640
CONF_THRES = 0.25
IOU_THRES = 0.5

# COCO thresholds
SMALL_TH = 32 ** 2
MEDIUM_TH = 96 ** 2

# UTILS
def compute_iou(box1, box2):
    # box format: [x1,y1,x2,y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter + 1e-6
    return inter / union


def get_size_bucket(area):
    if area < SMALL_TH:
        return "small"
    elif area < MEDIUM_TH:
        return "medium"
    else:
        return "large"


# LOAD MODEL
device = select_device("0")
model = DetectMultiBackend(WEIGHTS, device=device)
stride = model.stride
model.warmup(imgsz=(1, 3, IMG_SIZE, IMG_SIZE))

dataset = LoadImages(VAL_IMAGES, img_size=IMG_SIZE, stride=stride)

# Counters
metrics = {
    "small": {"TP":0,"FP":0,"FN":0},
    "medium": {"TP":0,"FP":0,"FN":0},
    "large": {"TP":0,"FP":0,"FN":0}
}

# LOOP
for path, im, im0s, vid_cap, s in tqdm(dataset):

    image_name = os.path.basename(path)
    label_path = os.path.join(VAL_LABELS, image_name.replace(".jpg",".txt"))

    # Load GT
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f.readlines():
                cls, x, y, w, h = map(float, line.strip().split())
                
                x1 = (x - w/2) * im0s.shape[1]
                y1 = (y - h/2) * im0s.shape[0]
                x2 = (x + w/2) * im0s.shape[1]
                y2 = (y + h/2) * im0s.shape[0]

                area = (x2-x1)*(y2-y1)
                bucket = get_size_bucket(area)

                gt_boxes.append({
                    "box":[x1,y1,x2,y2],
                    "bucket":bucket,
                    "matched":False
                })

    # Inference
    im_tensor = torch.from_numpy(im).to(device).float() / 255.0
    if len(im_tensor.shape)==3:
        im_tensor = im_tensor[None]

    preds = model(im_tensor)
    preds = non_max_suppression(preds, CONF_THRES, IOU_THRES)

    pred_boxes = []
    for det in preds:
        if len(det):
            det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0s.shape)
            for *xyxy, conf, cls in det:
                pred_boxes.append([float(xyxy[0]),float(xyxy[1]),float(xyxy[2]),float(xyxy[3])])

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

    # FN
    for gt in gt_boxes:
        if not gt["matched"]:
            metrics[gt["bucket"]]["FN"] += 1


# COMPUTE METRICS
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

df.to_csv(os.path.join(output_dir, "size_based_metrics_yolov5_320.csv"), index=False)

print(df)
print("Size-based evaluation completed.")