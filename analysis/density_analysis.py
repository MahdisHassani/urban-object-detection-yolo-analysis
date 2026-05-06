import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
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

IMG_SIZE = 320
CONF_THRES = 0.25
IOU_THRES = 0.45

OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["bicycle", "bus", "car", "motorbike", "person", "train"]


# LOAD MODEL
device = select_device("0")
model = DetectMultiBackend(WEIGHTS, device=device)
stride = model.stride
model.warmup(imgsz=(1, 3, IMG_SIZE, IMG_SIZE))


# LOAD DATA
dataset = LoadImages(VAL_IMAGES, img_size=IMG_SIZE, stride=stride)

results = []

# INFERENCE LOOP
for path, im, im0s, vid_cap, s in tqdm(dataset):
    
    im = torch.from_numpy(im).to(device)
    im = im.float() / 255.0
    if len(im.shape) == 3:
        im = im[None]

    pred = model(im)
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES)

    counts = {name: 0 for name in CLASS_NAMES}

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
            
            for *xyxy, conf, cls in det:
                cls_name = CLASS_NAMES[int(cls)]
                counts[cls_name] += 1

    total_objects = sum(counts.values())

    if total_objects < 3:
        traffic = "low"
    elif total_objects < 7:
        traffic = "medium"
    else:
        traffic = "high"

    row = {
        "image": os.path.basename(path),
        **counts,
        "total_objects": total_objects,
        "traffic_level": traffic
    }

    results.append(row)

# SAVE CSV
df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTPUT_DIR, "density_results.csv"), index=False)

print("CSV saved.")

# PLOTS
class_totals = df[CLASS_NAMES].sum()

plt.figure(figsize=(10, 6))
class_totals.plot(kind="bar")
plt.title("Class Distribution in Validation Set")
plt.ylabel("Total Detections")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))

plt.figure()
df["total_objects"].hist(bins=20)
plt.title("Objects per Image Distribution")
plt.xlabel("Number of Objects")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "objects_per_image_hist.png"))

top10 = df.sort_values("total_objects", ascending=False).head(10)
top10.to_csv(os.path.join(OUTPUT_DIR, "top_10_crowded_images.csv"), index=False)

print("All analysis completed successfully.")