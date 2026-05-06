# Urban Object Detection: YOLOv5 vs YOLOv8 Analysis

This project presents a **controlled experimental study on urban object detection**, focusing on how model architecture and input resolution affect detection performance.

The study compares **YOLOv5 and YOLOv8** on a filtered subset of the Pascal VOC 2007 dataset, with an emphasis on:

* Scale sensitivity (small, medium, large objects)
* Resolution impact (320 vs 640)
* Per-class performance differences
* Detection behavior trade-offs (recall vs localization)

---

## Project Overview

Rather than building a detector only, this project is designed as an **analysis-driven pipeline**.

Key aspects:

* Same dataset
* Same training epochs
* Same evaluation pipeline
* Only controlled variables (model type & resolution)

---

## Dataset

* Dataset: Pascal VOC 2007
* Official: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/

Filtered urban classes:

* bicycle
* bus
* car
* motorbike
* person
* train

Final dataset:

* 2365 training images
* 606 validation images

---

## Experiments

| Model  | Image Size | Epochs |
| ------ | ---------- | ------ |
| YOLOv5 | 320        | 40     |
| YOLOv5 | 640        | 40     |
| YOLOv8 | 320        | 40     |
| YOLOv8 | 640        | 40     |

All experiments use identical conditions for fair comparison.

---

## Overall Performance

| Model  | Img Size | mAP50 | mAP50-95 | Recall | Precision |
| ------ | -------- | ----- | -------- | ------ | --------- |
| YOLOv5 | 320      | 0.766 | 0.478    | 0.646  | 0.874     |
| YOLOv5 | 640      | 0.803 | 0.492    | 0.716  | 0.849     |
| YOLOv8 | 320      | 0.757 | 0.526    | 0.630  | 0.846     |
| YOLOv8 | 640      | 0.798 | 0.559    | 0.695  | 0.844     |

---

## Key Findings

### 1️⃣ Resolution Impact

* Increasing resolution improves **recall significantly**
* Especially effective for **small objects**

### 2️⃣ YOLOv5 vs YOLOv8

* YOLOv5:

  * Higher recall
  * Detects more objects overall

* YOLOv8:

  * Higher mAP50-95
  * More precise localization

---

### 3️⃣ Small Object Detection

* Strong improvement from 320 → 640
* Performance is highly resolution-dependent

---

### 4️⃣ Error Patterns

* Medium-sized objects produce most false positives
* Class imbalance affects minority classes

---

## Analysis Modules

The project includes multiple evaluation pipelines:

* Per-class evaluation
* COCO-style size-based analysis
* Object density analysis

All results are saved in structured CSV format for reproducibility.

---

## Setup & Usage

### 1️⃣ Clone Repository

```bash
git clone https://github.com/MahdisHassani/urban-object-detection-yolo-analysis.git
cd urban-object-detection-yolo-analysis
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Prepare Dataset

Download Pascal VOC 2007:

http://host.robots.ox.ac.uk/pascal/VOC/voc2007/

Then run:

```bash
python data_processing/create_urban_dataset.py
```

---

### 4️⃣ Train Models

#### YOLOv5

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
python train.py --img 640 --epochs 40 --data ../configs/urban_dataset.yaml --weights yolov5n.pt
```

#### YOLOv8

```bash
yolo detect train \
  model=yolov8n.pt \
  data=configs/urban_dataset.yaml \
  imgsz=640 \
  epochs=40
```

---

### 5️⃣ Run Analysis

```bash
python analysis/per_class_evaluation_yolov5.py
python analysis/per_class_evaluation_yolov8.py

python analysis/size_based_evaluation_yolov5.py
python analysis/size_based_evaluation_yolov8.py
```

---

## Notes

* Trained weights are not included due to size limitations
* Place weights in:

```
weights/
```

* Scripts use relative paths for portability

---

## Conclusion

This project demonstrates that:

* Higher resolution improves detection performance
* YOLOv5 and YOLOv8 exhibit different strengths
* Model selection depends on application requirements
