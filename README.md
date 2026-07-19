# Low-Light Image Enhancement and Object Detection

A computer-vision pipeline that enhances low-light road scenes with a Zero-DCE-based model before applying YOLO11 object detection.

## Demo

<p align="center">
  <img src="assets/demo.gif" alt="Low-Light Enhancement + YOLO11 Detection Demo" width="850">
</p>

## Pipeline

```text
Low-light video
      │
      ▼
Zero-DCE Enhancement
      │
      ▼
YOLO11 Object Detection
      │
      ▼
Annotated Output Video
```

## Setup

```bash
pip install -r requirements.txt
```

Place the trained models at:

- `weights/zero_dce/best_model.pth`
- `weights/yolo11/best.pt`

## Run Inference

```bash
python pipeline.py --input samples/test.mp4 --output output_detections.mp4
```

## Prepare Enhanced YOLO Dataset

```bash
python prepare_yolo_dataset.py --input loli_yolo --output loli_yolo_enhanced
```

## Train YOLO11

```bash
python train_yolo.py --data loli_yolo_enhanced/data.yaml --epochs 30
```

## Dataset

The training datasets are excluded from the repository because of their size. Keep local or external backups of:

- `lol_dataset`
- `loli_yolo`
- `loli_yolo_enhanced`

## Repository Structure

```text
├── assets/
│   └── demo.gif
├── model.py
├── utils.py
├── dataloader.py
├── pipeline.py
├── prepare_yolo_dataset.py
├── train_yolo.py
├── requirements.txt
├── weights/
│   ├── zero_dce/
│   │   └── best_model.pth
│   └── yolo11/
│       └── best.pt
└── samples/
```
