# Low-Light Image Enhancement and Object Detection

A computer-vision pipeline that enhances low-light road scenes with a Zero-DCE-based model before applying YOLO11 object detection.

## Pipeline

Low-light video → Zero-DCE enhancement → YOLO11 detection → Annotated video

## Setup

```bash
pip install -r requirements.txt
```

Place the trained models at:

- `weights/zero_dce/best_model.pth`
- `weights/yolo11/best.pt`

## Run inference

```bash
python pipeline.py --input samples/test.mp4 --output output_detections.mp4
```

## Prepare enhanced YOLO data

```bash
python prepare_yolo_dataset.py --input loli_yolo --output loli_yolo_enhanced
```

## Train YOLO11

```bash
python train_yolo.py --data loli_yolo_enhanced/data.yaml --epochs 30
```

## Data

Training datasets are excluded from Git due to their size. Keep local or external backups of `lol_dataset`, `loli_yolo`, and `loli_yolo_enhanced`.

## Repository layout

```text
├── model.py
├── utils.py
├── dataloader.py
├── pipeline.py
├── prepare_yolo_dataset.py
├── train_yolo.py
├── requirements.txt
├── weights/
│   ├── zero_dce/best_model.pth
│   └── yolo11/best.pt
└── samples/
    ├── test.mp4
    └── output_detections.mp4
```
