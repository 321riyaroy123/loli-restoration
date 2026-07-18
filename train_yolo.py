import argparse
from ultralytics import YOLO

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--model-size",choices=["n","s","m","l","x"],default="n")
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--img-size",type=int,default=640)
    p.add_argument("--device",default="0")
    a=p.parse_args()
    YOLO(f"yolo11{a.model_size}.pt").train(
        data=a.data, epochs=a.epochs, batch=a.batch_size, imgsz=a.img_size,
        device=a.device, patience=50, save=True, save_period=10,
        optimizer="AdamW", lr0=0.01, lrf=0.01, weight_decay=0.0005,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5,
        fliplr=0.5, mosaic=1.0, val=True, plots=True,
        project="yolo_training", name="enhanced_road_detection", exist_ok=True)
