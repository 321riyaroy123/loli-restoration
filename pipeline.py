import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
import model

class RealTimeDetectionPipeline:
    def __init__(self, dce_weights_path, yolo_weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enhance_net = model.enhance_net_nopool().to(self.device)
        checkpoint = torch.load(dce_weights_path, map_location=self.device)
        state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        self.enhance_net.load_state_dict(state)
        self.enhance_net.eval()
        self.yolo = YOLO(yolo_weights_path)

    def process_frame(self, frame, conf=0.25):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, enhanced, _ = self.enhance_net(tensor)
        enhanced = enhanced[0].permute(1, 2, 0).cpu().numpy()
        enhanced = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
        result = self.yolo.predict(enhanced, conf=conf, verbose=False)[0]
        return result

    def run_video(self, input_path, output_path, conf=0.25):
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        try:
            for _ in tqdm(range(total), desc="Processing"):
                ok, frame = cap.read()
                if not ok:
                    break
                start = time.perf_counter()
                result = self.process_frame(frame, conf)
                annotated = result.plot()  # Ultralytics returns BGR
                elapsed = time.perf_counter() - start
                cv2.putText(annotated, f"FPS: {1/elapsed:.1f}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                writer.write(annotated)
        finally:
            cap.release()
            writer.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="output_detections.mp4")
    parser.add_argument("--dce-weights", default="weights/zero_dce/best_model.pth")
    parser.add_argument("--yolo-weights", default="weights/yolo11/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    RealTimeDetectionPipeline(args.dce_weights, args.yolo_weights).run_video(args.input, args.output, args.conf)

if __name__ == "__main__":
    main()
