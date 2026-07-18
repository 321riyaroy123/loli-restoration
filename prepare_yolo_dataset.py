import argparse
import shutil
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm
import model

CLASS_NAMES = ["bicycle","bus","car","lane","person","road","road sign","traffic cone",
               "traffic signal","truck","two-wheeler","zebra crossing"]

class YOLODatasetPreparer:
    def __init__(self, weights):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = model.enhance_net_nopool().to(self.device)
        ckpt = torch.load(weights, map_location=self.device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.net.load_state_dict(state)
        self.net.eval()

    def enhance(self, path):
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
        tensor = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, enhanced, _ = self.net(tensor)
        image = enhanced[0].permute(1,2,0).cpu().numpy()
        return (np.clip(image,0,1)*255).astype(np.uint8)

    def prepare(self, source, destination):
        source, destination = Path(source), Path(destination)
        for split in ("train","val","test"):
            src_images, src_labels = source/split/"images", source/split/"labels"
            if not src_images.exists():
                continue
            dst_images, dst_labels = destination/split/"images", destination/split/"labels"
            dst_images.mkdir(parents=True, exist_ok=True)
            dst_labels.mkdir(parents=True, exist_ok=True)
            images = list(src_images.glob("*.jpg")) + list(src_images.glob("*.jpeg")) + list(src_images.glob("*.png"))
            for image_path in tqdm(images, desc=split):
                Image.fromarray(self.enhance(image_path)).save(dst_images/image_path.name)
                label = src_labels/f"{image_path.stem}.txt"
                if label.exists():
                    shutil.copy2(label, dst_labels/label.name)
        data = {"path": str(destination.resolve()), "train":"train/images","val":"val/images",
                "test":"test/images","nc":len(CLASS_NAMES),"names":CLASS_NAMES}
        with open(destination/"data.yaml","w",encoding="utf-8") as f:
            yaml.safe_dump(data,f,sort_keys=False)

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--weights",default="weights/zero_dce/best_model.pth")
    a=p.parse_args()
    YOLODatasetPreparer(a.weights).prepare(a.input,a.output)
