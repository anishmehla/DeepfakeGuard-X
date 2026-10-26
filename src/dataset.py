# src/dataset.py
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def apply_stress_degradation(image: np.ndarray, stress_params: dict | None) -> np.ndarray:
    """
    Deterministic post-processing degradations for social media stress testing:
      - JPEG Compression: Q in [90, 70, 50, 30]
      - Gaussian Blur: sigma in [1, 3, 5]
      - Spatial Rescaling: downscale by 0.5x or 0.25x, then upscale back to 256x256
    """
    if not stress_params:
        return image

    deg_type = stress_params.get("type")
    val = stress_params.get("value")
    h, w = image.shape[:2]

    if deg_type == "jpeg":
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(val)]
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), encode_param)
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

    elif deg_type == "blur":
        sigma = float(val)
        ksize = int(2 * round(3 * sigma) + 1)
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    elif deg_type == "scale":
        scale = float(val)
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        down = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        return cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)

    return image


def get_transforms(image_size: int = 256, mode: str = "val", robust_aug: bool = False):
    if mode == "train":
        aug_list = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
        ]
        if robust_aug:
            aug_list.extend([
                A.ImageCompression(quality_lower=30, quality_upper=90, p=0.5),
                A.GaussianBlur(blur_limit=(3, 9), sigma_limit=(1.0, 5.0), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
            ])
        aug_list.extend([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        return A.Compose(aug_list)

    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class DeepfakeDataset(Dataset):
    def __init__(self, csv_path: str, crops_dir: str, transform=None, stress_params: dict | None = None):
        self.df = pd.read_csv(csv_path)
        self.crops_dir = Path(crops_dir)
        self.transform = transform
        self.stress_params = stress_params
        self.samples = []

        for _, row in self.df.iterrows():
            vid_id = str(row["video_id"])
            label = int(row["label"])
            vid_folder = self.crops_dir / vid_id

            if vid_folder.exists():
                frames = sorted(list(vid_folder.glob("*.png")) + list(vid_folder.glob("*.jpg")))
                for frame_path in frames:
                    self.samples.append({
                        "frame_path": str(frame_path),
                        "video_id": vid_id,
                        "label": label,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        img_bgr = cv2.imread(item["frame_path"])
        if img_bgr is None:
            raise RuntimeError(f"Could not read image: {item['frame_path']}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.stress_params is not None:
            img_rgb = apply_stress_degradation(img_rgb, self.stress_params)

        if self.transform is not None:
            img_tensor = self.transform(image=img_rgb)["image"]
        else:
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        label = torch.tensor(item["label"], dtype=torch.float32)
        return img_tensor, label, item["video_id"]