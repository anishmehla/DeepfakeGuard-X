# src/evaluate.py
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm

from src.dataset import DeepfakeDataset, get_transforms
from src.model import DeepfakeDetector


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    return float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0))


def compute_binary_metrics(y_true: list[int], y_scores: list[float]) -> dict:
    y_t = np.asarray(y_true, dtype=int)
    y_s = np.asarray(y_scores, dtype=float)
    if len(np.unique(y_t)) < 2:
        return {"AUC": 0.0, "pAUC_0.05": 0.0, "EER": 0.5, "AP": 0.0}

    return {
        "AUC": float(roc_auc_score(y_t, y_s)),
        "pAUC_0.05": float(roc_auc_score(y_t, y_s, max_fpr=0.05)),
        "EER": compute_eer(y_t, y_s),
        "AP": float(average_precision_score(y_t, y_s)),
    }


@torch.no_grad()
def evaluate_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device, temperature: float = 1.0) -> dict:
    model.eval()
    frame_labels, frame_probs = [], []
    video_probs = defaultdict(list)
    video_labels = {}

    for images, labels, video_ids in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        logits = model(images) / temperature
        probs = torch.sigmoid(logits).cpu().numpy()
        lbls = labels.numpy()

        frame_probs.extend(probs.tolist())
        frame_labels.extend(lbls.tolist())

        for vid_id, p, l in zip(video_ids, probs, lbls):
            video_probs[vid_id].append(float(p))
            video_labels[vid_id] = int(l)

    vid_y, vid_mean_p, vid_top3_p = [], [], []
    for vid_id, p_list in video_probs.items():
        vid_y.append(video_labels[vid_id])
        vid_mean_p.append(float(np.mean(p_list)))
        top3 = sorted(p_list, reverse=True)[:3]
        vid_top3_p.append(float(np.mean(top3)))

    return {
        "frame_level": compute_binary_metrics(frame_labels, frame_probs),
        "video_mean": compute_binary_metrics(vid_y, vid_mean_p),
        "video_top3": compute_binary_metrics(vid_y, vid_top3_p),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--crops_dir", type=str, default="data/crops")
    parser.add_argument("--output", type=str, default="reports/eval_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})
    temperature = float(ckpt.get("temperature", 1.0))

    model = DeepfakeDetector(
        backbone_name=cfg.get("model", {}).get("backbone", "efficientnet_b4"),
        pretrained=False,
        use_spatial_attention=cfg.get("model", {}).get("use_spatial_attention", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    ds = DeepfakeDataset(args.split, args.crops_dir, transform=get_transforms(256, mode="val"))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    metrics = evaluate_loader(model, loader, device, temperature=temperature)

    # Compute Cross-Generator Drop (Delta_AUC = AUC_FF++ - AUC_Celeb-DF)
    ffpp_val_auc = ckpt.get("val_auc")
    if ffpp_val_auc is not None:
        metrics["Delta_AUC_mean"] = float(ffpp_val_auc - metrics["video_mean"]["AUC"])

    print(json.dumps(metrics, indent=2))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()