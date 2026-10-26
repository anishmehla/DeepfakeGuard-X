# src/calibrate.py
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import DeepfakeDataset, get_transforms
from src.model import DeepfakeDetector


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    preds = (probs >= 0.5).astype(int)
    confidences = np.where(preds == 1, probs, 1.0 - probs)
    accuracies = (preds == labels).astype(float)

    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if np.any(mask):
            acc = np.mean(accuracies[mask])
            conf = np.mean(confidences[mask])
            ece += np.abs(conf - acc) * np.mean(mask)
    return float(ece)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val_split", type=str, default="data/splits/ffpp_val_split.csv")
    parser.add_argument("--crops_dir", type=str, default="data/crops")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})

    model = DeepfakeDetector(
        backbone_name=cfg.get("model", {}).get("backbone", "efficientnet_b4"),
        pretrained=False,
        use_spatial_attention=cfg.get("model", {}).get("use_spatial_attention", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = DeepfakeDataset(args.val_split, args.crops_dir, transform=get_transforms(256, mode="val"))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            all_logits.append(model(images.to(device)).cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits).to(device)
    labels = torch.cat(all_labels).to(device)

    ece_before = compute_ece(torch.sigmoid(logits).cpu().numpy(), labels.cpu().numpy())

    temperature = nn.Parameter(torch.ones(1, device=device) * 1.5)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / temperature.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    optimal_T = float(temperature.clamp(min=1e-3).item())
    ece_after = compute_ece(torch.sigmoid(logits / optimal_T).cpu().numpy(), labels.cpu().numpy())

    print(f"Learned Temperature T = {optimal_T:.4f} | ECE Before: {ece_before:.4f} -> ECE After: {ece_after:.4f}")
    ckpt["temperature"] = optimal_T
    torch.save(ckpt, args.checkpoint)


if __name__ == "__main__":
    main()