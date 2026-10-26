# src/train.py
import argparse
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.dataset import DeepfakeDataset, get_transforms
from src.model import DeepfakeDetector
from src.evaluate import evaluate_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = Path(cfg["train"]["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    use_wandb = cfg["train"].get("use_wandb", False)
    if use_wandb:
        wandb.init(project=cfg["project_name"], name=cfg["run_name"], config=cfg)

    train_tf = get_transforms(
        image_size=cfg["data"]["image_size"],
        mode="train",
        robust_aug=cfg["data"].get("use_robust_aug", False),
    )
    val_tf = get_transforms(image_size=cfg["data"]["image_size"], mode="val")

    train_ds = DeepfakeDataset(cfg["data"]["train_csv"], cfg["data"]["crops_dir"], transform=train_tf)
    val_ds = DeepfakeDataset(cfg["data"]["val_csv"], cfg["data"]["crops_dir"], transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    model = DeepfakeDetector(
        backbone_name=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        use_spatial_attention=cfg["model"].get("use_spatial_attention", False),
        dropout=cfg["model"].get("dropout", 0.3),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_auc = 0.0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['train']['epochs']}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.size(0)

        scheduler.step()
        avg_loss = total_loss / max(1, len(train_ds))
        val_metrics = evaluate_loader(model, val_loader, device)
        val_auc = val_metrics["video_mean"]["AUC"]
        val_eer = val_metrics["video_mean"]["EER"]

        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Val Video AUC: {val_auc:.4f} | Val EER: {val_eer:.4f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_video_auc": val_auc,
                "val_video_eer": val_eer,
                "val_video_pauc": val_metrics["video_mean"]["pAUC_0.05"],
            })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": best_val_auc,
                "config": cfg,
                "temperature": 1.0,
            }, save_path)
            print(f" -> Saved best model checkpoint to {save_path}")


if __name__ == "__main__":
    main()