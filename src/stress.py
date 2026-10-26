# src/stress.py
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.dataset import DeepfakeDataset, get_transforms
from src.model import DeepfakeDetector
from src.evaluate import evaluate_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="data/splits/celebdf_test_split.csv")
    parser.add_argument("--crops_dir", type=str, default="data/crops")
    parser.add_argument("--output", type=str, default="reports/stress_results.json")
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

    sweeps = [
        ("clean", None),
        ("jpeg_90", {"type": "jpeg", "value": 90}),
        ("jpeg_70", {"type": "jpeg", "value": 70}),
        ("jpeg_50", {"type": "jpeg", "value": 50}),
        ("jpeg_30", {"type": "jpeg", "value": 30}),
        ("blur_1", {"type": "blur", "value": 1}),
        ("blur_3", {"type": "blur", "value": 3}),
        ("blur_5", {"type": "blur", "value": 5}),
        ("scale_0.5", {"type": "scale", "value": 0.5}),
        ("scale_0.25", {"type": "scale", "value": 0.25}),
    ]

    stress_results = {}
    val_tf = get_transforms(256, mode="val")

    for name, params in sweeps:
        print(f"Evaluating degradation sweep: {name}")
        ds = DeepfakeDataset(args.split, args.crops_dir, transform=val_tf, stress_params=params)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
        stress_results[name] = evaluate_loader(model, loader, device, temperature=temperature)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stress_results, f, indent=2)
    print(f"Exported stress test results to {out_path}")


if __name__ == "__main__":
    main()