# src/data.py
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def generate_ffpp_splits(raw_ffpp_dir: Path, splits_dir: Path, seed: int = 42):
    """
    Expects raw FF++ folders under data/raw/ffpp/ (e.g., original, Deepfakes, Face2Face, FaceSwap, NeuralTextures).
    """
    records = []
    if not raw_ffpp_dir.exists():
        print(f"[Info] {raw_ffpp_dir} not found. Skipping FF++ CSV generation.")
        return []

    for vid_path in sorted(raw_ffpp_dir.rglob("*.mp4")):
        folder = vid_path.parent.name
        stem = vid_path.stem
        # In FF++, manipulated videos are named <source_id>_<target_id>.mp4
        source_id = stem.split("_")[0]

        if folder.lower() in {"real", "original", "actors", "youtube"}:
            label = 0
            manip_type = "Original"
        else:
            label = 1
            manip_type = folder

        records.append({
            "video_id": f"ffpp_{manip_type}_{stem}",
            "video_path": str(vid_path),
            "source_id": f"ffpp_{source_id}",
            "manipulation_type": manip_type,
            "label": label,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return []

    unique_sources = df["source_id"].unique()
    train_ids, val_ids = train_test_split(unique_sources, test_size=0.20, random_state=seed)

    # Mandatory disjoint identity check
    assert len(set(train_ids) & set(val_ids)) == 0, "Identity leakage detected between train and val!"

    train_df = df[df["source_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["source_id"].isin(val_ids)].reset_index(drop=True)

    splits_dir.mkdir(parents=True, exist_ok=True)
    train_csv = splits_dir / "ffpp_train_split.csv"
    val_csv = splits_dir / "ffpp_val_split.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    print(f"Saved {len(train_df)} train and {len(val_df)} val records to {splits_dir}")
    return [train_csv, val_csv]


def generate_celebdf_split(raw_celebdf_dir: Path, splits_dir: Path):
    """
    Expects Celeb-DF folders under data/raw/celebdf/ (e.g., Celeb-real, YouTube-real, Celeb-synthesis).
    """
    records = []
    if not raw_celebdf_dir.exists():
        print(f"[Info] {raw_celebdf_dir} not found. Skipping Celeb-DF CSV generation.")
        return []

    for vid_path in sorted(raw_celebdf_dir.rglob("*.mp4")):
        folder = vid_path.parent.name
        stem = vid_path.stem
        source_id = stem.split("_")[0]

        if "real" in folder.lower():
            label = 0
            manip_type = "Real"
        else:
            label = 1
            manip_type = "Celeb-synthesis"

        records.append({
            "video_id": f"celebdf_{folder}_{stem}",
            "video_path": str(vid_path),
            "source_id": f"celebdf_{source_id}",
            "manipulation_type": manip_type,
            "label": label,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return []

    splits_dir.mkdir(parents=True, exist_ok=True)
    test_csv = splits_dir / "celebdf_test_split.csv"
    df.to_csv(test_csv, index=False)
    print(f"Saved {len(df)} Celeb-DF test records to {test_csv}")
    return [test_csv]


def crop_with_margin(frame_rgb: np.ndarray, box: np.ndarray, margin_ratio: float = 0.20, size: int = 256):
    h, w, _ = frame_rgb.shape
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    # Expand bounding box by 20% outer margin
    mx = bw * margin_ratio
    my = bh * margin_ratio
    nx1 = max(0, int(x1 - mx))
    ny1 = max(0, int(y1 - my))
    nx2 = min(w, int(x2 + mx))
    ny2 = min(h, int(y2 + my))

    crop = frame_rgb[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return None
    crop_resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return crop_resized


def extract_faces_from_csv(csv_path: Path, crops_dir: Path, mtcnn: MTCNN, num_frames: int = 16, margin: float = 0.20, size: int = 256):
    df = pd.read_csv(csv_path)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Cropping {csv_path.name}"):
        vid_id = str(row["video_id"])
        vid_path = str(row["video_path"])
        out_dir = crops_dir / vid_id

        if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= num_frames:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        saved_count = 0

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(frame_rgb)

            if boxes is not None and len(boxes) > 0:
                # Select largest detected face
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                best_box = boxes[int(np.argmax(areas))]
                face_rgb = crop_with_margin(frame_rgb, best_box, margin_ratio=margin, size=size)
            else:
                # Fallback center crop if face detector misses extreme frame
                h, w, _ = frame_rgb.shape
                s = min(h, w)
                y1, x1 = (h - s) // 2, (w - s) // 2
                face_rgb = cv2.resize(frame_rgb[y1:y1 + s, x1:x1 + s], (size, size))

            if face_rgb is not None:
                out_file = out_dir / f"frame_{saved_count:02d}.png"
                cv2.imwrite(str(out_file), cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))
                saved_count += 1

        cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffpp_dir", type=str, default="data/raw/ffpp")
    parser.add_argument("--celebdf_dir", type=str, default="data/raw/celebdf")
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    parser.add_argument("--crops_dir", type=str, default="data/crops")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, device=device)

    splits_dir = Path(args.splits_dir)
    crops_dir = Path(args.crops_dir)

    csv_files = []
    csv_files.extend(generate_ffpp_splits(Path(args.ffpp_dir), splits_dir))
    csv_files.extend(generate_celebdf_split(Path(args.celebdf_dir), splits_dir))

    # Also process existing CSVs in data/splits/ if raw video scan was skipped
    if not csv_files and splits_dir.exists():
        csv_files = sorted(list(splits_dir.glob("*.csv")))

    for csv_path in csv_files:
        extract_faces_from_csv(
            csv_path,
            crops_dir,
            mtcnn,
            num_frames=args.num_frames,
            margin=args.margin,
            size=args.image_size,
        )


if __name__ == "__main__":
    main()