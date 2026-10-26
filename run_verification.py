# run_verification.py
import sys
from pathlib import Path
import cv2
import numpy as np


def create_dummy_video(path: Path, num_frames: int = 24, size: tuple = (320, 320)):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Using mp4v codec for cross-platform compatibility
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 12.0, size)

    if not out.isOpened():
        raise RuntimeError(f"OpenCV VideoWriter failed to open for {path}. Check your opencv installation.")

    for i in range(num_frames):
        # Draw a synthetic colored circle moving across the frame so it simulates a target object
        frame = np.full((size[1], size[0], 3), 40, dtype=np.uint8)
        cx = int((i / num_frames) * (size[0] - 80)) + 40
        cy = size[1] // 2
        cv2.circle(frame, (cx, cy), 35, (180, 105, 255), -1)
        out.write(frame)

    out.release()


def main():
    print("1. Creating dummy dataset directories and videos...")
    base_raw = Path("data/raw")

    # FF++ requires at least 5 distinct sources for an 80/20 train/val split without errors
    for i in range(1, 6):
        # Original (Real)
        create_dummy_video(base_raw / f"ffpp/original/{i:03d}.mp4")
        # Manipulated (Fake): source_id is stem before '_'
        create_dummy_video(base_raw / f"ffpp/Deepfakes/{i:03d}_{i+10:03d}.mp4")

    # Celeb-DF test clips
    create_dummy_video(base_raw / "celebdf/Celeb-real/id0_0001.mp4")
    create_dummy_video(base_raw / "celebdf/Celeb-synthesis/id0_id1_0001.mp4")

    raw_videos = list(base_raw.rglob("*.mp4"))
    print(f" -> Generated {len(raw_videos)} dummy .mp4 videos in data/raw/")

    print("\n2. Executing src.data pipeline...")
    # Import and run src.data
    import src.data as data_module
    sys.argv = ["src.data"]
    data_module.main()

    # Verify created artifacts
    splits = list(Path("data/splits").glob("*.csv"))
    crops = list(Path("data/crops").rglob("*.png"))

    print("\n3. Verification Results:")
    print(f" -> Splits created ({len(splits)} CSVs): {[s.name for s in splits]}")
    print(f" -> Face crops created: {len(crops)} images across {len(list(Path('data/crops').iterdir()))} video directories")

    if len(splits) > 0 and len(crops) > 0:
        print("\nAll pipeline components in src/data.py work as expected.")
    else:
        print("\nVerification incomplete. Ensure data/splits and data/crops directories are writable.")


if __name__ == "__main__":
    main()