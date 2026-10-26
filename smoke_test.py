# smoke_test.py
from pathlib import Path
import cv2
import numpy as np

def make_dummy_video(path: Path, num_frames: int = 20):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, (320, 320))
    for _ in range(num_frames):
        frame = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

# Create 5 dummy identities for FF++ (so 80/20 split has 4 train, 1 val)
for i in range(1, 6):
    make_dummy_video(Path(f"data/raw/ffpp/original/{i:03d}.mp4"))
    make_dummy_video(Path(f"data/raw/ffpp/Deepfakes/{i:03d}_{i+10:03d}.mp4"))

# Create 2 dummy videos for Celeb-DF
make_dummy_video(Path("data/raw/celebdf/Celeb-real/id0_0001.mp4"))
make_dummy_video(Path("data/raw/celebdf/Celeb-synthesis/id0_id1_0001.mp4"))

print("Dummy videos created in data/raw/! Ready to test the pipeline.")