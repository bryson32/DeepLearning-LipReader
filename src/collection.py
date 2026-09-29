import argparse
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import cv2

from camera import run_camera
from config import DATA_DIR, FRAME_COUNT


def save_take(frames, output, word):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", word):
        raise ValueError("Word must contain only letters, numbers, underscores or hyphens")
    if len(frames) != FRAME_COUNT:
        raise ValueError(f"Expected {FRAME_COUNT} frames")
    directory = Path(output) / word
    directory.mkdir(parents=True, exist_ok=True)
    number = 1
    while (directory / f"take_{number}").exists():
        number += 1
    destination = directory / f"take_{number}"
    with TemporaryDirectory(dir=directory) as temporary:
        stage = Path(temporary) / "recording"
        stage.mkdir()
        for i, frame in enumerate(frames):
            if not cv2.imwrite(str(stage / f"frame_{i:02d}.png"), frame):
                raise OSError("Could not save camera frame")
        stage.rename(destination)
    return f"Saved {destination}"


def main():
    parser = argparse.ArgumentParser(description="Record mouth clips for one word.")
    parser.add_argument("word")
    parser.add_argument("--output", type=Path, default=DATA_DIR)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.word):
        parser.error("Word must contain only letters, numbers, underscores or hyphens")
    print(f"Recording word: {args.word}")
    run_camera(lambda frames: save_take(frames, args.output, args.word), args.camera)


if __name__ == "__main__":
    main()
