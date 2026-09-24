import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from config import DATA_DIR, PREPROCESSING_VERSION, PROCESSED_DIR
from images import load_take


def process_dataset(source, output):
    source, output = Path(source), Path(output)
    if not source.is_dir():
        raise ValueError(f"Recording directory not found: {source}")
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new directory.")
    takes = [(word.name, take) for word in sorted(source.iterdir())
             if word.is_dir() and not word.name.startswith(".")
             for take in sorted(word.glob("take_*")) if take.is_dir()]
    if not takes:
        raise ValueError(f"No recordings found in {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "processed"
        stage.mkdir()
        for word, take in takes:
            destination = stage / word
            destination.mkdir(exist_ok=True)
            np.save(destination / f"{take.name}.npy", load_take(take))
        (stage / "preprocessing.json").write_text(json.dumps({"version": PREPROCESSING_VERSION}) + "\n")
        stage.rename(output)
    print(f"Processed {len(takes)} recordings into {output}")


def main():
    parser = argparse.ArgumentParser(description="Prepare recorded mouth clips for training.")
    parser.add_argument("--input", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    process_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
