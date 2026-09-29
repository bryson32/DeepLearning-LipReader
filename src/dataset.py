import hashlib
import json
from pathlib import Path

import numpy as np

from config import INPUT_SHAPE, PREPROCESSING_VERSION


def load_dataset(directory, labels=None):
    directory = Path(directory)
    metadata = directory / "preprocessing.json"
    if not metadata.is_file() or json.loads(metadata.read_text()).get("version") != PREPROCESSING_VERSION:
        raise ValueError(f"Run preprocess.py on fresh recordings first: {directory}")
    words = sorted(p.name for p in directory.iterdir() if p.is_dir() and not p.name.startswith("."))
    if labels is None:
        labels = words
    if set(words) - set(labels):
        raise ValueError("Dataset contains words absent from the model")
    samples, targets, hashes = [], [], []
    for index, word in enumerate(labels):
        files = sorted((directory / word).glob("*.npy"))
        if word in words and not files:
            raise ValueError(f"No clips for {word}")
        for file in files:
            clip = np.load(file, allow_pickle=False)
            if clip.shape != INPUT_SHAPE[:-1] or not np.issubdtype(clip.dtype, np.floating):
                raise ValueError(f"Invalid clip shape or dtype: {file}")
            if not np.isfinite(clip).all() or clip.min() < 0 or clip.max() > 1:
                raise ValueError(f"Clip values must be finite and between 0 and 1: {file}")
            clip = clip.astype(np.float32)
            hashes.append(hashlib.sha256(clip.tobytes()).hexdigest())
            samples.append(clip[..., None])
            targets.append(index)
    if not samples:
        raise ValueError(f"No clips found in {directory}")
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate clips found; remove copied takes before splitting")
    return np.stack(samples), np.array(targets), list(labels), hashes
