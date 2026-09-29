import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from artifacts import load_run
from camera import run_camera
from config import INPUT_SHAPE, RUN_DIR
from images import load_take, prepare_sequence


def predict_sequence(model, labels, sequence):
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.shape != INPUT_SHAPE[:-1] or not np.isfinite(sequence).all():
        raise ValueError("Expected a finite 22x80x112 clip")
    started = perf_counter()
    probabilities = model(sequence[None, ..., None], training=False).numpy()[0]
    elapsed = (perf_counter() - started) * 1000
    if not np.isfinite(probabilities).all():
        raise ValueError("Model returned non-finite scores")
    index = int(np.argmax(probabilities))
    return {"word": labels[index], "score": float(probabilities[index]), "inference_ms": elapsed}


def main():
    parser = argparse.ArgumentParser(description="Classify a recorded clip or use the webcam.")
    parser.add_argument("--run", type=Path, default=RUN_DIR)
    parser.add_argument("--take", type=Path)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    model, metadata = load_run(args.run)
    labels = metadata["labels"]
    if args.take is not None:
        print(json.dumps(predict_sequence(model, labels, load_take(args.take)), indent=2))
        return

    def consume(frames):
        result = predict_sequence(model, labels, prepare_sequence(frames))
        return f"{result['word']} (score {result['score']:.2f})"

    run_camera(consume, args.camera)


if __name__ == "__main__":
    main()
