import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from artifacts import load_run
from config import RUN_DIR
from dataset import load_dataset


def evaluate(run, data):
    model, metadata = load_run(run)
    x, y, labels, hashes = load_dataset(data, metadata["labels"])
    if not metadata.get("training_hashes") or not metadata.get("validation_hashes"):
        raise ValueError("Run is missing its training and validation split")
    used = set(metadata["training_hashes"]) | set(metadata["validation_hashes"])
    if used.intersection(hashes):
        raise ValueError("Evaluation includes training or validation clips. Record a separate session.")
    scores = model.predict(x, batch_size=16, verbose=0)
    if not np.isfinite(scores).all():
        raise ValueError("Model returned non-finite scores")
    predicted = scores.argmax(axis=1)
    indices = list(range(len(labels)))
    return {
        "labels": labels, "clips": len(y), "model_sha256": metadata["model_sha256"],
        "clip_hashes": hashes, "accuracy": float(accuracy_score(y, predicted)),
        "classification_report": classification_report(y, predicted, labels=indices,
                                                       target_names=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, predicted, labels=indices).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on separately recorded, preprocessed clips.")
    parser.add_argument("--run", type=Path, default=RUN_DIR)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new file")
    result = evaluate(args.run, args.data)
    if args.output is not None:
        with args.output.open("x") as file:
            file.write(json.dumps(result, indent=2) + "\n")
        print(f"Saved evaluation to {args.output}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Macro F1: {result['classification_report']['macro avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()
