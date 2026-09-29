import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from artifacts import save_run
from config import INPUT_SHAPE, PROCESSED_DIR, RUN_DIR
from dataset import load_dataset
from network import build_3d_cnn


def train(data, output, epochs=20, batch_size=16, seed=42):
    if Path(output).exists():
        raise ValueError(f"Run already exists: {output}. Choose a new output directory.")
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    x, y, labels, hashes = load_dataset(data)
    if len(labels) < 2 or np.bincount(y, minlength=len(labels)).min() < 5:
        raise ValueError("Record at least five takes each for at least two words")
    tf.keras.utils.set_random_seed(seed)
    training, validation = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    model = build_3d_cnn(INPUT_SHAPE, len(labels))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(f"Training on {len(training)} clips; validating on {len(validation)} clips")
    print(f"Model parameters: {model.count_params():,}")
    history = model.fit(x[training], y[training], validation_data=(x[validation], y[validation]),
                        epochs=epochs, batch_size=batch_size, verbose=2,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                                    restore_best_weights=True)])
    loss, accuracy = model.evaluate(x[validation], y[validation], verbose=0)
    metadata = {
        "labels": labels, "seed": seed, "epochs_requested": epochs,
        "batch_size": batch_size, "learning_rate": 0.0003, "parameters": model.count_params(),
        "training_hashes": [hashes[i] for i in training],
        "validation_hashes": [hashes[i] for i in validation],
        "history": history.history, "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
        "validation": {"loss": float(loss), "accuracy": float(accuracy)},
    }
    save_run(model, output, metadata)
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Saved run to {output}")
    return model, metadata


def main():
    parser = argparse.ArgumentParser(description="Train a small vocabulary lip classifier.")
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output", type=Path, default=RUN_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
