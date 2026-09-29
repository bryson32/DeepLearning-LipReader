import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import tensorflow as tf

from config import INPUT_SHAPE, PREPROCESSING_VERSION


def save_run(model, directory, metadata):
    directory = Path(directory)
    if directory.exists():
        raise ValueError(f"Run already exists: {directory}. Choose a new output directory.")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory.parent) as temporary:
        stage = Path(temporary) / "run"
        stage.mkdir()
        model_path = stage / "model.keras"
        model.save(model_path)
        data = dict(metadata, input_shape=list(INPUT_SHAPE), preprocessing=PREPROCESSING_VERSION,
                    tensorflow=tf.__version__, model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest())
        (stage / "metadata.json").write_text(json.dumps(data, indent=2) + "\n")
        stage.rename(directory)


def load_run(directory):
    directory = Path(directory)
    model_path, metadata_path = directory / "model.keras", directory / "metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"No trained run at {directory}. Run train_model.py first.")
    metadata = json.loads(metadata_path.read_text())
    labels = metadata.get("labels")
    if not isinstance(labels, list) or len(labels) < 2 or not all(isinstance(x, str) and x for x in labels):
        raise ValueError("Run must contain an ordered word list")
    if len(set(labels)) != len(labels):
        raise ValueError("Run contains duplicate word labels")
    if metadata.get("input_shape") != list(INPUT_SHAPE) or metadata.get("preprocessing") != PREPROCESSING_VERSION:
        raise ValueError("Run uses incompatible input preprocessing; retrain the model")
    if metadata.get("model_sha256") != hashlib.sha256(model_path.read_bytes()).hexdigest():
        raise ValueError("Model does not match its metadata")
    model = tf.keras.models.load_model(model_path, compile=False)
    if tuple(model.input_shape[1:]) != INPUT_SHAPE or model.output_shape[-1] != len(labels):
        raise ValueError("Model shape does not match its metadata")
    return model, metadata
