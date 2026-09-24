from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = ROOT / "processed_data"
MODEL_DIR = ROOT / "model"
RUN_DIR = ROOT / "runs" / "lip-reader"
LANDMARKS = MODEL_DIR / "shape_predictor_68_face_landmarks.dat"
FRAME_COUNT = 22
IMAGE_SIZE = (112, 80)
INPUT_SHAPE = (FRAME_COUNT, IMAGE_SIZE[1], IMAGE_SIZE[0], 1)
PREPROCESSING_VERSION = 1
