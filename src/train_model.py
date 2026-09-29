import numpy as np
import tensorflow as tf
import os
import json
from sklearn.model_selection import train_test_split

from config import MODEL_DIR, PROCESSED_DIR
from network import build_3d_cnn

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.0003
INPUT_SHAPE = (22, 80, 112, 1)

PROCESSED_DATA_DIR = PROCESSED_DIR
words = sorted(os.listdir(PROCESSED_DATA_DIR))
word_to_index = {word: i for i, word in enumerate(words)}

X, y = [], []

print("\nLoading data...")

for word in words:
    word_path = os.path.join(PROCESSED_DATA_DIR, word)

    for take_file in sorted(os.listdir(word_path)):
        if take_file.endswith(".npy"):
            filepath = os.path.join(word_path, take_file)
            frames = np.load(filepath)

            if frames.shape == (22, 80, 112):
                frames = np.expand_dims(frames, axis=-1)
                X.append(frames)
                y.append(word_to_index[word])

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} samples across {len(words)} words.")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=len(words))
y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=len(words))

model = build_3d_cnn(INPUT_SHAPE, len(words))

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

print("\nTraining model...\n")

history = model.fit(
    X_train, y_train_onehot,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val_onehot),
    callbacks=[early_stopping],
    verbose=2,
)

MODEL_SAVE_PATH = MODEL_DIR / "lip_reader.keras"
MODEL_DIR.mkdir(exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"\nModel saved to {MODEL_SAVE_PATH}")

val_loss, val_acc = model.evaluate(X_val, y_val_onehot, verbose=0)
print(f"Validation accuracy: {val_acc:.4f}")
(MODEL_DIR / "history.json").write_text(json.dumps(history.history, indent=2) + "\n")
