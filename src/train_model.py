
import numpy as np
import tensorflow as tf
import os
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Progress bar

# ==== MODEL HYPERPARAMETERS ====
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.001
INPUT_SHAPE = (22, 80, 112, 1)  # (Frames, Height, Width, Channels)

# ==== LOAD DATA ====
PROCESSED_DATA_DIR = "processed_data/"
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

            if frames.shape == (22, 80, 112):  # Ensure correct shape
                frames = np.expand_dims(frames, axis=-1)  # Add channel dimension
                X.append(frames)
                y.append(word_to_index[word])

X = np.array(X)
y = np.array(y)

print(f"✅ Loaded {len(X)} samples across {len(words)} words.")

# Split into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ==== BUILD 3D CNN MODEL ====
def build_3d_cnn(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),

        tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),

        tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create model
model = build_3d_cnn(INPUT_SHAPE, len(words))

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ==== TRAIN THE MODEL WITH PROGRESS BAR & TIME ESTIMATE ====
print("\nTraining model...\n")

for epoch in range(EPOCHS):
    start_time = time.time()  # Track epoch start time

    # Train one epoch with progress bar
    with tqdm(total=len(X_train) // BATCH_SIZE, desc=f"Epoch {epoch + 1}/{EPOCHS}", unit="batch") as pbar:
        for i in range(0, len(X_train), BATCH_SIZE):
            batch_x = X_train[i:i + BATCH_SIZE]
            batch_y = y_train[i:i + BATCH_SIZE]
            model.train_on_batch(batch_x, batch_y)
            pbar.update(1)

    # Calculate time taken for this epoch
    epoch_time = time.time() - start_time
    remaining_time = epoch_time * (EPOCHS - epoch - 1)
    print(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f} seconds.")
    print(f"Estimated time remaining: {remaining_time / 60:.2f} minutes.\n")

# Save model
MODEL_SAVE_PATH = "model/lip_reader_3dcnn.h5"
if not os.path.exists("model"):
    os.makedirs("model")
model.save(MODEL_SAVE_PATH)
print(f"\n✅ Model saved to {MODEL_SAVE_PATH}")

# ==== EVALUATE MODEL ====
test_loss, test_acc = model.evaluate(X_val, y_val)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
