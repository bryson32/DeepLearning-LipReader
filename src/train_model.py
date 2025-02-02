
import numpy as np
import tensorflow as tf
import os
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Progress bar

# ==== MODEL HYPERPARAMETERS ====
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.0003
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

# Convert labels to one-hot encoding
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=len(words))
y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=len(words))


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.2,  # Small horizontal shifts
    height_shift_range=0.2,  # Small vertical shifts
    brightness_range=[0.7, 1.3],  # Brightness variation
    horizontal_flip=True,  # Mirror augmentation
    zoom_range=0.2
)
# ==== BUILD 3D CNN MODEL ====
def build_3d_cnn(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv3D(8, (3, 3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=input_shape),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),

        tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),

        tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create model
model = build_3d_cnn(INPUT_SHAPE, len(words))

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# ==== TRAIN THE MODEL WITH PROGRESS BAR & TIME ESTIMATE ====
print("\nTraining model...\n")

history = model.fit(
    X_train, y_train_onehot,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val_onehot),
)

# Save model
MODEL_SAVE_PATH = "model/lip_reader_3dcnn.h5"
if not os.path.exists("model"):
    os.makedirs("model")
model.save(MODEL_SAVE_PATH)
print(f"\n✅ Model saved to {MODEL_SAVE_PATH}")

# ==== EVALUATE MODEL ====
test_loss, test_acc = model.evaluate(X_val, y_val_onehot)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

# ==== PLOT TRAINING PERFORMANCE ====
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 1, figsize=(8, 8))

axs[0].plot(history.history['loss'], label='Training Loss')
axs[0].plot(history.history['val_loss'], label='Validation Loss')
axs[0].legend(loc='upper right')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training and Validation Loss')

axs[1].plot(history.history['accuracy'], label='Training Accuracy')
axs[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axs[1].legend(loc='lower right')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Training and Validation Accuracy')

plt.xlabel('Epoch')
plt.show()