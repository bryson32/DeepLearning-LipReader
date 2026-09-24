import cv2
import numpy as np
import tensorflow as tf
import dlib
import os

from config import DATA_DIR, LANDMARKS, MODEL_DIR, PROCESSED_DIR

MODEL_PATH = MODEL_DIR / "lip_reader_3dcnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print(f"\nLoaded model from {MODEL_PATH}")

PROCESSED_DATA_DIR = PROCESSED_DIR
words = sorted(os.listdir(PROCESSED_DATA_DIR))
word_to_index = {word: i for i, word in enumerate(words)}
index_to_word = {i: word for word, i in word_to_index.items()}

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(LANDMARKS))

cap = cv2.VideoCapture(0)
print("\nPress 'L' to start prediction, 'Q' to exit...")

frames = []
FRAME_COUNT = 22
recording = False
predicted_word = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) > 0:
        for face in faces:
            landmarks = predictor(gray, face)

            x_min = min([landmarks.part(i).x for i in range(48, 68)])
            x_max = max([landmarks.part(i).x for i in range(48, 68)])
            y_min = min([landmarks.part(i).y for i in range(48, 68)])
            y_max = max([landmarks.part(i).y for i in range(48, 68)])

            x_min = max(0, x_min - 10)
            x_max = min(frame.shape[1], x_max + 10)
            y_min = max(0, y_min - 10)
            y_max = min(frame.shape[0], y_max + 10)

            box_width = x_max - x_min
            box_height = y_max - y_min

            if box_width > box_height:
                center_y = (y_min + y_max) // 2
                y_min = max(0, center_y - box_width // 2)
                y_max = min(frame.shape[0], center_y + box_width // 2)
            else:
                center_x = (x_min + x_max) // 2
                x_min = max(0, center_x - box_height // 2)
                x_max = min(frame.shape[1], center_x + box_height // 2)

            lip_region = frame[y_min:y_max, x_min:x_max]
            lip_region = cv2.resize(lip_region, (112, 80))

            gray_lip = cv2.cvtColor(lip_region, cv2.COLOR_BGR2GRAY)

            blurred = cv2.GaussianBlur(gray_lip, (5, 5), 0)

            min_pixel = np.min(blurred)
            max_pixel = np.max(blurred)
            contrast_stretched = (blurred - min_pixel) / (max_pixel - min_pixel + 1e-5) * 255
            contrast_stretched = contrast_stretched.astype(np.uint8)

            bilateral_filtered = cv2.bilateralFilter(contrast_stretched, 5, 75, 75)

            sharpen_kernel = np.array([[-1, -1, -1],
                                       [-1,  9, -1],
                                       [-1, -1, -1]])
            sharpened = cv2.filter2D(bilateral_filtered, -1, sharpen_kernel)

            final_processed = cv2.GaussianBlur(sharpened, (3, 3), 0)

            normalized = final_processed / 255.0
            frames.append(normalized)

            if recording and len(frames) == FRAME_COUNT:
                input_sequence = np.array(frames, dtype=np.float32)
                input_sequence = np.expand_dims(input_sequence, axis=0)
                input_sequence = np.expand_dims(input_sequence, axis=-1)

                prediction = model.predict(input_sequence)
                predicted_index = np.argmax(prediction)
                predicted_word = index_to_word[predicted_index]

                print(f"\nPredicted Word: {predicted_word}")

                frames = []
                recording = False

    else:
        print("No face detected, skipping frame.")

    if predicted_word:
        cv2.putText(frame, f"Predicted: {predicted_word}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Lip Reader - Press 'L' to start", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('l') and not recording:
        print(f"\nRecording started")
        recording = True
        frames = []

cap.release()
cv2.destroyAllWindows()
print("\nLive Lip Reading Stopped.")
