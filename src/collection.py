import cv2
import dlib
import os

from config import DATA_DIR, LANDMARKS, MODEL_DIR, PROCESSED_DIR
import time

WORD = "panda"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(LANDMARKS))

OUTPUT_DIR = DATA_DIR
FRAMES_PER_WORD = 22

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

cap = cv2.VideoCapture(0)

print(f"\nRecording word: '{WORD}'. Press 'L' to start recording.")
print("Press 'Q' to quit.")

recording = False
frame_count = 0
take_number = 1

word_dir = os.path.join(OUTPUT_DIR, WORD)
if not os.path.exists(word_dir):
    os.makedirs(word_dir)

while os.path.exists(os.path.join(word_dir, f"take_{take_number}")):
    take_number += 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        x_min = min([landmarks.part(i).x for i in range(48, 68)])
        x_max = max([landmarks.part(i).x for i in range(48, 68)])
        y_min = min([landmarks.part(i).y for i in range(48, 68)])
        y_max = max([landmarks.part(i).y for i in range(48, 68)])

        EXPAND_RATIO = 1.3

        lip_width = x_max - x_min
        lip_height = y_max - y_min

        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2

        x_min = max(0, int(x_center - (lip_width // 2) * EXPAND_RATIO))
        x_max = min(frame.shape[1], int(x_center + (lip_width // 2) * EXPAND_RATIO))
        y_min = max(0, int(y_center - (lip_height // 2) * EXPAND_RATIO))
        y_max = min(frame.shape[0], int(y_center + (lip_height // 2) * EXPAND_RATIO))

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        if recording and frame_count < FRAMES_PER_WORD:
            lip_region = frame[y_min:y_max, x_min:x_max]
            lip_region = cv2.resize(lip_region, (112, 80))

            take_dir = os.path.join(word_dir, f"take_{take_number}")
            if not os.path.exists(take_dir):
                os.makedirs(take_dir)

            frame_path = os.path.join(take_dir, f"frame_{frame_count}.png")
            cv2.imwrite(frame_path, lip_region)
            frame_count += 1

            if frame_count >= FRAMES_PER_WORD:
                print(f"Recorded {FRAMES_PER_WORD} frames for '{WORD}', saved in '{take_dir}'. Ready for next take.")
                recording = False
                frame_count = 0
                take_number += 1

    cv2.imshow(f"Lip Reader - Recording '{WORD}' (Press 'L' to start)", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('l') and not recording:
        print(f"Recording '{WORD}'... Speak now!")
        recording = True
        frame_count = 0

cap.release()
cv2.destroyAllWindows()
