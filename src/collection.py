import cv2
import dlib
import os
import time

# ==== SET THE WORD HERE ====
WORD = "panda"  # Change this before running the script

# Load Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

# Constants
OUTPUT_DIR = "data/"
FRAMES_PER_WORD = 22  # Fixed number of frames per take

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Start capturing video (Change 1 to your correct external webcam index)
cap = cv2.VideoCapture(0)  # Change this if needed

print(f"\nRecording word: '{WORD}'. Press 'L' to start recording.")
print("Press 'Q' to quit.")

recording = False
frame_count = 0
take_number = 1  # Start take count

# Find the next available take number
word_dir = os.path.join(OUTPUT_DIR, WORD)
if not os.path.exists(word_dir):
    os.makedirs(word_dir)

while os.path.exists(os.path.join(word_dir, f"take_{take_number}")):
    take_number += 1  # Increment take number until an available one is found

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert to grayscale for better face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)


        x_min = min([landmarks.part(i).x for i in range(48, 68)])
        x_max = max([landmarks.part(i).x for i in range(48, 68)])
        y_min = min([landmarks.part(i).y for i in range(48, 68)])
        y_max = max([landmarks.part(i).y for i in range(48, 68)])
        
        # Expand bounding box dynamically to avoid losing lips
        EXPAND_RATIO = 1.3  # Adjust this value if needed

        
        lip_width = x_max - x_min
        lip_height = y_max - y_min

        # Calculate new bounding box with expansion
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2

        x_min = max(0, int(x_center - (lip_width // 2) * EXPAND_RATIO))
        x_max = min(frame.shape[1], int(x_center + (lip_width // 2) * EXPAND_RATIO))
        y_min = max(0, int(y_center - (lip_height // 2) * EXPAND_RATIO))
        y_max = min(frame.shape[0], int(y_center + (lip_height // 2) * EXPAND_RATIO))

        # Draw rectangle around lips
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # If recording, save frames
        if recording and frame_count < FRAMES_PER_WORD:
            lip_region = frame[y_min:y_max, x_min:x_max]
            lip_region = cv2.resize(lip_region, (112, 80))

            # Create the take folder only once per recording session
            take_dir = os.path.join(word_dir, f"take_{take_number}")
            if not os.path.exists(take_dir):
                os.makedirs(take_dir)

            # Save the frame inside the take folder
            frame_path = os.path.join(take_dir, f"frame_{frame_count}.png")
            cv2.imwrite(frame_path, lip_region)
            frame_count += 1

            # Stop recording after collecting enough frames
            if frame_count >= FRAMES_PER_WORD:
                print(f"✅ Recorded {FRAMES_PER_WORD} frames for '{WORD}', saved in '{take_dir}'. Ready for next take.")
                recording = False
                frame_count = 0  # Reset for next take
                take_number += 1  # Move to next take

    # Display webcam feed with lip tracking
    cv2.imshow(f"Lip Reader - Recording '{WORD}' (Press 'L' to start)", frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break  # Quit if 'q' is pressed
    elif key == ord('l') and not recording:
        print(f"Recording '{WORD}'... Speak now!")
        recording = True
        frame_count = 0  # Reset frame count

# Cleanup
cap.release()
cv2.destroyAllWindows()
