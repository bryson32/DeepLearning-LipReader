import cv2
import os
import numpy as np

from config import DATA_DIR, PROCESSED_DIR

INPUT_DIR = DATA_DIR
OUTPUT_DIR = PROCESSED_DIR

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

words = sorted(os.listdir(INPUT_DIR))

for word in words:
    word_path = os.path.join(INPUT_DIR, word)

    if not os.path.isdir(word_path):
        continue

    print(f"Processing word: {word}")

    word_output_path = os.path.join(OUTPUT_DIR, word)
    if not os.path.exists(word_output_path):
        os.makedirs(word_output_path)

    takes = sorted(os.listdir(word_path))

    for take in takes:
        take_path = os.path.join(word_path, take)
        if not os.path.isdir(take_path):
            continue

        print(f"Processing take: {take}")

        frames = []
        frame_files = sorted(os.listdir(take_path))

        for frame_file in frame_files:
            frame_path = os.path.join(take_path, frame_file)
            image = cv2.imread(frame_path)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

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

            frames.append(final_processed)

        frames = np.array(frames, dtype=np.float32) / 255.0
        npy_path = os.path.join(word_output_path, f"{take}.npy")
        np.save(npy_path, frames)

print("\nPreprocessing complete! Processed data saved in 'processed_data/'")
