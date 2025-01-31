import cv2
import os
import numpy as np

# Input and output directories
INPUT_DIR = "data/"
OUTPUT_DIR = "processed_data/"

# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Get the list of words
words = sorted(os.listdir(INPUT_DIR))

for word in words:
    word_path = os.path.join(INPUT_DIR, word)
    
    if not os.path.isdir(word_path):
        continue  # Skip if not a directory

    print(f"Processing word: {word}")

    # Create output directory for this word
    word_output_path = os.path.join(OUTPUT_DIR, word)
    if not os.path.exists(word_output_path):
        os.makedirs(word_output_path)

    # Process each take
    takes = sorted(os.listdir(word_path))

    for take in takes:
        take_path = os.path.join(word_path, take)
        if not os.path.isdir(take_path):
            continue  # Skip if not a directory

        print(f"  → Processing take: {take}")

        frames = []
        frame_files = sorted(os.listdir(take_path))

        for frame_file in frame_files:
            frame_path = os.path.join(take_path, frame_file)
            image = cv2.imread(frame_path)

            # === Step 1: Convert to Grayscale ===
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # === Step 2: Gaussian Blurring (Reduce Noise) ===
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # === Step 3: Contrast Stretching (Enhance Visibility) ===
            min_pixel = np.min(blurred)
            max_pixel = np.max(blurred)
            contrast_stretched = (blurred - min_pixel) / (max_pixel - min_pixel + 1e-5) * 255  # Avoid division by zero
            contrast_stretched = contrast_stretched.astype(np.uint8)

            # === Step 4: Bilateral Filtering (Smooth Noise, Keep Edges) ===
            bilateral_filtered = cv2.bilateralFilter(contrast_stretched, 5, 75, 75)

            # === Step 5: Sharpening (Enhance Lip Edges) ===
            sharpen_kernel = np.array([[-1, -1, -1], 
                                       [-1,  9, -1], 
                                       [-1, -1, -1]])
            sharpened = cv2.filter2D(bilateral_filtered, -1, sharpen_kernel)

            # === Step 6: Final Gaussian Blurring (Prevent Over-Sharpening Artifacts) ===
            final_processed = cv2.GaussianBlur(sharpened, (3, 3), 0)

            # Append the processed frame
            frames.append(final_processed)

        # Save as NumPy array
        frames = np.array(frames, dtype=np.float32) / 255.0  # Normalize pixel values
        npy_path = os.path.join(word_output_path, f"{take}.npy")
        np.save(npy_path, frames)

print("\n✅ Preprocessing complete! Processed data saved in 'processed_data/'")
