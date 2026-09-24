import cv2
import numpy as np

from config import IMAGE_SIZE


def crop_mouth(frame, landmarks):
    points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])
    left, top = points.min(axis=0)
    right, bottom = points.max(axis=0)
    if right <= left or bottom <= top:
        raise ValueError("Invalid mouth landmarks")
    width = max((right - left) * 1.3, (bottom - top) * 1.3 * IMAGE_SIZE[0] / IMAGE_SIZE[1])
    height = width * IMAGE_SIZE[1] / IMAGE_SIZE[0]
    cx, cy = (left + right) / 2, (top + bottom) / 2
    x0, x1 = max(0, round(cx - width / 2)), min(frame.shape[1], round(cx + width / 2))
    y0, y1 = max(0, round(cy - height / 2)), min(frame.shape[0], round(cy + height / 2))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Mouth is outside the frame")
    return cv2.resize(frame[y0:y1, x0:x1], IMAGE_SIZE), (x0, y0, x1, y1)


def preprocess_frame(image):
    if image is None or image.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3):
        raise ValueError("Expected an 80x112 color mouth crop")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    low, high = blurred.min(), blurred.max()
    stretched = ((blurred - low) / (float(high) - float(low) + 1e-5) * 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(stretched, 5, 75, 75)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(filtered, -1, kernel)
    result = cv2.GaussianBlur(sharpened, (3, 3), 0)
    return result.astype(np.float32) / 255.0
