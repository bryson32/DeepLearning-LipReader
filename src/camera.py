import cv2
import dlib

from config import FRAME_COUNT, LANDMARKS
from images import crop_mouth


def run_camera(consume, device=0):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(LANDMARKS))
    cap = cv2.VideoCapture(device)
    frames = []
    recording = False
    status = "Press L to record, Q to quit"
    print(status)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {device}")
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Could not read a camera frame")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            crop = None
            if len(faces) == 1:
                try:
                    crop, (x0, y0, x1, y1) = crop_mouth(frame, predictor(gray, faces[0]))
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                except ValueError:
                    status = "Keep your mouth in view"
            else:
                status = "Keep one face in view"
            if crop is None:
                if recording:
                    status = "Recording cancelled. Press L to retry"
                frames.clear()
                recording = False
            elif recording:
                frames.append(crop)
                status = f"Recording {len(frames)}/{FRAME_COUNT}"
                if len(frames) == FRAME_COUNT:
                    status = consume(frames)
                    print(status)
                    frames.clear()
                    recording = False
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.imshow("Lip Reader", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("l") and not recording and crop is not None:
                frames.clear()
                recording = True
    finally:
        cap.release()
        cv2.destroyAllWindows()
