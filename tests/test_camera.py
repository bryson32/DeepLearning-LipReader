from contextlib import ExitStack
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from camera import run_camera
from config import FRAME_COUNT


class CameraTests(TestCase):
    def run_feed(self, keys, faces=None, consume=None):
        cap = Mock()
        cap.isOpened.return_value = True
        cap.read.side_effect = lambda: (True, np.zeros((100, 150, 3), np.uint8))
        crop = np.zeros((80, 112, 3), np.uint8)
        detector = Mock(side_effect=faces) if faces is not None else Mock(return_value=[object()])
        received = []
        if consume is None:
            def consume(frames):
                received.append(list(frames))
                return "Recorded"
        with ExitStack() as stack:
            for name, value in {"VideoCapture": cap, "waitKey": None, "imshow": None,
                                "destroyAllWindows": None}.items():
                mocked = stack.enter_context(patch(f"camera.cv2.{name}", return_value=value))
                if name == "waitKey":
                    mocked.side_effect = keys
            stack.enter_context(patch("camera.dlib.get_frontal_face_detector", return_value=detector))
            stack.enter_context(patch("camera.dlib.shape_predictor"))
            stack.enter_context(patch("camera.crop_mouth", return_value=(crop.copy(), (1, 1, 20, 20))))
            try:
                run_camera(consume)
            finally:
                cap.release.assert_called_once()
        return received

    def test_idle_frames_do_not_enter_next_recording(self):
        keys = [-1] * 100 + [ord("l")] + [-1] * (FRAME_COUNT - 1) + [ord("q")]
        clips = self.run_feed(keys)
        self.assertEqual(len(clips), 1)
        self.assertEqual(len(clips[0]), FRAME_COUNT)

    def test_face_loss_cancels_partial_recording(self):
        keys = [ord("l")] + [-1] * 4 + [ord("q")]
        self.assertEqual(self.run_feed(keys, [[object()]] * 3 + [[]] * 3), [])

    def test_multiple_faces_do_not_mix_into_a_clip(self):
        self.assertEqual(self.run_feed([ord("l"), ord("q")], [[object(), object()]] * 2), [])

    def test_consumer_failure_still_releases_camera(self):
        keys = [ord("l")] + [-1] * FRAME_COUNT
        with self.assertRaisesRegex(ValueError, "failed"):
            self.run_feed(keys, consume=Mock(side_effect=ValueError("failed")))
