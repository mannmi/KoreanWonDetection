import unittest
from src.camera.camera import Camera


class TestCamera(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(camera_index=0)

    def test_camera_initialization(self):
        self.assertTrue(self.camera.cap.isOpened(), "Camera should be initialized and opened.")

    def test_read_frame(self):
        frame = self.camera.read_frame()
        self.assertIsNotNone(frame, "Frame should not be None.")
        self.assertEqual(len(frame.shape), 3, "Frame should be a color image.")

    def tearDown(self):
        self.camera.release()


if __name__ == '__main__':
    unittest.main()
