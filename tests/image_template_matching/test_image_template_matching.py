import unittest
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

from src.image_template_matching.TemplateMatcher import TemplateMatcher


class TestTemplateMatcher(unittest.TestCase):
    @patch('cv2.VideoCapture')
    @patch('cv2.imread')
    def setUp(self, mock_imread, mock_VideoCapture):
        # Mock the template image
        self.mock_template = np.zeros((100, 100), dtype=np.uint8)
        mock_imread.return_value = self.mock_template

        # Mock the camera
        self.mock_cap = MagicMock()
        self.mock_cap.isOpened.return_value = True
        mock_VideoCapture.return_value = self.mock_cap

        self.matcher = TemplateMatcher('fake_path.jpg')

        # Mock the ORB detector
        self.mock_orb = MagicMock()
        self.matcher.orb = self.mock_orb

        # Mock the BFMatcher
        self.mock_bf = MagicMock()
        self.matcher.bf = self.mock_bf

    def test_init_template_not_found(self):
        with patch('cv2.imread', return_value=None):
            with self.assertRaises(ValueError):
                TemplateMatcher('fake_path.jpg')

    def test_init_camera_not_opened(self):
        with patch('cv2.VideoCapture') as mock_VideoCapture:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_VideoCapture.return_value = mock_cap
            with self.assertRaises(ValueError):
                TemplateMatcher('fake_path.jpg')



if __name__ == '__main__':
    unittest.main()
