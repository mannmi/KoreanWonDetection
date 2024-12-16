import unittest
import cv2
import numpy as np
from src.edge_detection.EdgeDetector import EdgeDetection


class TestEdgeDetection(unittest.TestCase):
    def setUp(self):
        self.edge_detection = EdgeDetection(method='canny')
        self.test_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(self.test_image, (25, 25), (75, 75), 255, -1)

    def test_edge_detection(self):
        edges = self.edge_detection.detect_edges(self.test_image)
        self.assertIsNotNone(edges, "Edges should not be None.")


if __name__ == '__main__':
    unittest.main()