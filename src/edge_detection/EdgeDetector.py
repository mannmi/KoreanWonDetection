import cv2
import numpy as np

class EdgeDetection:
    def __init__(self, method='canny'):
        self.method = method

    def set_method(self, method):
        self.method = method

    def detect_edges(self, gray_frame):
        if self.method == 'canny':
            return self.canny_edge_detection(gray_frame)
        elif self.method == 'sobel':
            return self.sobel_edge_detection(gray_frame)
        elif self.method == 'laplacian':
            return self.laplacian_edge_detection(gray_frame)
        else:
            raise ValueError(f"Unknown edge detection method: {self.method}")

    def canny_edge_detection(self, gray_frame):
        edges = cv2.Canny(gray_frame, 50, 150)
        return edges

    def sobel_edge_detection(self, gray_frame):
        grad_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        return edges

    def laplacian_edge_detection(self, gray_frame):
        edges = cv2.Laplacian(gray_frame, cv2.CV_64F)
        edges = cv2.convertScaleAbs(edges)
        return edges

# Example usage:
if __name__ == "__main__":
    edge_detector = EdgeDetection(method='canny')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = edge_detector.detect_edges(gray)

        cv2.imshow('Edges', edges)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            edge_detector.set_method('canny')
        elif key == ord('s'):
            edge_detector.set_method('sobel')
        elif key == ord('l'):
            edge_detector.set_method('laplacian')

    cap.release()
    cv2.destroyAllWindows()
