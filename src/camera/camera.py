import cv2


class Camera:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open camera.")

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Failed to grab frame")
        return frame

    def release(self):
        self.cap.release()
