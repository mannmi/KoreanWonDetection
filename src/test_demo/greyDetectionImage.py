import cv2
import numpy as np

# Load the template image
template = cv2.imread('../../images/won_1000.jpg', 0)
if template is None:
    print("Error: Template image not found or cannot be opened.")
    exit()


# w, h = template.shape[::-1]

# Access the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Increase contrast by applying histogram equalization
    high_contrast = cv2.equalizeHist(gray)

    # Perform edge detection
    edges = cv2.Canny(high_contrast, 50, 150)


    # Display the video feed with detected template
    cv2.imshow('Grayscale', gray)
    cv2.imshow('High Contrast', high_contrast)
    cv2.imshow('Edges', edges)
    # cv2.imshow('Detected Template', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()