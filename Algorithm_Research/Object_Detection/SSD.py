import cv2
import numpy as np

# Load the SSD model
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

# Class labels for SSD MobileNet
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Load the image
image = cv2.imread('../img/street.jpg')
(h, w) = image.shape[:2]

# Prepare the image for SSD
blob = cv2.dnn.blobFromImage(image, scalefactor=0.007843, size=(300, 300), mean=(127.5, 127.5, 127.5), swapRB=True, crop=False)
net.setInput(blob)

# Perform detection
detections = net.forward()

# Draw detected objects
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.2:  # Confidence threshold
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        label = f"{CLASSES[idx]}: {confidence:.2f}"
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the result
cv2.imshow('SSD Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
