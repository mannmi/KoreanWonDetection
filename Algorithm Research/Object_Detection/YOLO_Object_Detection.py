import cv2
import numpy as np

# Load YOLO model and class labels
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Load the image
img = cv2.imread('../img/street.jpg')
height, width, _ = img.shape

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get YOLO layer names and output
layer_names = net.getUnconnectedOutLayersNames()
layer_outputs = net.forward(layer_names)

# Process YOLO outputs
boxes, confidences, class_ids = [], [], []
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Confidence threshold
            box = detection[0:4] * np.array([width, height, width, height])
            (center_x, center_y, w, h) = box.astype("int")
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Max Suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw detected objects
for i in indices:
    i = i[0]
    box = boxes[i]
    (x, y, w, h) = box
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the result
cv2.imshow('YOLO Object Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
