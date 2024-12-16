import cv2
import numpy as np

img = cv2.imread('../img/house.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Shi-Tomasi corner detection method
corners = cv2.goodFeaturesToTrack(gray, 80, 0.01, 10)

# Convert floating-point coordinates to integer coordinates
corners = np.int32(corners)

# Draw circles at corner coordinates
for corner in corners:
    x, y = corner[0]
    cv2.circle(img, (x, y), 5, (0, 0, 255), 1, cv2.LINE_AA)

# Display the result
cv2.imshow('Corners', img)
cv2.waitKey()
cv2.destroyAllWindows()
