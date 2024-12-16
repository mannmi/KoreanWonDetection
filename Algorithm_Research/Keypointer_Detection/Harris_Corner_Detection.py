import cv2
import numpy as np

img = cv2.imread('../img/house.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Harris Corner Detection ---①
corner = cv2.cornerHarris(gray, 2, 3, 0.04)

# Get coordinates where the variation result exceeds 10% of the maximum value ---②
coord = np.where(corner > 0.1 * corner.max())
coord = np.stack((coord[1], coord[0]), axis=-1)

# Draw circles at corner coordinates ---③
for x, y in coord:
    cv2.circle(img, (x, y), 5, (0, 0, 255), 1, cv2.LINE_AA)

# Normalize the variation to 0-255 for visualization ---④
corner_norm = cv2.normalize(corner, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
# Convert to BGR for display
corner_norm = cv2.cvtColor(corner_norm, cv2.COLOR_GRAY2BGR)

# Merge the normalized image and the original image
merged = np.hstack((corner_norm, img))

# Display the result
cv2.imshow('Harris Corner', merged)
cv2.waitKey()
cv2.destroyAllWindows()
