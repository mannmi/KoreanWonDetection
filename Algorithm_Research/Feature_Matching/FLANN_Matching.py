import cv2
import numpy as np

img1 = cv2.imread('../img/taekwonv1.jpg')
img2 = cv2.imread('../img/figures.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Create SIFT detector
detector = cv2.xfeatures2d.SIFT_create()
# Extract keypoints and descriptors
kp1, desc1 = detector.detectAndCompute(gray1, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)

# Set index parameters and search parameters ---①
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# Create Flann-based matcher ---③
matcher = cv2.FlannBasedMatcher(index_params, search_params)
# Compute matches ---④
matches = matcher.match(desc1, desc2)
# Draw matches
res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# Display the result
cv2.imshow('Flann + SIFT', res)
cv2.waitKey()
cv2.destroyAllWindows()
