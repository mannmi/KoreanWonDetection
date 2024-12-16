import cv2

image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
keypoints = sift.detect(image, None)
output = cv2.drawKeypoints(image, keypoints, None)
cv2.imshow("SIFT Keypoints", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
