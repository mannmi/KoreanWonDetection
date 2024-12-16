import cv2
import numpy as np
from google.colab.patches import cv2_imshow

from google.colab import files

# Load images
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector
orb = cv2.ORB_create()

# Find keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Use BFMatcher to find matches between descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw top 10 matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Extract location of good matches
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Find the homography matrix using RANSAC
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

# Use the homography matrix to warp img1 to match img2
height, width, _ = img2.shape
img1_aligned = cv2.warpPerspective(img1, H, (width, height))

# Show the aligned image and matches (using cv2_imshow for Colab)
cv2_imshow(img1_aligned)  # This will display the aligned image
cv2_imshow(img_matches)   # This will display the matched keypoints

# Save the output images
cv2.imwrite('aligned_image.jpg', img1_aligned)
cv2.imwrite('matches.jpg', img_matches)

# Download the saved images to your local system
files.download('matches.jpg')