
# updatedDetectionGreyScale.py
import cv2
import numpy as np
import time  # time 모듈 추가
from camera_utils import open_camera

# Load the template image
template = cv2.imread('../../images/euro_5.jpg', 0)
if template is None:
    print("Error: Template image not found or cannot be opened.")
    exit()

# Initialize the ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB in the template
kp_template, des_template = orb.detectAndCompute(template, None)

# Use the common camera opening function
cap = open_camera()

# Record the time
start_time = time.time()

test_score = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the keypoints and descriptors with ORB in the frame
    kp_frame, des_frame = orb.detectAndCompute(gray, None)

    # Match descriptors using BFMatcher with Lowe's ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des_template, des_frame, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Draw matches
    frame_matches = cv2.drawMatches(template, kp_template, frame, kp_frame, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # If enough good matches are found, find the homography
    if len(good_matches) > 70:
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        h, w = template.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
        print("Match found!")
        break
    else:
        print("Not matching")

    # Display the video feed with detected template
    cv2.imshow('Detected Template', frame_matches)


    # Camera closed, if 20 seconds past
    elapsed_time = time.time() - start_time

    if elapsed_time > 20:
        break

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
