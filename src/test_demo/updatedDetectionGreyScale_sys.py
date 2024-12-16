# updatedDetectionGreyScale_sys.py
import cv2
import numpy as np
import time

from src.camera.camera import Camera

# Load the template image
template = cv2.imread('../../images/won_1000.jpg', 0)
if template is None:
    print("Error: Template image not found or cannot be opened.")
    exit()

# Convert template to grayscale if needed
if len(template.shape) == 3:
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Initialize the ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB in the template
kp_template, des_template = orb.detectAndCompute(template, None)

# Use the common camera opening function
cap = Camera()

# record the start time
start_time = time.time()

while True:
    # Capture frame-by-frame
    frame = cap.read_frame()

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
    frame_matches = cv2.drawMatches(template, kp_template, frame, kp_frame, good_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # If enough good matches are found, find the homography
    if len(good_matches) > 60:
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        h, w = template.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # extract selected area (ROI)
        x, y, w, h = cv2.boundingRect(np.int32(dst))
        roi = gray[y:y + h, x:x + w]

        # separate from 02, 03 and 06 areas, these numbers related to hidden images
        roi_02 = roi[:, 0:200]
        roi_03 = roi[:, 200:400]
        roi_06 = roi[:, 400:600]

        # 각 ROI에 대한 이미지 처리 및 분석 수행
        # part 02
        template_02 = template[:, 0:200]
        if len(template_02.shape) == 3:
            template_02 = cv2.cvtColor(template_02, cv2.COLOR_BGR2GRAY)
        res_02 = cv2.matchTemplate(roi_02, template_02, cv2.TM_CCOEFF_NORMED)
        threshold_02 = 0.8
        loc_02 = np.where(res_02 >= threshold_02)
        if len(loc_02[0]) > 0:
            print("02 part matched")
        else:
            print("02 part not matched")

        # part 03
        template_03 = template[:, 200:400]
        if len(template_03.shape) == 3:
            template_03 = cv2.cvtColor(template_03, cv2.COLOR_BGR2GRAY)
        res_03 = cv2.matchTemplate(roi_03, template_03, cv2.TM_CCOEFF_NORMED)
        threshold_03 = 0.8
        loc_03 = np.where(res_03 >= threshold_03)
        if len(loc_03[0]) > 0:
            print("03 part matched")
        else:
            print("03 part not matched")

        # part 06
        template_06 = template[:, 400:600]
        if len(template_06.shape) == 3:
            template_06 = cv2.cvtColor(template_06, cv2.COLOR_BGR2GRAY)

        # resize part 06
        if roi_06.shape[0] < template_06.shape[0] or roi_06.shape[1] < template_06.shape[1]:
            roi_06 = cv2.resize(roi_06, template_06.shape[:2], interpolation=cv2.INTER_AREA)

        res_06 = cv2.matchTemplate(roi_06, template_06, cv2.TM_CCOEFF_NORMED)
        threshold_06 = 0.8
        loc_06 = np.where(res_06 >= threshold_06)
        if len(loc_06[0]) > 0:
            print("06 part matched")
        else:
            print("06 part not matched")

        frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
        print("Match found!")
    else:
        print("Not matching")

    # Display the video feed with detected template
    cv2.imshow('Detected Template', frame_matches)

    # if 10 seconds passed, terminate the program
    elapsed_time = time.time() - start_time
    if elapsed_time > 5:
        break

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
