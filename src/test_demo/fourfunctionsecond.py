import cv2
import numpy as np
from sklearn.cluster import KMeans

# Initialize ORB and SIFT Detectors
orb = cv2.ORB_create()
sift = cv2.SIFT_create()


def detect_keypoints_and_descriptors(image, method="ORB"):
    """
    Detect keypoints and compute descriptors using ORB or SIFT.
    """
    if method == "ORB":
        keypoints, descriptors = orb.detectAndCompute(image, None)
    elif method == "SIFT":
        keypoints, descriptors = sift.detectAndCompute(image, None)
    else:
        raise ValueError("Invalid method. Choose 'ORB' or 'SIFT'.")
    return keypoints, descriptors


def match_keypoints(des_template, des_frame, method="ORB"):
    """
    Match descriptors using BFMatcher with Lowe's ratio test.
    """
    norm_type = cv2.NORM_HAMMING if method == "ORB" else cv2.NORM_L2
    bf = cv2.BFMatcher(norm_type)
    matches = bf.knnMatch(des_template, des_frame, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return good_matches


def find_homography(template, kp_template, frame, kp_frame, good_matches):
    """
    Find homography and draw bounding box if enough matches are found.
    """
    if len(good_matches) > 10:  # Threshold for homography
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            h, w = template.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            print("Homography computed and bounding box drawn.")
        return frame
    return frame


def cluster_and_visualize_keypoints(kp_template, kp_frame, good_matches, frame):
    """
    Cluster matching keypoints and visualize matches with different colors.
    """
    src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches])
    all_pts = np.vstack((src_pts, dst_pts))

    if all_pts.shape[0] < 2:
        print("Not enough valid points for clustering. Skipping clustering visualization.")
        return

    n_clusters = min(len(all_pts), 5)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_pts)
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(n_clusters)]
    cluster_labels = kmeans.labels_

    visualization = frame.copy()
    for i, match in enumerate(good_matches):
        pt1 = tuple(map(int, src_pts[i]))
        pt2 = tuple(map(int, dst_pts[i]))
        color = colors[cluster_labels[i]]
        cv2.line(visualization, pt1, pt2, color, 2)

    cv2.imshow("Clustered Matches", visualization)


def dense_optical_flow(frame1, frame2):
    """
    Visualize dense optical flow between two frames.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    visualization = frame1.copy()
    step = 16
    for y in range(0, flow.shape[0], step):
        for x in range(0, flow.shape[1], step):
            fx, fy = flow[y, x]
            cv2.arrowedLine(visualization, (x, y), (int(x + fx), int(y + fy)), (0, 255, 0), 1)

    cv2.imshow("Dense Optical Flow", visualization)
    return visualization


def saliency_based_matching(frame, keypoints, matches):
    """
    Overlay matches on a saliency map for visualization.
    """
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(frame)
    saliency_map = (saliency_map * 255).astype("uint8")
    visualization = cv2.cvtColor(saliency_map, cv2.COLOR_GRAY2BGR)

    for match in matches:
        pt1 = tuple(map(int, keypoints[match.queryIdx].pt))
        pt2 = tuple(map(int, keypoints[match.trainIdx].pt))
        cv2.line(visualization, pt1, pt2, (0, 255, 0), 2)

    cv2.imshow("Saliency-Based Matching", visualization)

# Main script
template_path = '../../images/won_1000.jpg'
template = cv2.imread(template_path, 0)
if template is None:
    print("Error: Template image not found or cannot be opened.")
    exit()

kp_template, des_template = detect_keypoints_and_descriptors(template, method="SIFT")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

previous_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect and match keypoints
    kp_frame, des_frame = detect_keypoints_and_descriptors(gray, method="SIFT")
    if des_frame is not None:
        good_matches = match_keypoints(des_template, des_frame, method="SIFT")

        # Compute and draw homography if matches are sufficient
        frame = find_homography(template, kp_template, frame, kp_frame, good_matches)

        # Visualize matched keypoints in a separate window
        if good_matches:
            match_img = cv2.drawMatches(template, kp_template, frame, kp_frame, good_matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Matched Keypoints", match_img)

        # Side-by-side display of template and frame
        resized_template = cv2.resize(template, (frame.shape[1], frame.shape[0]))
        combined_view = cv2.hconcat([cv2.cvtColor(resized_template, cv2.COLOR_GRAY2BGR), frame])
        cv2.imshow("Side-by-Side Display", combined_view)

        # Optional: Clustered keypoints visualization
        cluster_and_visualize_keypoints(kp_template, kp_frame, good_matches, frame)

        # Optional: Saliency-based matching visualization
        saliency_based_matching(frame, kp_frame, good_matches)

    # Visualize dense optical flow if a previous frame exists
    if previous_frame is not None:
        dense_optical_flow(previous_frame, frame)

    previous_frame = frame.copy()

    # Display the frame with bounding box
    cv2.imshow("Detected Template", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

