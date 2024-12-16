import cv2
import numpy as np
import json


class TemplateMatcher:
    def __init__(self, template_path, camera_index=0):
        # Load the template image in grayscale
        print(f"Loading template image from: {template_path}")  # Debug statement
        self.template = cv2.imread(template_path, 0)
        if self.template is None:
            raise ValueError("Error: Template image not found or cannot be opened.")
        print("Template image loaded successfully")  # Debug statement
        print(f"Template image type: {type(self.template)}")  # Debug statement
        print(f"Template image shape: {self.template.shape}")  # Debug statement

        # Initialize the camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open camera.")

        # Initialize the BFMatcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def process_frame(self, frame, orb):
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect keypoints and descriptors in the frame
        kp_frame, des_frame = orb.detectAndCompute(gray, None)
        if des_frame is None:
            print("No descriptors found in frame.")
            return frame, False

        # Match descriptors between the template and the frame
        matches = self.bf.knnMatch(self.des_template, des_frame, k=2)
        good_matches = [m for m, n in matches if len(matches) > 1 and m.distance < 0.75 * n.distance]
        # Draw matches on the frame
        frame_matches = cv2.drawMatches(self.template, self.kp_template, frame, kp_frame, good_matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Check if there are enough good matches
        if len(good_matches) > 30:  # Increased threshold for good matches
            # Find homography and draw the detected object
            src_pts = np.float32([self.kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # matches_mask = mask.ravel().tolist()

            h, w = self.template.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            print("Match found!")
            print(len(good_matches))
            return frame_matches, True

        return frame_matches, False

    def find_best_parameters(self):
        # Experiment with different ORB parameters
        nfeatures_list = [500, 1000, 1500]
        scaleFactor_list = [1.1, 1.2, 1.3]
        nlevels_list = [8, 10, 12]
        edgeThreshold_list = [15, 31, 50]
        fastThreshold_list = [10, 20, 30]

        best_params = None
        best_matches = 0

        # Loop through all combinations of ORB parameters
        for nfeatures in nfeatures_list:
            for scaleFactor in scaleFactor_list:
                for nlevels in nlevels_list:
                    for edgeThreshold in edgeThreshold_list:
                        for fastThreshold in fastThreshold_list:
                            print(f"Testing ORB parameters: nfeatures={nfeatures}, scaleFactor={scaleFactor}, "
                                  f"nlevels={nlevels}, edgeThreshold={edgeThreshold}, fastThreshold={fastThreshold}")
                            # Create ORB detector with current parameters
                            orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels,
                                                 edgeThreshold=edgeThreshold, firstLevel=0, WTA_K=2,
                                                 scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31,
                                                 fastThreshold=fastThreshold)
                            # Detect keypoints and descriptors in the template
                            self.kp_template, self.des_template = orb.detectAndCompute(self.template, None)
                            if self.des_template is None:
                                print("No descriptors found in template.")
                                continue

                            # Capture a frame from the camera
                            ret, frame = self.cap.read()
                            if not ret:
                                print("Failed to grab frame")
                                continue

                            # Process the frame and display the matches
                            frame_matches, match = self.process_frame(frame, orb)
                            cv2.imshow('Testing ORB Parameters', frame_matches)
                            cv2.waitKey(1)  # Display the frame for a short time

                            # Update the best parameters if a better match is found
                            if match:
                                num_matches = len([m for m, n in self.bf.knnMatch(self.des_template,
                                                                                  orb.detectAndCompute(
                                                                                      cv2.cvtColor(frame,
                                                                                                   cv2.COLOR_BGR2GRAY),
                                                                                      None)[1], k=2) if
                                                   m.distance < 0.75 * n.distance])
                                if num_matches > best_matches:
                                    best_matches = num_matches
                                    best_params = (nfeatures, scaleFactor, nlevels, edgeThreshold, fastThreshold)

        # Check if best_params is not None before accessing its elements
        if best_params is not None:
            print(f"Best ORB parameters: nfeatures={best_params[0]}, scaleFactor={best_params[1]}, "
                  f"nlevels={best_params[2]}, edgeThreshold={best_params[3]}, fastThreshold={best_params[4]}")
            print(f"Best number of matches: {best_matches}")

            with open('../best_orb_params.json', 'w') as f:
                json.dump({
                    'nfeatures': best_params[0],
                    'scaleFactor': best_params[1],
                    'nlevels': best_params[2],
                    'edgeThreshold': best_params[3],
                    'fastThreshold': best_params[4]
                }, f)
        else:
            print("No good parameters found.")

        self.cap.release()
        cv2.destroyAllWindows()

    def load_and_use_parameters(self):
        # Load the best parameters from a file
        with open(r'/src/image_template_matching/best_orb_params.json', 'r') as f:
            params = json.load(f)

        # Create ORB detector with the loaded parameters
        orb = cv2.ORB_create(nfeatures=params['nfeatures'], scaleFactor=params['scaleFactor'],
                             nlevels=params['nlevels'],
                             edgeThreshold=params['edgeThreshold'], firstLevel=0, WTA_K=2,
                             scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=params['fastThreshold'])
        # Detect keypoints and descriptors in the template
        self.kp_template, self.des_template = orb.detectAndCompute(self.template, None)
        if self.des_template is None:
            raise ValueError("No descriptors found in template.")

        # Continuously capture frames and process them
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process the frame and display the matches
            frame_matches, match = self.process_frame(frame, orb)
            cv2.imshow('Detected Template', frame_matches)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


# Example usage:
if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else 'find'

    matcher = TemplateMatcher(r'/images/won_1000.jpg')
    if mode == 'find':
        matcher.find_best_parameters()
    elif mode == 'use':
        matcher.load_and_use_parameters()
