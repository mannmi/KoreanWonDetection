import cv2
import numpy as np
import json

from PyQt6.QtWidgets import QApplication

from src.camera.camera import Camera
from src.edge_detection.EdgeDetector import EdgeDetection
from src.ui.pop_up import SimplePopup




class TemplateMatcher:
    def __init__(self, template_path, edge_method='canny',popUp=False, camera_index=0):
        self.template = cv2.imread(template_path, 0)
        if self.template is None:
            raise ValueError("Error: Template image not found or cannot be opened.")

        self.popup = popUp
        self.camera = Camera(camera_index)
        self.edge_detection = EdgeDetection(edge_method)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.bbox = None
        self.found =False
        self.popup = None

    def match_found(self):
        app = QApplication(sys.argv)
        message = "Match to Currency was found."
        self.popup = SimplePopup(message)
        self.popup.show()
        # sys.exit(app.exec())
        #sys.exit(app.exec())

    def set_edge_method(self, method):
        self.edge_detection.set_method(method)

    def process_frame(self, frame, orb):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = self.edge_detection.detect_edges(gray)

        highlighted_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        highlighted_edges[np.where((highlighted_edges == [255, 255, 255]).all(axis=2))] = [0, 0, 255]

        kp_frame, des_frame = orb.detectAndCompute(gray, None)
        if des_frame is None:
            print("No descriptors found in frame.")
            return frame, False

        matches = self.bf.knnMatch(self.des_template, des_frame, k=2)
        good_matches = [m for m, n in matches if len(matches) > 1 and m.distance < 0.75 * n.distance]

        matches_mask = None
        if len(good_matches) > 150:
            src_pts = np.float32([self.kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            h, w = self.template.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            print("Match found!")
            print(len(good_matches))
            if not self.found and self.popup:
                self.match_found()
                self.found = True



        frame_matches = cv2.drawMatches(self.template, self.kp_template, frame, kp_frame, good_matches, None,
                                        matchesMask=matches_mask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('Edges', highlighted_edges)
        return frame_matches, len(good_matches) > 30

    def find_best_parameters(self):
        nfeatures_list = [500, 1000, 1500]
        scaleFactor_list = [1.1, 1.2, 1.3]
        nlevels_list = [8, 10, 12]
        edgeThreshold_list = [15, 31, 50]
        fastThreshold_list = [10, 20, 30]

        best_params = None
        best_matches = 0

        for nfeatures in nfeatures_list:
            for scaleFactor in scaleFactor_list:
                for nlevels in nlevels_list:
                    for edgeThreshold in edgeThreshold_list:
                        for fastThreshold in fastThreshold_list:
                            print(f"Testing ORB parameters: nfeatures={nfeatures}, scaleFactor={scaleFactor}, "
                                  f"nlevels={nlevels}, edgeThreshold={edgeThreshold}, fastThreshold={fastThreshold}")
                            orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels,
                                                 edgeThreshold=edgeThreshold, firstLevel=0, WTA_K=2,
                                                 scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31,
                                                 fastThreshold=fastThreshold)
                            self.kp_template, self.des_template = orb.detectAndCompute(self.template, None)
                            if self.des_template is None:
                                print("No descriptors found in template.")
                                continue

                            frame = self.camera.read_frame()
                            frame_matches, match = self.process_frame(frame, orb)
                            cv2.imshow('Testing ORB Parameters', frame_matches)
                            cv2.waitKey(1)

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

                        self.camera.release()
                        cv2.destroyAllWindows()

    def load_and_use_parameters(self):
        with open(
                r'C:\Users\mannnmi\PycharmProjects\KoreanWonDetection\src\image_template_matching\best_orb_params.json',
                'r') as f:
            params = json.load(f)

        orb = cv2.ORB_create(nfeatures=params['nfeatures'], scaleFactor=params['scaleFactor'],
                             nlevels=params['nlevels'],
                             edgeThreshold=params['edgeThreshold'], firstLevel=0, WTA_K=2,
                             scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31,
                             fastThreshold=params['fastThreshold'])
        self.kp_template, self.des_template = orb.detectAndCompute(self.template, None)
        if self.des_template is None:
            raise ValueError("No descriptors found in template.")

        while True:
            frame = self.camera.read_frame()
            frame_matches, match = self.process_frame(frame, orb)
            cv2.imshow('Detected Template', frame_matches)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.set_edge_method('canny')
            elif key == ord('s'):
                self.set_edge_method('sobel')
            elif key == ord('l'):
                self.set_edge_method('laplacian')

        self.camera.release()
        cv2.destroyAllWindows()


# Example usage:
if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else 'find'

    matcher = TemplateMatcher(
        r'C:\Users\mannnmi\PycharmProjects\KoreanWonDetection\images\won_1000.jpg', "sobel",True)
    if mode == 'find':
        matcher.find_best_parameters()
    elif mode == 'use':
        matcher.load_and_use_parameters()