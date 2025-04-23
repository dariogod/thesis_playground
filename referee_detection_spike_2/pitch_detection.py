import cv2
import numpy as np

class PitchDetector:
    def detect_lines(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
        return lines

    def find_homography(self, image_points, pitch_points):
        H, status = cv2.findHomography(image_points, pitch_points, method=cv2.RANSAC)
        return H 