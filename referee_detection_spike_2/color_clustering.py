import numpy as np
from sklearn.cluster import KMeans
import cv2
from collections import Counter

class RefereeIdentifier:
    def __init__(self):
        pass

    def extract_color(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        
        # Ensure bbox is within frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        # Check if the crop area is valid
        if x2 <= x1 or y2 <= y1:
            return np.array([0, 0, 0])  # Return a default color
            
        person_crop = frame[y1:y2, x1:x2]
        
        # Check if crop is empty
        if person_crop.size == 0:
            return np.array([0, 0, 0])  # Return a default color
            
        hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (36, 0, 0), (86, 255, 255))
        hsv[mask > 0] = 0  # remove green
        
        # Check if there are non-zero values after masking
        non_zero = hsv[hsv[:, :, 0] > 0]
        if non_zero.size == 0:
            return np.array([0, 0, 0])  # Return a default color
            
        mean_color = non_zero.mean(axis=0)
        return mean_color

    def cluster_jerseys(self, color_features):
        k = min(3, len(color_features))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(color_features)
        return kmeans.labels_, kmeans.cluster_centers_

    def identify_referee(self, detections, color_features, labels):
        cluster_counts = Counter(labels)
        referee_cluster = min(cluster_counts, key=cluster_counts.get)
        for idx, label in enumerate(labels):
            if label == referee_cluster:
                detections[idx]['role'] = 'referee'
            else:
                detections[idx]['role'] = 'player'
        return detections 