from norfair import Detection as NorfairDetection, Tracker
import numpy as np

class TrackManager:
    def __init__(self):
        self.tracker = Tracker(distance_function=self.euclidean_distance, distance_threshold=30)

    def euclidean_distance(self, det1, det2):
        # Check if we're comparing two Detections
        if hasattr(det1, 'points') and hasattr(det2, 'points'):
            return np.linalg.norm(det1.points - det2.points)
        # If we're comparing a Detection with a TrackedObject
        elif hasattr(det1, 'points') and hasattr(det2, 'estimate'):
            return np.linalg.norm(det1.points - det2.estimate)
        # If we're comparing a TrackedObject with a Detection
        elif hasattr(det1, 'estimate') and hasattr(det2, 'points'):
            return np.linalg.norm(det1.estimate - det2.points)
        # If we're comparing two TrackedObjects
        elif hasattr(det1, 'estimate') and hasattr(det2, 'estimate'):
            return np.linalg.norm(det1.estimate - det2.estimate)
        else:
            raise AttributeError("Objects don't have either points or estimate attributes")

    def update(self, detections):
        wrapped_detections = [
            NorfairDetection(
                points=np.array([
                    [d['bbox'][0] + d['bbox'][2], d['bbox'][1] + d['bbox'][3]]
                ])
            ) 
            for d in detections
        ]
        return self.tracker.update(detections=wrapped_detections) 