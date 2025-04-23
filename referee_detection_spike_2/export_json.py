import json

class JSONExporter:
    def __init__(self):
        self.data = {}

    def add_frame(self, frame_id, tracked_objects):
        self.data[f"frame_{frame_id:04d}"] = [
            {
                "id": obj['id'],
                "role": obj.get('role', 'player'),
                "bbox": obj['bbox'],
                "field_coord": obj.get('field_coord', None)
            }
            for obj in tracked_objects
        ]

    def add_pitch_info(self, homography, lines):
        self.data['pitch_homography'] = homography.tolist()
        self.data['pitch_lines'] = lines

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=2) 