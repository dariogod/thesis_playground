import cv2
import numpy as np

class Minimap:
    def __init__(self, field_size=(105, 68)):
        self.field_size = field_size
        self.scale = 10  # pixels per meter
        self.canvas_size = (int(field_size[0] * self.scale), int(field_size[1] * self.scale))

    def draw_pitch(self):
        canvas = np.ones((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8) * 255
        cv2.rectangle(canvas, (0, 0), (self.canvas_size[0]-1, self.canvas_size[1]-1), (0, 255, 0), 2)
        center = (self.canvas_size[0] // 2, self.canvas_size[1] // 2)
        radius = int(9.15 * self.scale)
        cv2.circle(canvas, center, radius, (0, 0, 255), 1)
        return canvas

    def draw_players(self, canvas, positions):
        for p in positions:
            x, y = int(p[0] * self.scale), int(p[1] * self.scale)
            cv2.circle(canvas, (x, y), 4, (255, 0, 0), -1)
        return canvas 