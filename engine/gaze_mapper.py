import numpy as np
import cv2
from collections import deque
import os
import matplotlib.pyplot as plt

class HeatmapAccumulator:
    def __init__(self, screen_w=1920, screen_h=1080):
        self.points = []
        self.screen_size = (screen_w, screen_h)

    def add_point(self, x, y):
        self.points.append((x, y))

    def save(self, output_path):
        if not self.points:
            return
        xs, ys = zip(*self.points)
        heatmap, xedges, yedges = np.histogram2d(
            xs, ys, bins=50,
            range=[[0, self.screen_size[0]], [0, self.screen_size[1]]]
        )
        plt.figure(figsize=(10, 6))
        plt.imshow(heatmap.T, origin='upper', cmap='hot',
                   extent=[0, self.screen_size[0], self.screen_size[1], 0])
        plt.colorbar(label='gaze density')
        plt.savefig(output_path)
        plt.close()


class GazeMapper:
    def __init__(self, cfg):
        self.cfg = cfg
        self.calibrated = False
        self.h_matrix = None
        self.focus_zones = cfg.focus_zones if hasattr(cfg, 'focus_zones') else []
        self.distraction_zones = cfg.distraction_zones if hasattr(cfg, 'distraction_zones') else []
        self.heatmap = HeatmapAccumulator()

    def calibrate(self, camera_gaze_points, screen_points):
        if len(camera_gaze_points) != 4 or len(screen_points) != 4:
            return
        src = np.array(camera_gaze_points, dtype=np.float32)
        dst = np.array(screen_points, dtype=np.float32)
        self.h_matrix, _ = cv2.findHomography(src, dst)
        self.calibrated = True

    def map_to_screen(self, gaze_x, gaze_y):
        if not self.calibrated or self.h_matrix is None:
            return None
        pt = np.array([[gaze_x, gaze_y]], dtype=np.float32).reshape(-1, 1, 2)
        screen_pt = cv2.perspectiveTransform(pt, self.h_matrix)
        return screen_pt[0][0]

    def check_zone(self, screen_x, screen_y):
        pt = (int(screen_x), int(screen_y))
        for poly in self.focus_zones:
            if len(poly) >= 3 and cv2.pointPolygonTest(np.array(poly, dtype=np.int32), pt, False) >= 0:
                return 'focus'
        for poly in self.distraction_zones:
            if len(poly) >= 3 and cv2.pointPolygonTest(np.array(poly, dtype=np.int32), pt, False) >= 0:
                return 'distraction'
        return 'other'