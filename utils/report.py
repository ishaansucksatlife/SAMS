import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

class ReportGenerator:
    def __init__(self):
        self.states = []
        self.scores = []
        self.gaze_points = []

    def add_frame_data(self, state, score, gaze_x=None, gaze_y=None):
        self.states.append(state)
        self.scores.append(score)
        if gaze_x is not None and gaze_y is not None:
            self.gaze_points.append((gaze_x, gaze_y))

    def generate_pdf(self, output_path):
        plt.figure(figsize=(10, 4))
        plt.plot(self.scores, label='Attention Score')
        plt.axhline(y=75, color='g', linestyle='--')
        plt.axhline(y=45, color='r', linestyle='--')
        plt.title('Attention Over Time')
        plt.legend()
        plt.savefig(output_path.replace('.pdf', '.png'))
        plt.close()

    def save_heatmap(self, output_path):
        if not self.gaze_points:
            return
        xs, ys = zip(*self.gaze_points)
        heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=50, range=[[0, 1920], [0, 1080]])
        plt.figure(figsize=(10, 6))
        plt.imshow(heatmap.T, origin='upper', cmap='hot', extent=[0, 1920, 1080, 0])
        plt.colorbar(label='gaze density')
        plt.savefig(output_path)
        plt.close()