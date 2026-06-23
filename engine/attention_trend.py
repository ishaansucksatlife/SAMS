import time
import numpy as np
from collections import deque

class AttentionTrendAnalyzer:
    def __init__(self, cfg):
        self.window = deque(maxlen=300)
        self.slope = 0.0
        self.intercept = 0.0
        self.critical_threshold = cfg.break_critical_threshold
        self.decline_threshold = cfg.break_decline_threshold
        self.time_to_threshold = None
        self.session_start = time.time()
        self.min_session_sec = cfg.break_minimum_session_minutes * 60

    def update(self, score):
        now = time.time()
        self.window.append((now, score))
        self._compute_trend()
        return self.time_to_threshold

    def _compute_trend(self):
        if len(self.window) < 2:
            self.slope = 0.0
            self.intercept = 0.0
            self.time_to_threshold = None
            return

        times, scores = zip(*self.window)
        x = np.array(times)
        y = np.array(scores)
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.dot(x, y)
        sum_x2 = np.dot(x, x)
        denom = n * sum_x2 - sum_x**2
        if denom != 0:
            self.slope = (n * sum_xy - sum_x * sum_y) / denom
            self.intercept = (sum_y - self.slope * sum_x) / n
        else:
            self.slope = 0.0
            self.intercept = np.mean(y)

        if self.slope < 0:
            t_now = times[-1]
            t_critical = (self.critical_threshold - self.intercept) / self.slope
            self.time_to_threshold = max(0, t_critical - t_now)
        else:
            self.time_to_threshold = None

    def should_recommend_break(self):
        now = time.time()
        if (now - self.session_start) < self.min_session_sec:
            return False
        if self.slope < self.decline_threshold:
            if self.time_to_threshold is not None and self.time_to_threshold < 60:
                return True
        return False