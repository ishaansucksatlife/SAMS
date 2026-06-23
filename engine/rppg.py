from collections import deque
import numpy as np

class rPPGExtractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.signal = deque(maxlen=150)
        self.heart_rate = None

    def process(self, frame, face_landmarks):
        if face_landmarks is None:
            return
        h, w = frame.shape[:2]
        fx = int(face_landmarks[10].x * w)
        fy = int(face_landmarks[10].y * h)
        roi = frame[max(0, fy - 30):fy + 20, max(0, fx - 25):fx + 25]
        if roi.size == 0:
            return
        G = np.mean(roi[:, :, 1])
        self.signal.append(G)
        if len(self.signal) == 150:
            signal = np.array(self.signal)
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
            fft = np.fft.rfft(signal * np.hanning(len(signal)))
            freqs = np.fft.rfftfreq(len(signal), d=1/30.0)
            valid = (freqs >= 0.7) & (freqs <= 4.0)
            if np.any(valid):
                peak_idx = np.argmax(np.abs(fft[valid]))
                hr_freq = freqs[valid][peak_idx]
                self.heart_rate = hr_freq * 60.0