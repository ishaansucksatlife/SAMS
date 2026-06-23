import cv2
import numpy as np
import os
import urllib.request
from collections import deque
import onnxruntime as ort

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "fer2013_mini_xception.onnx")

class MiniXceptionEmotion:
    def __init__(self, smooth_frames=7, confidence_thresh=0.5):
        self.smooth_frames = smooth_frames
        self.confidence_thresh = confidence_thresh
        self._ensure_assets()
        self.session = ort.InferenceSession(MODEL_PATH)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = (64, 64)
        self.prediction_window = deque(maxlen=smooth_frames)

    @staticmethod
    def _ensure_assets():
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"ONNX model not found at {MODEL_PATH}. "
                "Please convert fer2013_mini_xception.h5 to ONNX first."
            )

    def align_face(self, frame, left_eye, right_eye):
        desired_left_eye = (0.35, 0.35)
        desired_right_eye = (0.65, 0.35)
        desired_size = self.input_size[0]
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        dist = np.sqrt(dX**2 + dY**2)
        desired_dist = (desired_right_eye[0] - desired_left_eye[0]) * desired_size
        scale = desired_dist / dist
        eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                       (left_eye[1] + right_eye[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        tX = desired_size * 0.5
        tY = desired_size * desired_left_eye[1]
        M[0, 2] += tX - eyes_center[0]
        M[1, 2] += tY - eyes_center[1]
        aligned = cv2.warpAffine(frame, M, (desired_size, desired_size), flags=cv2.INTER_CUBIC)
        return aligned

    def preprocess(self, aligned_face):
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        gray = gray.astype('float32') / 255.0
        gray = np.expand_dims(gray, axis=-1)
        gray = np.expand_dims(gray, axis=0)
        return gray

    def predict(self, frame, landmarks, w, h):
        left_eye = (int(landmarks[33].x * w), int(landmarks[33].y * h))
        right_eye = (int(landmarks[263].x * w), int(landmarks[263].y * h))
        aligned = self.align_face(frame, left_eye, right_eye)
        input_tensor = self.preprocess(aligned)
        input_tensor = input_tensor.astype(np.float32)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        preds = outputs[0][0]
        self.prediction_window.append(preds)
        avg_preds = np.mean(self.prediction_window, axis=0)
        emotion_idx = np.argmax(avg_preds)
        confidence = avg_preds[emotion_idx]
        if confidence < self.confidence_thresh:
            return "Uncertain", confidence
        return EMOTIONS[emotion_idx], confidence