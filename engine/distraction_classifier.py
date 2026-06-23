import cv2
import numpy as np
import onnxruntime as ort
import urllib.request
import os

MODEL_URLS = [
    "https://github.com/yakhyo/yolov5-onnx-inference/releases/download/v0.0.1/yolov5n.onnx",
    "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"
]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "yolov5n.onnx")
DISTRACTION_CLASSES = {67: 'cell phone', 73: 'laptop', 0: 'person', 39: 'bottle', 56: 'chair'}

class DistractionClassifier:
    def __init__(self):
        self._ensure_model()
        self.session = ort.InferenceSession(MODEL_PATH)
        self.input_name = self.session.get_inputs()[0].name
        # Determine the expected input type
        self.input_type = self.session.get_inputs()[0].type  # e.g. 'tensor(float)' or 'tensor(float16)'
        self.use_float16 = 'float16' in self.input_type

    @staticmethod
    def _ensure_model():
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        if not os.path.exists(MODEL_PATH):
            for url in MODEL_URLS:
                try:
                    urllib.request.urlretrieve(url, MODEL_PATH)
                    break
                except Exception:
                    continue
            if not os.path.exists(MODEL_PATH):
                raise RuntimeError("Could not download YOLOv5n ONNX model from any source.")

    def detect_distractions(self, frame):
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 640))
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            if self.use_float16:
                img = img.astype(np.float16)
            outputs = self.session.run(None, {self.input_name: img})[0][0]

            distractions = []
            for det in outputs:
                conf = float(det[4])
                if conf < 0.5:
                    continue
                class_id = int(np.argmax(det[5:]))
                if class_id in DISTRACTION_CLASSES:
                    cx, cy, w, h = det[:4]
                    x1 = int((cx - w / 2) * frame.shape[1] / 640)
                    y1 = int((cy - h / 2) * frame.shape[0] / 640)
                    x2 = int((cx + w / 2) * frame.shape[1] / 640)
                    y2 = int((cy + h / 2) * frame.shape[0] / 640)
                    distractions.append((DISTRACTION_CLASSES[class_id], conf, x1, y1, x2, y2))
            return distractions
        except Exception as e:
            print(f"Distraction detection error: {e}")
            return []