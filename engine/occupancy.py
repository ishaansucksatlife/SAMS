import cv2

class RoomOccupancy:
    def __init__(self, cfg):
        self.face_cascade = None
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                self.face_cascade = None
        except Exception:
            self.face_cascade = None

    def count_others(self, frame, primary_landmarks):
        if self.face_cascade is None:
            return 0
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            return max(0, len(faces) - 1)
        except Exception:
            return 0