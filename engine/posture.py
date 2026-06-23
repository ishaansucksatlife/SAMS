import numpy as np

class PostureCoach:
    def __init__(self, cfg):
        self.cfg = cfg
        self.forward_head_threshold = 20   # degrees pitch forward = slouch
        self.lateral_tilt_threshold = 15   # degrees roll
        self.stable_frames = 0
        self.posture_state = "Good"

    def assess(self, face_landmarks, h, w):
        """
        face_landmarks: NormalizedLandmark list (MediaPipe 468 points)
        Returns a human‑readable posture alert or None.
        """
        if face_landmarks is None:
            return None

        # Extract key landmarks
        nose_tip = np.array([face_landmarks[1].x * w, face_landmarks[1].y * h])
        left_eye_inner = np.array([face_landmarks[33].x * w, face_landmarks[33].y * h])
        right_eye_inner = np.array([face_landmarks[263].x * w, face_landmarks[263].y * h])
        left_mouth = np.array([face_landmarks[61].x * w, face_landmarks[61].y * h])
        right_mouth = np.array([face_landmarks[291].x * w, face_landmarks[291].y * h])

        # Head pitch (same as attention engine)
        eye_center = (left_eye_inner + right_eye_inner) / 2.0
        mouth_center = (left_mouth + right_mouth) / 2.0
        face_center = (eye_center + mouth_center) / 2.0
        dx = nose_tip[0] - face_center[0]
        dy = nose_tip[1] - face_center[1]
        eye_dist = np.linalg.norm(right_eye_inner - left_eye_inner)
        pitch = np.degrees(np.arctan2(2 * dy, eye_dist))   # positive = head up, negative = head down

        # Head roll: angle between the eyes
        dX_eyes = right_eye_inner[0] - left_eye_inner[0]
        dY_eyes = right_eye_inner[1] - left_eye_inner[1]
        roll = np.degrees(np.arctan2(dY_eyes, dX_eyes))

        # Shoulder estimation (approximate from face position relative to frame center)
        # We use the vertical position of the eyes as a proxy for sitting height.
        # If eyes drop below 40% of frame height, likely slouching.
        eye_avg_y = (left_eye_inner[1] + right_eye_inner[1]) / 2.0
        frame_mid_y = h * 0.4
        vertical_shift = (eye_avg_y - frame_mid_y) / h   # positive if lower

        alert = None
        issues = []

        if pitch < -self.forward_head_threshold:
            issues.append("Head tilted forward")
        if abs(roll) > self.lateral_tilt_threshold:
            issues.append("Head tilted sideways")
        if vertical_shift > 0.1:    # eyes more than 10% below ideal
            issues.append("Slouching detected")

        if issues:
            alert = "; ".join(issues)
            self.stable_frames = 0
            self.posture_state = "Poor"
        else:
            self.stable_frames += 1
            if self.stable_frames > 30:   # about 1 second of good posture
                self.posture_state = "Good"
                alert = None

        return alert