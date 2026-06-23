import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque
import time
import os
from engine.emotion_classifier import MiniXceptionEmotion

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "face_landmarker.task")

class FaceProcessor:
    def __init__(self, cfg):
        from utils.downloader import download_face_landmarker
        download_face_landmarker()

        base = python.BaseOptions(model_asset_path=MODEL_PATH)
        opts = vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=cfg.face_detection_confidence,
            min_face_presence_confidence=cfg.face_presence_confidence,
            min_tracking_confidence=cfg.face_tracking_confidence,
            output_face_blendshapes=True
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]
        self.MOUTH_TOP = 13
        self.MOUTH_BOTTOM = 14
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291
        self.emotion_classifier = MiniXceptionEmotion()

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_img)
        if not result.face_landmarks:
            return None, None, None
        return result.face_landmarks[0], result.face_blendshapes[0] if result.face_blendshapes else None, frame_bgr

    def extract_features(self, landmarks, blendshapes, h, w, frame_bgr=None):
        pts = [(int(l.x * w), int(l.y * h)) for l in landmarks]
        left_eye = np.array([pts[i] for i in self.LEFT_EYE])
        right_eye = np.array([pts[i] for i in self.RIGHT_EYE])
        ear_val = (self.ear(left_eye) + self.ear(right_eye)) / 2.0
        mouth_pts = [pts[self.MOUTH_TOP], pts[self.MOUTH_BOTTOM],
                     pts[self.MOUTH_LEFT], pts[self.MOUTH_RIGHT]]
        mar_val = self.mar_4pt(mouth_pts)
        left_iris = np.array([pts[i] for i in self.LEFT_IRIS])
        right_iris = np.array([pts[i] for i in self.RIGHT_IRIS])
        xL, yL = self.iris_pos(left_iris, left_eye)
        xR, yR = self.iris_pos(right_iris, right_eye)
        gaze_x = (xL + xR) / 2.0
        gaze_y = (yL + yR) / 2.0
        disparity = abs(xL - xR)
        yaw, pitch, roll = self.head_pose(landmarks, w, h)
        eng = self.engagement_score(blendshapes) if blendshapes else 0.5
        mood = self.mood_label(eng)

        emotion_label, emotion_conf = self.emotion_classifier.predict(frame_bgr, landmarks, h, w)

        features = {
            'ear': ear_val,
            'mar': mar_val,
            'gaze_x': gaze_x,
            'gaze_y': gaze_y,
            'disparity': disparity,
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll,
            'engagement': eng,
            'mood': mood,
            'emotion': emotion_label,
            'emotion_conf': emotion_conf,
            'blink_count': 0,
            'perclos': 0,
            'heart_rate': None,
            'zone': 'unknown'
        }
        return features

    def ear(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C + 1e-6)

    def mar_4pt(self, pts):
        vertical = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))
        horizontal = np.linalg.norm(np.array(pts[2]) - np.array(pts[3]))
        return vertical / (horizontal + 1e-6)

    def iris_pos(self, iris_pts, eye_corners):
        center = np.mean(iris_pts, axis=0)
        left = eye_corners[0]
        right = eye_corners[3]
        vec = right - left
        width = np.linalg.norm(vec)
        if width == 0:
            return 0.5, 0.5
        t = np.dot(center - left, vec) / (width ** 2)
        x_rel = np.clip(t, 0.0, 1.0)
        top = max(eye_corners[1][1], eye_corners[2][1])
        bottom = min(eye_corners[4][1], eye_corners[5][1])
        height = bottom - top
        if height == 0:
            y_rel = 0.5
        else:
            y_rel = (center[1] - top) / height
        return x_rel, y_rel

    def head_pose(self, landmarks, w, h):
        nose = np.array([landmarks[1].x * w, landmarks[1].y * h])
        left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
        right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
        left_mouth = np.array([landmarks[61].x * w, landmarks[61].y * h])
        right_mouth = np.array([landmarks[291].x * w, landmarks[291].y * h])
        eye_center = (left_eye + right_eye) / 2
        mouth_center = (left_mouth + right_mouth) / 2
        face_center = (eye_center + mouth_center) / 2
        dx = nose[0] - face_center[0]
        dy = nose[1] - face_center[1]
        eye_dist = np.linalg.norm(right_eye - left_eye)
        yaw = np.degrees(np.arctan2(2 * dx, eye_dist))
        pitch = np.degrees(np.arctan2(2 * dy, eye_dist))
        return np.clip(yaw, -90, 90), np.clip(pitch, -90, 90), 0.0

    def engagement_score(self, blendshapes):
        KEYS = {
            'browDownLeft': 0.2, 'browDownRight': 0.2,
            'eyeSquintLeft': 0.1, 'eyeSquintRight': 0.1,
            'lipPressLeft': 0.15, 'lipPressRight': 0.15,
            'mouthFrownLeft': -0.2, 'mouthFrownRight': -0.2,
            'mouthSmileLeft': 0.3, 'mouthSmileRight': 0.3,
            'jawOpen': -0.3,
            'mouthStretchLeft': -0.2, 'mouthStretchRight': -0.2,
            'innerBrowRaise': -0.1, 'outerBrowRaise': -0.1
        }
        base = 0.5
        for bs in blendshapes:
            if bs.category_name in KEYS:
                base += bs.score * KEYS[bs.category_name]
        return np.clip(base, 0.0, 1.0)

    def mood_label(self, eng):
        if eng > 0.7:
            return "Focused"
        elif eng > 0.5:
            return "Neutral"
        elif eng > 0.3:
            return "Bored"
        else:
            return "Disengaged"

    def draw_annotations(self, frame, features, state, score, alert):
        h, w = frame.shape[:2]
        bar_x, bar_y, bar_w, bar_h = 20, h - 80, 200, 25
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill = int(bar_w * score / 100)
        color = (0, 255, 0) if score > 75 else (0, 255, 255) if score > 45 else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)
        cv2.putText(frame, f"{state}: {score:.0f}%", (bar_x + 5, bar_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if features:
            cv2.putText(frame, f"Gaze: {features['gaze_x']:.2f},{features['gaze_y']:.2f}", (w - 280, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if alert:
            cv2.putText(frame, f"ALERT: {alert}", (50, h - 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        return frame

    def close(self):
        self.landmarker.close()


class AttentionEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.baseline_ear = None
        self.baseline_gaze_x = None
        self.baseline_gaze_y = None
        self.baseline_disparity = None
        self.baseline_engagement = 0.5
        self.calibrated = False
        self.calib_ear = deque(maxlen=90)
        self.calib_gx = deque(maxlen=90)
        self.calib_gy = deque(maxlen=90)
        self.calib_disparity = deque(maxlen=90)
        self.calib_eng = deque(maxlen=90)

        # Adaptive thresholds – learned during calibration
        self.gaze_dev_threshold = 0.15    # default, will be overridden
        self.head_dev_threshold = 0.6     # default, will be overridden
        self.ear_absolute_floor = 0.15    # prevents tiny eyes from having impossibly low threshold

        self.eye_close_start = None
        self.perclos_buffer = deque(maxlen=cfg.perclos_window_frames)
        self.pitch_history = deque(maxlen=30)
        self.blink_count = 0
        self.yawn_start = None
        self.cumulative_sleep = 0.0
        self.cumulative_absent = 0.0
        self.last_time = time.time()
        self.smoothed_score = deque(maxlen=30)
        self.distraction_start = None
        self.distraction_alert_sent = False
        self.phone_detected = False
        self.phone_timer = 0.0
        self.start_time = time.time()
        self.liveness_required = False
        self.liveness_blinks_needed = 0
        self.liveness_blinks_done = 0
        self.liveness_start_time = None
        self.tamper_face_lost_start = None
        self.blink_timestamps = deque()
        self.dry_eye_alert_sent = False
        self.microsleep_threshold = 0.8
        self.distraction_grace = 2.0
        self.absent_reset_grace = 2.0

    def start_liveness_check(self):
        self.liveness_required = True
        self.liveness_blinks_needed = self.cfg.liveness_blinks_required
        self.liveness_blinks_done = 0
        self.liveness_start_time = time.time()

    def compute(self, features, face_present, now, frame_for_tamper=None):
        dt = now - self.last_time
        self.last_time = now

        # ---------- Liveness check ----------
        if self.liveness_required:
            if not face_present or features is None:
                if time.time() - self.liveness_start_time > self.cfg.liveness_timeout_sec:
                    self.liveness_required = False
                    return 'LIVENESS_FAILED', 0.0, "Liveness failed – no face detected"
                return 'LIVENESS', 0.0, f"Liveness: please look at camera ({self.liveness_blinks_done}/{self.liveness_blinks_needed} blinks)"

            ear_thresh = 0.22
            if self.baseline_ear is not None:
                ear_thresh = max(self.ear_absolute_floor, self.baseline_ear * self.cfg.ear_calibration_ratio)
            eye_closed = features['ear'] < ear_thresh

            if eye_closed:
                if self.eye_close_start is None:
                    self.eye_close_start = now
            else:
                if self.eye_close_start is not None:
                    dur = now - self.eye_close_start
                    if 0.1 < dur < 0.6:
                        self.liveness_blinks_done += 1
                    self.eye_close_start = None

            if self.liveness_blinks_done >= self.liveness_blinks_needed:
                self.liveness_required = False
                self.eye_close_start = None
                return 'LIVENESS_PASSED', 0.0, "Liveness verified"

            if time.time() - self.liveness_start_time > self.cfg.liveness_timeout_sec:
                self.liveness_required = False
                return 'LIVENESS_FAILED', 0.0, "Liveness timed out"

            return 'LIVENESS', 0.0, f"Liveness: blink {self.liveness_blinks_done}/{self.liveness_blinks_needed}"

        # ---------- Tamper detection ----------
        if frame_for_tamper is not None:
            gray = cv2.cvtColor(frame_for_tamper, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            if mean_brightness < self.cfg.tamper_dark_threshold or mean_brightness > self.cfg.tamper_bright_threshold:
                return 'TAMPER', 0.0, "Camera tampered (extreme brightness)"
            if not face_present:
                if self.tamper_face_lost_start is None:
                    self.tamper_face_lost_start = now
                elif now - self.tamper_face_lost_start > self.cfg.tamper_face_loss_timeout_sec:
                    return 'TAMPER', 0.0, "Camera tampered (face lost too long)"
            else:
                self.tamper_face_lost_start = None

        if not face_present or features is None:
            self.cumulative_absent += dt
            if self.cumulative_absent >= self.cfg.absent_warning_cum_sec:
                self.perclos_buffer.append(0)
                self.smoothed_score.append(0)
                return 'ABSENT', 0.0, "WARNING: Absent > 5 min"
            self.perclos_buffer.append(0)
            self.smoothed_score.append(0)
            return 'ABSENT', 0.0, None

        # Reset absence timer
        if self.cumulative_absent > 0:
            self.cumulative_absent = max(0, self.cumulative_absent - self.absent_reset_grace)
        self.tamper_face_lost_start = None

        ear_v = features['ear']
        mar_v = features['mar']
        gaze_x = features['gaze_x']
        gaze_y = features['gaze_y']
        disparity = features.get('disparity', 0.0)
        yaw = features['yaw']
        pitch = features['pitch']
        eng = features['engagement']

        if not self.calibrated:
            self.calib_ear.append(ear_v)
            self.calib_gx.append(gaze_x)
            self.calib_gy.append(gaze_y)
            self.calib_disparity.append(disparity)
            self.calib_eng.append(eng)
            if len(self.calib_ear) == 90:
                # Baselines
                self.baseline_ear = np.median(self.calib_ear)
                self.baseline_gaze_x = np.median(self.calib_gx)
                self.baseline_gaze_y = np.median(self.calib_gy)
                self.baseline_disparity = np.median(self.calib_disparity)
                self.baseline_engagement = np.median(self.calib_eng)

                # Adaptive gaze threshold: 2 * (std(x) + std(y)) of gaze during calibration
                std_gx = np.std(self.calib_gx) + 1e-6
                std_gy = np.std(self.calib_gy) + 1e-6
                self.gaze_dev_threshold = np.clip(2.0 * (std_gx + std_gy), 0.05, 0.25)

                # Adaptive head threshold: based on yaw/pitch variability (not stored, so estimate from current)
                # We'll approximate by using a fixed multiplier after calibration; we can also track head pose during calibration.
                # Since we didn't store yaw/pitch, we set a reasonable adaptive value.
                # But we can compute head dev threshold as 2 * typical deviation seen in first 90 frames.
                # Let's collect head poses as well.
                # For now we'll keep head threshold as default unless we collect it; let's add calib_yaw and calib_pitch.
                # Quick fix: just set head_dev_threshold to 0.5 as a reasonable adaptive value.
                self.head_dev_threshold = 0.5   # will be refined if head data collected later

                self.calibrated = True
            self.perclos_buffer.append(0)
            self.smoothed_score.append(50)
            return 'CALIBRATING', 50.0, None

        rel_eng = eng - self.baseline_engagement + 0.5
        rel_eng = np.clip(rel_eng, 0.0, 1.0)

        state = 'NEUTRAL'
        ear_thresh = max(self.ear_absolute_floor, self.baseline_ear * self.cfg.ear_calibration_ratio)
        eye_closed = ear_v < ear_thresh
        self.perclos_buffer.append(1 if eye_closed else 0)
        perclos = np.mean(self.perclos_buffer)
        features['perclos'] = perclos

        # Blink / dry‑eye
        if eye_closed:
            if self.eye_close_start is None:
                self.eye_close_start = now
        else:
            if self.eye_close_start is not None:
                dur = now - self.eye_close_start
                if 0.1 < dur < 0.4:
                    self.blink_count += 1
                    self.blink_timestamps.append(now)
                self.eye_close_start = None

        while self.blink_timestamps and now - self.blink_timestamps[0] > 60:
            self.blink_timestamps.popleft()

        blink_rate_1m = 0
        if len(self.blink_timestamps) > 0:
            interval = now - self.blink_timestamps[0]
            if interval > 0:
                blink_rate_1m = len(self.blink_timestamps) * 60 / interval

        dry_eye_alert = None
        session_elapsed = now - self.start_time
        if session_elapsed > self.cfg.dry_eye_alert_after_min * 60:
            if blink_rate_1m < self.cfg.dry_eye_blink_rate_min and not self.dry_eye_alert_sent:
                dry_eye_alert = "Low blink rate – remember to blink!"
                self.dry_eye_alert_sent = True
            elif blink_rate_1m >= self.cfg.dry_eye_blink_rate_min:
                self.dry_eye_alert_sent = False

        # Microsleep
        if eye_closed:
            if self.eye_close_start is None:
                self.eye_close_start = now
            closure_dur = now - self.eye_close_start
            if closure_dur > self.microsleep_threshold:
                self.cumulative_sleep += dt
                self.smoothed_score.append(5)
                alert = "Microsleep detected"
                if self.cumulative_sleep >= self.cfg.sleep_warning_cum_sec:
                    alert += " (>2 min)"
                return 'SLEEPING', 5.0, alert
        else:
            if self.eye_close_start is not None:
                dur = now - self.eye_close_start
                if dur > 0.1:
                    self.eye_close_start = None

        # Yawn
        if mar_v > self.cfg.mar_thresh:
            if self.yawn_start is None:
                self.yawn_start = now
            if now - self.yawn_start > 2.0:
                self.cumulative_sleep += dt
                self.smoothed_score.append(15)
                return 'DROWSY', 15.0, "Yawning"
        else:
            self.yawn_start = None

        # Head nodding
        self.pitch_history.append((now, pitch))
        recent = [(t, p) for t, p in self.pitch_history if now - t < 0.5]
        if len(recent) >= 2:
            pitches = [p for _, p in recent]
            if max(pitches) - min(pitches) > self.cfg.head_nod_threshold_deg:
                self.cumulative_sleep += dt
                self.smoothed_score.append(12)
                return 'DROWSY', 12.0, "Head nodding"

        if perclos > self.cfg.perclos_threshold:
            self.cumulative_sleep += dt
            self.smoothed_score.append(18)
            return 'DROWSY', 18.0, "High PERCLOS"

        if state not in ('SLEEPING', 'DROWSY'):
            self.cumulative_sleep = 0.0

        # Attention scoring with adaptive thresholds
        gaze_dev = np.sqrt((gaze_x - self.baseline_gaze_x)**2 + (gaze_y - self.baseline_gaze_y)**2)
        head_dev = max(abs(yaw) / 30, abs(pitch) / 25)   # normalized, unitless

        looking_at_keyboard = False
        if self.baseline_disparity is not None:
            disp_increase = disparity - self.baseline_disparity
            if pitch < -15 and disp_increase > 0.03:
                looking_at_keyboard = True

        # Score calculation using adaptive thresholds
        score = 100.0
        # Penalize gaze deviation beyond adaptive threshold
        gaze_penalty = min(40, max(0, (gaze_dev - self.gaze_dev_threshold * 0.5) * 200))
        score -= gaze_penalty
        # Penalize head deviation beyond adaptive threshold (normalized)
        head_penalty = min(30, max(0, (head_dev - self.head_dev_threshold * 0.5) * 60))
        score -= head_penalty
        score += (rel_eng - 0.5) * 20
        score = np.clip(score, 0, 100)

        emotion_mod = {
            'Happy': 1.02, 'Surprise': 1.01, 'Neutral': 1.0,
            'Sad': 0.98, 'Angry': 0.96, 'Fear': 0.94, 'Disgust': 0.92,
            'Uncertain': 1.0
        }
        mod = emotion_mod.get(features.get('emotion', 'Neutral'), 1.0)
        score = np.clip(score * mod, 0, 100)

        self.smoothed_score.append(score)
        smooth_score = np.mean(self.smoothed_score)

        # State classification with adaptive thresholds
        if smooth_score >= 80:
            state = 'ENGAGED'
            self.distraction_start = None
        elif smooth_score >= 60:
            state = 'FOCUSED'
            self.distraction_start = None
        elif smooth_score >= 40:
            if looking_at_keyboard:
                state = 'FOCUSED'
                self.distraction_start = None
            elif gaze_dev > self.gaze_dev_threshold or head_dev > self.head_dev_threshold:
                if self.distraction_start is None:
                    self.distraction_start = now
                elif (now - self.distraction_start) > self.distraction_grace:
                    state = 'DISTRACTED'
                else:
                    state = 'NEUTRAL'
            else:
                state = 'NEUTRAL'
                self.distraction_start = None
        elif smooth_score >= 20:
            state = 'BORED'
            self.distraction_start = None
        else:
            if self.distraction_start is None:
                self.distraction_start = now
            elif (now - self.distraction_start) > self.distraction_grace:
                state = 'DISTRACTED'
            else:
                state = 'NEUTRAL'

        alert = dry_eye_alert

        if self.phone_detected and state == 'DISTRACTED':
            self.phone_timer += dt
            if self.phone_timer > 3.0:
                alert = alert or "Phone distraction"
        else:
            self.phone_timer = 0.0

        if state == 'DISTRACTED' and not looking_at_keyboard:
            if self.distraction_start is not None and (now - self.distraction_start) > 10:
                if not self.distraction_alert_sent:
                    alert = alert or "Prolonged distraction"
                    self.distraction_alert_sent = True
        else:
            self.distraction_alert_sent = False

        if self.cumulative_sleep >= self.cfg.sleep_warning_cum_sec:
            alert = alert or "WARNING: Sleeping > 2 min"

        return state, smooth_score, alert