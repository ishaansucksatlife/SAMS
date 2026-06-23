import json
import os

class Config:
    def __init__(self, path="config.json"):
        self.path = path
        self.data = {}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {
                "cam_id": 0,
                "flip_horizontal": True,
                "face_detection_confidence": 0.5,
                "face_presence_confidence": 0.5,
                "face_tracking_confidence": 0.5,
                "pose_detection_confidence": 0.5,
                "pose_tracking_confidence": 0.5,
                "ear_calibration_ratio": 0.75,
                "mar_thresh": 0.7,
                "microsleep_sec": 0.3,
                "perclos_window_frames": 60,
                "perclos_threshold": 0.4,
                "head_nod_threshold_deg": 10,
                "blink_rate_alert_per_min": 20,
                "verify_thresh": 0.6,
                "verify_every_n": 15,
                "sleep_warning_cum_sec": 120,
                "absent_warning_cum_sec": 300,
                "break_grace_period_sec": 120,
                "beep_on_warning": True,
                "openrouter_api_key": "",
                "chat_model": "openai/gpt-4o-mini",
                "chat_system_prompt": "You are a supportive study coach. Keep answers short and motivating.",
                "pomodoro_work_min": 25,
                "pomodoro_short_break_min": 5,
                "pomodoro_long_break_min": 15,
                "pomodoro_cycles_before_long_break": 4,
                "rppg_enabled": True,
                "object_detection_enabled": True,
                "audio_analysis_enabled": False,
                "screen_calibration_points": {
                    "top_left": [0, 0],
                    "top_right": [1920, 0],
                    "bottom_right": [1920, 1080],
                    "bottom_left": [0, 1080]
                },
                "focus_zones": [],
                "distraction_zones": []
            }
            self.save()

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def __getattr__(self, name):
        return self.data.get(name, None)