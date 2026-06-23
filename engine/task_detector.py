import pygetwindow as gw
import json
import os

class TaskDetector:
    def __init__(self, config_path="config.json"):
        self.profiles = {}
        self.current_profile = "default"
        self.load_profiles(config_path)

    def load_profiles(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                cfg = json.load(f)
            self.profiles = cfg.get("task_profiles", {})
        # Default profiles if none defined
        if not self.profiles:
            self.profiles = {
                "coding": {"keywords": ["code", "visual studio", "pycharm", "vscode", "terminal"]},
                "reading": {"keywords": ["pdf", "reader", "chrome", "firefox", "edge"]},
                "writing": {"keywords": ["word", "notepad", "writer", "notes"]},
                "meeting": {"keywords": ["zoom", "teams", "meet", "skype"]}
            }

    def detect(self):
        try:
            win = gw.getActiveWindow()
            if win:
                title = win.title.lower()
                for profile, data in self.profiles.items():
                    keywords = data.get("keywords", [])
                    if any(kw in title for kw in keywords):
                        self.current_profile = profile
                        return profile
        except Exception:
            pass
        self.current_profile = "default"
        return "default"