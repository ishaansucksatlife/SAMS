import tkinter as tk
from tkinter import ttk, messagebox
import sv_ttk
from PIL import Image, ImageTk
import cv2
import queue
import threading
import time
import numpy as np
from ui.widgets import ModernProgressBar, ModernGraph
from ui.mini_widget import MiniWidget
from ui.chatbox import ModernChat
from engine.attention import FaceProcessor, AttentionEngine
from engine.gaze_mapper import GazeMapper
from engine.posture import PostureCoach
from engine.drowsiness import DrowsinessForecast
from engine.occupancy import RoomOccupancy
from engine.task_detector import TaskDetector
from engine.attention_trend import AttentionTrendAnalyzer
from engine.distraction_classifier import DistractionClassifier
from utils.config import Config

class Dashboard:
    def __init__(self, root):
        self.root = root
        self.cfg = Config()
        self.running = True
        self.start_time = time.time()
        self.break_recommended = False
        self.chat_popup = None
        self.chat_instance = None
        self.break_suggestions_enabled = tk.BooleanVar(value=self.cfg.break_suggestions_enabled)

        self._setup_styles()

        main_frame = ttk.Frame(root, style='Background.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Sidebar
        sidebar = ttk.Frame(main_frame, width=220, style='Sidebar.TFrame')
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0,1))
        sidebar.pack_propagate(False)

        ttk.Label(sidebar, text="SAMS", font=('Segoe UI', 22, 'bold'),
                  foreground='#00E676', background='#1E1E2E').pack(pady=20)

        self.theme_var = tk.StringVar(value='dark')
        ttk.Button(sidebar, text="☀️ / 🌙", command=self.toggle_theme,
                   style='Sidebar.TButton').pack(pady=10)

        ttk.Button(sidebar, text="📌 Mini", command=self.toggle_mini,
                   style='Sidebar.TButton').pack(pady=5)

        ttk.Button(sidebar, text="💬 Chat", command=self.toggle_chat,
                   style='Sidebar.TButton').pack(pady=5)

        ttk.Button(sidebar, text="🎯 Calibrate Zones", command=self.calibrate_zones,
                   style='Sidebar.TButton').pack(pady=5)

        ttk.Button(sidebar, text="🔍 Liveness", command=self.safe_start_liveness,
                   style='Sidebar.TButton').pack(pady=5)

        ttk.Checkbutton(sidebar, text="Break Suggestions", variable=self.break_suggestions_enabled,
                        style='Switch.TCheckbutton').pack(pady=10)

        self.quick_score = ttk.Label(sidebar, text="Score: --", font=('Segoe UI', 14, 'bold'),
                                     foreground='white', background='#1E1E2E')
        self.quick_score.pack(pady=20)

        self.quick_state = ttk.Label(sidebar, text="State: --", font=('Segoe UI', 12),
                                     foreground='#9E9E9E', background='#1E1E2E')
        self.quick_state.pack()

        content = ttk.Frame(main_frame, style='Background.TFrame')
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.cam_frame = ttk.Frame(content, style='Card.TFrame')
        self.cam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10,5), pady=10)

        self.cam_label = ttk.Label(self.cam_frame, background='#13131A')
        self.cam_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Right panel: stats (4 columns)
        stats_panel = ttk.Frame(content, width=300, style='Card.TFrame')
        stats_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0,10), pady=10)
        stats_panel.pack_propagate(False)

        att_header = ttk.Frame(stats_panel, style='Card.TFrame')
        att_header.pack(fill=tk.X)
        ttk.Label(att_header, text="Attention", font=('Segoe UI', 12, 'bold'),
                  foreground='white', background='#1E1E2E').pack(pady=(10,0))
        self.att_bar = ModernProgressBar(att_header, width=260, height=16)
        self.att_bar.pack(pady=10)

        # Stats grid: 8 items, 4 columns × 2 rows
        stats_grid = ttk.Frame(stats_panel, style='Background.TFrame')
        stats_grid.pack(fill=tk.BOTH, expand=True)

        cards = [
            ("Task", "Default"),
            ("Emotion", "---"),
            ("Mood", "---"),
            ("Gaze", "---"),
            ("Blinks", "---"),
            ("Posture", "Good"),
            ("Drowsiness", "Low"),
            ("Room", "1")
        ]
        self.stat_labels = {}
        for i, (label, value) in enumerate(cards):
            card = ttk.Frame(stats_grid, style='StatCard.TFrame')
            card.grid(row=i//4, column=i%4, padx=2, pady=2, sticky='nsew')
            ttk.Label(card, text=label, font=('Segoe UI', 8), foreground='#9E9E9E',
                      background='#1E1E2E').pack()
            self.stat_labels[label] = ttk.Label(card, text=value, font=('Segoe UI', 10, 'bold'),
                                                foreground='white', background='#1E1E2E')
            self.stat_labels[label].pack()
        # Make columns equal weight
        for col in range(4):
            stats_grid.columnconfigure(col, weight=1)

        graph_card = ttk.Frame(stats_panel, style='Card.TFrame')
        graph_card.pack(fill=tk.BOTH, expand=True, pady=5)
        self.att_graph = ModernGraph(graph_card, width=260, height=130)
        self.att_graph.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.time_label = ttk.Label(graph_card, text="Focus stable", font=('Segoe UI', 10),
                                    foreground='#9E9E9E', background='#1E1E2E')
        self.time_label.pack()

        self.frame_queue = queue.Queue(maxsize=1)
        self.state_queue = queue.Queue()
        self.mini_widget = MiniWidget(self.cfg)

        self.processor = None
        self.engine = None
        self.gaze_mapper = None
        self.posture = None
        self.drowsiness = None
        self.occupancy = None
        self.task_detector = None
        self.trend_analyzer = None
        self.distraction_classifier = None

        self.root.after(100, self.update_ui)

    def _setup_styles(self):
        style = ttk.Style()
        style.configure('Background.TFrame', background='#13131A')
        style.configure('Sidebar.TFrame', background='#1E1E2E')
        style.configure('Card.TFrame', background='#1E1E2E', relief='flat', borderwidth=1)
        style.configure('StatCard.TFrame', background='#1E1E2E', relief='flat', borderwidth=0)
        style.configure('Accent.TButton', background='#00E676', foreground='black')
        style.configure('Sidebar.TButton', background='#1E1E2E', foreground='white')
        style.map('Sidebar.TButton', background=[('active', '#2E2E3E')])
        style.configure('Switch.TCheckbutton', background='#1E1E2E', foreground='white')

    def toggle_theme(self):
        if self.theme_var.get() == 'dark':
            sv_ttk.set_theme("light")
            self.theme_var.set('light')
        else:
            sv_ttk.set_theme("dark")
            self.theme_var.set('dark')

    def toggle_mini(self):
        if self.mini_widget.visible:
            self.mini_widget.hide()
        else:
            self.mini_widget.show()

    def toggle_chat(self):
        if self.chat_popup is not None and self.chat_popup.winfo_exists():
            self.chat_popup.destroy()
            self.chat_popup = None
            self.chat_instance = None
        else:
            self.chat_popup = tk.Toplevel(self.root)
            self.chat_popup.title("Study Assistant")
            self.chat_popup.geometry("400x500")
            self.chat_popup.configure(bg='#1E1E2E')
            self.chat_instance = ModernChat(self.chat_popup, self.cfg)
            self.chat_instance.pack(fill=tk.BOTH, expand=True)
            self.chat_popup.protocol("WM_DELETE_WINDOW", lambda: (
                self.chat_popup.destroy(),
                setattr(self, 'chat_popup', None),
                setattr(self, 'chat_instance', None)
            ))

    def safe_start_liveness(self):
        if self.engine:
            self.engine.start_liveness_check()
        else:
            messagebox.showwarning("Not Ready", "Please wait for the engine to initialise.")

    def calibrate_zones(self):
        """Capture gaze at four screen corners and calibrate gaze mapper."""
        if not self.gaze_mapper:
            messagebox.showwarning("Not Ready", "Please wait for the engine to start.")
            return

        corners = ["top‑left", "top‑right", "bottom‑right", "bottom‑left"]
        screen_coords = [
            (0, 0),
            (self.cfg.screen_calibration_points["top_right"][0], 0),
            (self.cfg.screen_calibration_points["bottom_right"][0], self.cfg.screen_calibration_points["bottom_right"][1]),
            (0, self.cfg.screen_calibration_points["bottom_left"][1])
        ]
        gaze_points = []

        for corner_name, screen_pt in zip(corners, screen_coords):
            messagebox.showinfo("Calibration", f"Look at the {corner_name} corner of your screen and press OK.")
            time.sleep(0.5)
            # Drain the state queue to get the most recent gaze values
            latest_state = None
            while not self.state_queue.empty():
                latest_state = self.state_queue.get()
            if latest_state:
                gaze_x = latest_state.get('gaze_x', 0.5)
                gaze_y = latest_state.get('gaze_y', 0.5)
                gaze_points.append((gaze_x, gaze_y))
            else:
                gaze_points.append((0.5, 0.5))

        if len(gaze_points) == 4:
            self.gaze_mapper.calibrate(gaze_points, screen_coords)
            messagebox.showinfo("Calibration", "Screen zones calibrated successfully!")
        else:
            messagebox.showerror("Calibration", "Failed to capture enough gaze points.")

    def run_engine(self):
        self.processor = FaceProcessor(self.cfg)
        self.engine = AttentionEngine(self.cfg)
        self.gaze_mapper = GazeMapper(self.cfg)
        self.posture = PostureCoach(self.cfg)
        self.drowsiness = DrowsinessForecast(self.cfg)
        self.occupancy = RoomOccupancy(self.cfg)
        self.task_detector = TaskDetector()
        self.trend_analyzer = AttentionTrendAnalyzer(self.cfg)
        self.distraction_classifier = DistractionClassifier()

        cap = cv2.VideoCapture(self.cfg.cam_id)
        if not cap.isOpened():
            print("ERROR: Cannot open webcam")
            return

        # Remove FPS limiters: set camera to maximum available FPS
        cap.set(cv2.CAP_PROP_FPS, 60)          # request 60 FPS (most webcams will cap at 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # minimize frame buffering

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1) if self.cfg.flip_horizontal else frame
            now = time.time()

            landmarks, blendshapes, _ = self.processor.process(frame)
            face_present = landmarks is not None
            features = None
            alert = None
            state = "ABSENT"
            score = 0.0
            current_task = "Default"

            if face_present:
                h, w = frame.shape[:2]
                features = self.processor.extract_features(landmarks, blendshapes, h, w, frame)

                if self.gaze_mapper.calibrated and features:
                    screen_pt = self.gaze_mapper.map_to_screen(features['gaze_x'], features['gaze_y'])
                    if screen_pt is not None:
                        self.gaze_mapper.heatmap.add_point(screen_pt[0], screen_pt[1])
                        zone = self.gaze_mapper.check_zone(screen_pt[0], screen_pt[1])
                        features['zone'] = zone

                state, score, alert = self.engine.compute(features, face_present, now, frame_for_tamper=frame)

                posture_alert = self.posture.assess(landmarks, h, w)
                if posture_alert:
                    self.stat_labels["Posture"].config(text=posture_alert)

                drowsy_risk = self.drowsiness.update(self.engine.blink_count,
                                                     features.get('perclos', 0),
                                                     features.get('heart_rate'))
                drowsy_str = self.drowsiness.risk_level if hasattr(self.drowsiness, 'risk_level') else "Low"
                self.stat_labels["Drowsiness"].config(text=drowsy_str)

                if drowsy_risk:
                    alert = f"{alert}; {drowsy_risk}" if alert else drowsy_risk

                extra_faces = self.occupancy.count_others(frame, landmarks)
                self.stat_labels["Room"].config(text=str(1+extra_faces))

                if int(now) % 30 == 0 and self.distraction_classifier:
                    distractions = self.distraction_classifier.detect_distractions(frame)
                    self.engine.phone_detected = any(d[0] == 'cell phone' for d in distractions)
                    if distractions and not self.engine.phone_detected:
                        alert = (alert or "") + f" {distractions[0][0]}"

                if int(now) % 5 == 0:
                    current_task = self.task_detector.detect()
                    self.stat_labels["Task"].config(text=current_task.capitalize())

                if self.break_suggestions_enabled.get():
                    self.trend_analyzer.update(score)
                    if self.trend_analyzer.should_recommend_break() and not self.break_recommended:
                        alert = (alert or "") + " Consider a break"
                        self.break_recommended = True
                    elif not self.trend_analyzer.should_recommend_break():
                        self.break_recommended = False

            else:
                state, score = 'ABSENT', 0.0
                self.trend_analyzer.update(0)

            frame = self.processor.draw_annotations(frame, features, state, score, alert)
            self.draw_buddy(frame, state)

            if not self.frame_queue.full():
                self.frame_queue.put(frame.copy())

            past_x, past_y, future_x, future_y = [], [], [], []
            if len(self.trend_analyzer.window) > 5:
                now = time.time()
                past_times = np.array([t - now for t, s in self.trend_analyzer.window])
                past_scores = np.array([s for t, s in self.trend_analyzer.window])
                past_x, past_y = past_times.tolist(), past_scores.tolist()
                future_times = np.linspace(0, 120, 30)
                future_scores = self.trend_analyzer.slope * (now + future_times) + self.trend_analyzer.intercept
                future_scores = np.clip(future_scores, 0, 100)
                future_x, future_y = future_times.tolist(), future_scores.tolist()

            self.state_queue.put({
                'state': state,
                'score': score,
                'alert': alert,
                'mood': features['mood'] if features else 'Unknown',
                'emotion': features['emotion'] if features else 'Unknown',
                'emotion_conf': features['emotion_conf'] if features else 0.0,
                'gaze_x': features['gaze_x'] if features else 0.5,
                'gaze_y': features['gaze_y'] if features else 0.5,
                'blink_count': self.engine.blink_count,
                'time_to_threshold': self.trend_analyzer.time_to_threshold,
                'past_x': past_x, 'past_y': past_y,
                'future_x': future_x, 'future_y': future_y,
                'task': current_task
            })

            self.root.after(10, self.update_ui)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            # No time.sleep – camera runs as fast as possible

        cap.release()
        self.processor.landmarker.close()

    def draw_buddy(self, frame, state):
        emoji = "😐" if state == 'NEUTRAL' else "😊" if state in ('ENGAGED','FOCUSED') else "😴" if state == 'DROWSY' else "😕"
        cv2.putText(frame, emoji, (frame.shape[1]-80, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    def update_ui(self):
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2_img)
            img = img.resize((640, 480), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.cam_label.imgtk = imgtk
            self.cam_label.configure(image=imgtk)

        if not self.state_queue.empty():
            state = self.state_queue.get()
            self.att_bar.set_score(state['score'])
            self.quick_score.config(text=f"Score: {state['score']:.0f}%")
            self.quick_state.config(text=f"State: {state['state']}")

            self.stat_labels["Emotion"].config(text=state['emotion'])
            self.stat_labels["Mood"].config(text=state['mood'])
            self.stat_labels["Gaze"].config(text=f"{state['gaze_x']:.2f}, {state['gaze_y']:.2f}")
            self.stat_labels["Blinks"].config(text=str(state['blink_count']))

            ttt = state.get('time_to_threshold')
            if ttt is not None:
                mins, secs = divmod(int(ttt), 60)
                self.time_label.config(text=f"Focus drop in {mins}m {secs}s")
            else:
                self.time_label.config(text="Focus stable")

            if state['past_x']:
                try:
                    self.att_graph.update_plot(state['past_x'], state['past_y'],
                                               state['future_x'], state['future_y'])
                except Exception:
                    pass

            if state['state'] in ('DISTRACTED', 'DROWSY', 'SLEEPING') and state['alert']:
                if self.chat_instance and self.chat_popup and self.chat_popup.winfo_exists():
                    try:
                        self.chat_instance.display_message("SAMS", state['alert'], '#FF1744')
                    except Exception:
                        pass

            self.mini_widget.update_score(state['score'])

        self.root.after(30, self.update_ui)