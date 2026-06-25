# SAMS – Student Attention Management System

SAMS is a real time focus‑monitoring tool that uses a webcam and AI models to detect attention, drowsiness, distractions, emotions, and posture.  
It includes a chat assistant, break suggestions, screen‑zone calibration, and a lot more.

---

## Requirements

- **Python 3.14** (3.14 recommended)
- **Windows 10/11** (primary target; macOS/Linux may work with minor tweaks)
- **Webcam** (built‑in or USB)
- **OpenRouter API key** (optional, only for the chat assistant)

---

## Environment Setup

1. **Clone the repository** (or download the ZIP) and open a terminal in the project folder.

2. **Create a virtual environment** (replace `py` with the Python command for your system):
   ```cmd
   py -3.14 -m venv attention-env
   ```
3. **Activate the environment:**
   ```cmd
   attention-env\Scripts\activate
   ```
4. **Install dependencies:**
   ```cmd
   pip install opencv-python mediapipe numpy scipy pillow requests pygetwindow pystray matplotlib sv-ttk onnxruntime sounddevice
   ```
5. **Running Sams:**
   ```cmd
   python main.py
   ```
