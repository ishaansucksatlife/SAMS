import tkinter as tk
import sv_ttk
import threading
from ui.dashboard import Dashboard
from ui.system_tray import SystemTray
from utils.downloader import download_face_landmarker

def main():
    download_face_landmarker()

    root = tk.Tk()
    root.title("SAMS")
    root.geometry("1280x720")
    root.resizable(False, False)
    sv_ttk.set_theme("dark")

    app = Dashboard(root)
    engine_thread = threading.Thread(target=app.run_engine, daemon=True)
    engine_thread.start()

    tray = SystemTray(root, app)
    root.protocol('WM_DELETE_WINDOW', tray.minimize_to_tray)
    root.mainloop()

if __name__ == "__main__":
    main()