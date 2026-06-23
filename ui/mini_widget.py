import tkinter as tk

class MiniWidget:
    def __init__(self, config):
        self.cfg = config
        self.window = None
        self.visible = False
        self.score = 0
        self.label = None

    def show(self):
        if self.window is None or not self.window.winfo_exists():
            self.window = tk.Toplevel()
            self.window.overrideredirect(True)
            self.window.attributes('-topmost', True)
            self.window.geometry("200x40+100+100")
            self.window.configure(bg='#1E1E2E')
            self.label = tk.Label(self.window, text="Attention: --%", fg='white', bg='#1E1E2E',
                                  font=('Segoe UI', 12, 'bold'))
            self.label.pack(expand=True, fill=tk.BOTH)
            self.window.protocol('WM_DELETE_WINDOW', self.hide)
        self.window.deiconify()
        self.visible = True

    def hide(self):
        if self.window and self.window.winfo_exists():
            self.window.withdraw()
        self.visible = False

    def update_score(self, score):
        self.score = score
        if self.label and self.visible and self.window and self.window.winfo_exists():
            self.label.config(text=f"Attention: {score:.0f}%")