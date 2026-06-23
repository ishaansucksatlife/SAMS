import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ModernProgressBar(tk.Canvas):
    def __init__(self, parent, width=400, height=12, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg='#1E1E2E', highlightthickness=0, **kwargs)
        self.width = width
        self.height = height
        self.score = 0
        self.draw()

    def set_score(self, score):
        self.score = max(0, min(100, score))
        self.draw()

    def draw(self):
        self.delete("all")
        # Rounded background
        radius = self.height // 2
        self.create_rounded_rect(0, 0, self.width, self.height, radius,
                                 fill='#2E2E3E', outline='')
        # Fill
        fill_width = int(self.width * self.score / 100)
        if self.score > 75:
            color1, color2 = '#00C853', '#64DD17'
        elif self.score > 45:
            color1, color2 = '#FFD600', '#FFAB00'
        else:
            color1, color2 = '#FF1744', '#D50000'
        if fill_width > 0:
            self.create_rounded_rect(0, 0, fill_width, self.height, radius,
                                     fill=color1, outline='')
            # Gradient effect: overlay a slightly lighter rect
            self.create_rounded_rect(0, 0, fill_width, self.height//2, radius,
                                     fill=color2, outline='')
        # Text
        self.create_text(self.width/2, self.height/2, text=f"{self.score:.0f}%",
                         fill='white', font=('Segoe UI', 10, 'bold'))

    def create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1+r, y1, x1+r, y1,
            x2-r, y1, x2-r, y1,
            x2, y1, x2, y1+r,
            x2, y2-r, x2, y2,
            x2-r, y2, x2-r, y2,
            x1+r, y2, x1+r, y2,
            x1, y2, x1, y2-r,
            x1, y1+r, x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)


class ModernGraph(ttk.Frame):
    def __init__(self, parent, width=300, height=200):
        super().__init__(parent)
        self.figure = Figure(figsize=(width/100, height/100), dpi=100,
                             facecolor='#1E1E2E')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1E1E2E')
        self.ax.tick_params(colors='#9E9E9E')
        self.ax.spines['bottom'].set_color('#3E3E4E')
        self.ax.spines['left'].set_color('#3E3E4E')
        self.ax.set_xlabel('Time (s)', color='#9E9E9E')
        self.ax.set_ylabel('Attention', color='#9E9E9E')
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plot(self, past_x, past_y, future_x, future_y):
        self.ax.clear()
        self.ax.set_facecolor('#1E1E2E')
        self.ax.tick_params(colors='#9E9E9E')
        self.ax.spines['bottom'].set_color('#3E3E4E')
        self.ax.spines['left'].set_color('#3E3E4E')
        if past_x:
            self.ax.plot(past_x, past_y, color='#00E676', linewidth=2, label='History')
        if future_x:
            self.ax.plot(future_x, future_y, color='#FFAB00', linestyle='--', linewidth=1.5, label='Projected')
        self.ax.axhline(y=50, color='#FF1744', linestyle=':', alpha=0.6)
        self.ax.set_xlabel('Time (s)', color='#9E9E9E')
        self.ax.set_ylabel('Attention', color='#9E9E9E')
        self.ax.legend(loc='upper left', facecolor='#2E2E3E', edgecolor='#3E3E4E',
                       labelcolor='white')
        self.canvas.draw()