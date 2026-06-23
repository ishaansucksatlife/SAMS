import pystray
from PIL import Image, ImageDraw
import threading

class SystemTray:
    def __init__(self, root, dashboard):
        self.root = root
        self.dashboard = dashboard
        self.icon = None
        self.mini_widget_visible = False
        self.create_tray_icon()

    def create_tray_icon(self):
        image = Image.new('RGB', (64, 64), color=(0, 120, 212))
        dc = ImageDraw.Draw(image)
        dc.rectangle([16, 16, 48, 48], fill='white')
        self.icon = pystray.Icon("SAMS", image, "SAMS", self.create_menu())
        threading.Thread(target=self.icon.run, daemon=True).start()

    def create_menu(self):
        return pystray.Menu(
            pystray.MenuItem("Show Dashboard", self.show_window, default=True),
            pystray.MenuItem("Show Mini Widget", self.toggle_mini_widget),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self.quit_app)
        )

    def show_window(self, icon, item):
        self.root.deiconify()
        self.root.lift()

    def toggle_mini_widget(self, icon, item):
        if self.mini_widget_visible:
            self.dashboard.mini_widget.hide()
            self.mini_widget_visible = False
        else:
            self.dashboard.mini_widget.show()
            self.mini_widget_visible = True

    def minimize_to_tray(self):
        self.root.withdraw()

    def quit_app(self, icon, item):
        self.icon.stop()
        self.root.quit()