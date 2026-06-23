import tkinter as tk
from tkinter import ttk
import requests
import threading

class ModernChat(ttk.Frame):
    def __init__(self, parent, config, **kwargs):
        super().__init__(parent, **kwargs)
        self.cfg = config
        self.configure(style='Card.TFrame')

        # Header
        header = ttk.Label(self, text="Study Assistant", font=('Segoe UI', 14, 'bold'),
                           foreground='#E0E0E0', background='#1E1E2E')
        header.pack(pady=(10,5))

        # Chat log
        self.chat_log = tk.Text(self, state='disabled', wrap=tk.WORD,
                                bg='#13131A', fg='#E0E0E0', font=('Segoe UI', 10),
                                relief='flat', borderwidth=0, padx=10, pady=10)
        self.chat_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Input area
        input_frame = ttk.Frame(self, style='Card.TFrame')
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        self.entry = ttk.Entry(input_frame, font=('Segoe UI', 11))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        self.entry.bind("<Return>", lambda e: self.send_message())

        self.send_btn = ttk.Button(input_frame, text="Send", style='Accent.TButton',
                                   command=self.send_message)
        self.send_btn.pack(side=tk.RIGHT, padx=(5,0))

        self.api_key = self.cfg.openrouter_api_key
        self.model = self.cfg.chat_model
        self.system_prompt = self.cfg.chat_system_prompt

    def send_message(self):
        msg = self.entry.get().strip()
        if not msg:
            return
        self.display_message("You", msg, '#2E7D32')
        self.entry.delete(0, tk.END)
        threading.Thread(target=self.get_reply, args=(msg,), daemon=True).start()

    def get_reply(self, msg):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": msg}
            ]
        }
        try:
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers=headers, json=payload, timeout=10)
            data = resp.json()
            reply = data['choices'][0]['message']['content'] if 'choices' in data else "API error"
        except Exception as e:
            reply = f"Error: {e}"
        self.display_message("Assistant", reply, '#1565C0')

    def display_message(self, sender, msg, color):
        # Guard against destroyed widget (when popup closed)
        if not self.winfo_exists():
            return
        self.chat_log.configure(state='normal')
        self.chat_log.insert(tk.END, f"{sender}\n", ('sender',))
        self.chat_log.insert(tk.END, f"{msg}\n\n", ('message',))
        self.chat_log.tag_configure('sender', foreground=color, font=('Segoe UI', 11, 'bold'))
        self.chat_log.tag_configure('message', foreground='#E0E0E0', font=('Segoe UI', 10))
        self.chat_log.configure(state='disabled')
        self.chat_log.see(tk.END)