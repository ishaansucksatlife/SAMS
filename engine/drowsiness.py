import time
from collections import deque

class DrowsinessForecast:
    def __init__(self, cfg):
        self.cfg = cfg
        # Short‑term blink rate tracking (per minute)
        self.blink_times = deque()
        self.last_perclos = 0
        self.risk_level = "Low"
        self.high_risk_start = None
        self.alert_sent = False

    def update(self, blink_count, perclos, hr):
        """
        blink_count: total blinks so far (cumulative, from attention engine)
        perclos: current PERCLOS value (0‑1)
        hr: heart rate (bpm) or None
        Returns a human‑readable risk description or None.
        """
        now = time.time()
        risk = None

        # Record blink event (we use blink_count change to deduce new blinks)
        # This method only knows the cumulative count; we need a different approach.
        # Instead, we'll keep a rolling window of perclos values and compute trends.
        # We'll assume blink_count is not directly accessible as per‑second here,
        # so we rely on perclos and a simple heuristic.
        
        # Track PERCLOS trend
        perclos_delta = perclos - self.last_perclos
        self.last_perclos = perclos

        # Determine risk based on multiple factors
        high_risk = False
        if perclos > 0.4:
            high_risk = True
        elif perclos > 0.3 and perclos_delta > 0.02:   # increasing rapidly
            high_risk = True
        elif hr and hr < 50:   # very low heart rate (unlikely with rPPG but possible)
            high_risk = True

        if high_risk:
            if self.high_risk_start is None:
                self.high_risk_start = now
            elif now - self.high_risk_start > 10:   # sustained high risk for 10 seconds
                self.risk_level = "High"
                risk = "High drowsiness risk"
                if not self.alert_sent:
                    self.alert_sent = True
                    return risk
        else:
            self.high_risk_start = None
            self.alert_sent = False
            if perclos < 0.2:
                self.risk_level = "Low"
            else:
                self.risk_level = "Medium"

        return risk if self.risk_level == "High" else None