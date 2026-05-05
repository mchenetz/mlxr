#!/usr/bin/env python3
"""MLXr macOS menu bar app. Run with: python menubar.py
Requires: pip install rumps"""
import os
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

try:
    import rumps
except ImportError:
    print("Install rumps: pip install rumps")
    raise

SCRIPT_DIR = Path(__file__).parent


class MLXrBar(rumps.App):
    def __init__(self):
        super().__init__("MLXr", title="◆", quit_button=None)
        self._proc = None
        self._port = int(os.environ.get("MLXR_PORT", "8000"))
        self._poll_thread = threading.Thread(target=self._poll_health, daemon=True)
        self._poll_thread.start()
        self.menu = [
            rumps.MenuItem("Open Dashboard", callback=self.open_dashboard),
            None,
            rumps.MenuItem("Start Server", callback=self.start_server),
            rumps.MenuItem("Stop Server", callback=self.stop_server),
            None,
            rumps.MenuItem("Quit MLXr", callback=self.quit_app),
        ]

    def _url(self):
        return f"http://localhost:{self._port}"

    def _poll_health(self):
        import urllib.request
        while True:
            try:
                urllib.request.urlopen(f"{self._url()}/health", timeout=2)
                running = True
            except Exception:
                running = False
            self.title = "◆ on" if running else "◆"
            time.sleep(3)

    @rumps.clicked("Open Dashboard")
    def open_dashboard(self, _):
        webbrowser.open(self._url())

    @rumps.clicked("Start Server")
    def start_server(self, _):
        if self._proc and self._proc.poll() is None:
            rumps.alert("MLXr is already running.")
            return
        sh = SCRIPT_DIR / "run.sh"
        self._proc = subprocess.Popen(
            ["bash", str(sh), "--port", str(self._port)],
            cwd=str(SCRIPT_DIR),
        )

    @rumps.clicked("Stop Server")
    def stop_server(self, _):
        if self._proc:
            self._proc.terminate()
            self._proc = None

    @rumps.clicked("Quit MLXr")
    def quit_app(self, _):
        if self._proc:
            self._proc.terminate()
        rumps.quit_application()


if __name__ == "__main__":
    MLXrBar().run()
