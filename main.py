"""
main.py
Author: Matt Lindborg
Course: MS548 - Advanced Programming Concepts and AI
Assignment: Week 6
Date: 10/15/2025

Purpose:
This is the entry point for the Learnflow Base application.
It wires together the user interface (ui.py) with the service layer (service.py).
The structure follows best practices:
    - Keep main.py minimal (only startup logic).
    - Delegate business logic to service.py.
    - Delegate GUI rendering to ui.py.
"""

import tkinter as tk
from app_core.app_manager import AppManager
from app_core.config_manager import ConfigManager

def main():
    root = tk.Tk()
    root.title("SifVision Studio")
    root.geometry("1172x820")

    app = AppManager(root)
    def on_close(self):
        """Save app state before closing."""
        try:
            # Save last-used model
            self.config["image_model"]["last_used"] = self.image_model_var.get()

            # Save avatar image path (if your app tracks it)
            if hasattr(self, "avatar_path"):
                self.config["app"]["avatar_image"] = self.avatar_path

            # Save current image generation settings
            if hasattr(self, "current_image_settings"):
                self.config["image_settings"] = self.current_image_settings

            # Save to JSON
            
            cfg_mgr = ConfigManager()
            cfg_mgr.data = self.config
            cfg_mgr.save()
            print("[INFO] Settings saved successfully.")

        except Exception as e:
            print(f"[WARN] Failed to save settings on exit: {e}")

        # Destroy the Tkinter root window
        self.root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == "__main__":
    main()
