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

def main():
    root = tk.Tk()
    root.title("SifVision Studio")
    root.geometry("1172x820")

    app = AppManager(root)
    root.mainloop()

if __name__ == "__main__":
    main()
