# ui_frontend.py
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ui.ui_logic import UiLogic

class AppUI:
    def __init__(self, root, service):
        self.root = root
        self.service = service
        self.logic = UiLogic(self)  # connect logic helper

        self.root.title("SifVision Studio")
        self.root.minsize(900, 600)
        self.root.resizable(True, True)

        self._build_layout()

    def _build_layout(self):
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Left + Right panels
        middle_frame = tk.Frame(main_frame)
        middle_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        middle_frame.columnconfigure(0, weight=1)
        middle_frame.columnconfigure(1, weight=3)

        # --- Avatar on Left ---
        left_frame = tk.Frame(middle_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        try:
            self.avatar_image = Image.open("images\\image2_50pc.png")
            self.avatar_photo = ImageTk.PhotoImage(self.avatar_image)
            self.image_label = tk.Label(left_frame, image=self.avatar_photo)
            self.image_label.pack(fill="both", expand=True, padx=5, pady=5)
        except Exception:
            pass

        # --- Bottom Controls ---
        controls_frame = tk.Frame(left_frame)
        controls_frame.pack(side="bottom", fill="x", pady=(8, 5))
        controls_frame.columnconfigure(1, weight=1)

        # Audio checkbox
        self.audio_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            controls_frame,
            text="Audio",
            variable=self.audio_var,
            command=lambda: self.logic.toggle_audio(self.audio_var.get())
        ).grid(row=0, column=0, sticky="w", padx=(5, 10))

        # Dropdown
        self.image_model_var = tk.StringVar(value="stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf")
        model_options = self.logic.get_model_options()
        self.model_dropdown = ttk.Combobox(
            controls_frame,
            textvariable=self.image_model_var,
            values=model_options,
            state="readonly",
            width=28
        )
        self.model_dropdown.grid(row=0, column=1, sticky="ew", padx=(5, 10))

        # Buttons
        tk.Button(controls_frame, text="⚙", width=4, command=self.logic.open_image_settings).grid(row=0, column=2, padx=2)
        tk.Checkbutton(controls_frame, text="Use Current Image", variable=self.logic.use_guidance_var).grid(row=0, column=3, padx=4)
        tk.Button(controls_frame, text="🗑 Clear", width=7, command=self.logic.clear_embedded_images).grid(row=0, column=4, padx=4)
        tk.Button(controls_frame, text="🖼 Generate", width=9, command=self.logic.generate_image_from_prompt).grid(row=0, column=5, padx=4)

        # Progress bar
        self.progress = ttk.Progressbar(controls_frame, orient="horizontal", mode="determinate")
        self.progress.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(6, 3), padx=(5, 5))
        self.progress["value"] = 0

        # --- Right Chat Panel ---
        right_frame = tk.Frame(middle_frame)
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.columnconfigure(0, weight=1)

        self.ai_output_box = tk.Text(right_frame, wrap="word", font=("Segoe UI", 10), bg="#1e1e1e", fg="#dcdcdc")
        self.ai_output_box.grid(row=0, column=0, sticky="nsew")
        self.ai_output_box.insert(tk.END, "Chat output...\n")

        # --- Bottom Bar ---
        self.response_label = tk.Label(main_frame, text="Response time: —", fg="#aaa", bg="#1e1e1e")
        self.response_label.grid(row=4, column=0, sticky="e", padx=10)
