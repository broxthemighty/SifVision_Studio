"""
ui.py
Author: Matt Lindborg
Course: MS548 - Advanced Programming Concepts and AI
Assignment: Week 6
Date: 10/15/2025

Purpose:
This file defines the Tkinter-based graphical user interface (GUI)
for the Learnflow application. The GUI layer is responsible for:
    - Displaying buttons, menus, and text areas to the user.
    - Collecting input via popup dialogs.
    - Rendering summaries and history of entries.
    - Delegating business logic to the LlmService (llm_service.py).

This file does NOT contain data storage logic. Instead:
    - It calls service methods to set/get data.
    - It uses the domain model (LearningLog, EntryType) indirectly.
    - It is designed to be event-driven (each button triggers a method).
"""

# --- Imports ---
# standard library
import json                            # for save/load functionality
import csv                             # excel file output
import tkinter as tk
from tkinter import filedialog, ttk         # standard Tkinter dialogs

# local
from memory.domain_models import EntryType           # domain layer EntryType
from llm.llm_service import LlmService
import threading
import os

from PIL import Image, ImageTk
from vision.image_gen import (
    list_available_models,
    generate_image_diffusers,
    clear_pipeline_cache
)
from app_core.config_manager import ConfigManager
from pathlib import Path


class AutoScrollbar(tk.Scrollbar):
    """
    A scrollbar that hides itself when not needed.
    Works only with grid geometry.
    """
    def set(self, lo, hi):
        # hide scrollbar if the text fits entirely in the view
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
        super().set(lo, hi)

    def pack(self, **kw):
        raise tk.TclError("Use grid instead of pack with AutoScrollbar")

    def place(self, **kw):
        raise tk.TclError("Use grid instead of place with AutoScrollbar")

class App:
    """
    The App class defines the GUI layout and event handlers.
    It is initialized with a root Tkinter window and a LlmService.
    """
    def __init__(self, root: tk.Tk, service: LlmService):
        """
        Constructor initializes the window, builds the layout,
        and renders the initial summary display.
        """
        # hold references to the Tk root and the business service
        self.root = root
        self.service = service
        self.service.ui = self
        
        self.avatar_min_size = 256
        self.avatar_max_size = 512

        # set base background and foreground
        self.root.option_add("*Background", "#2b2b2b")      # dark gray background
        self.root.option_add("*Foreground", "#ffffff")      # white text

        # button colors
        self.root.option_add("*Button.Background", "#444444")
        self.root.option_add("*Button.Foreground", "#ffffff")
        self.root.option_add("*Button.ActiveBackground", "#666666")
        self.root.option_add("*Button.ActiveForeground", "#ffffff")

        # entry box colors
        self.root.option_add("*Entry.Background", "#1e1e1e")
        self.root.option_add("*Entry.Foreground", "#dcdcdc")
        self.root.option_add("*Entry.InsertBackground", "#ffffff")

        # text box colors
        self.root.option_add("*Text.Background", "#1e1e1e")
        self.root.option_add("*Text.Foreground", "#dcdcdc")

        # menu colors
        self.root.option_add("*Menu.Background", "#2b2b2b")
        self.root.option_add("*Menu.Foreground", "#ffffff")
        self.root.option_add("*Menu.ActiveBackground", "#444444")
        self.root.option_add("*Menu.ActiveForeground", "#ffffff")

        # global font setting
        default_font = ("Segoe UI", 10)

        # --- Main container frame ---
        main_frame = tk.Frame(root, padx=10, pady=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        #main_frame.columnconfigure(0, weight=1)

        # --- Dynamic scaling ---
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=0)  # top/menu
        main_frame.rowconfigure(1, weight=1)  # middle_frame
        main_frame.rowconfigure(2, weight=0)  # optional padding
        main_frame.rowconfigure(3, weight=0)  # chat input
        main_frame.rowconfigure(4, weight=0)  # response label

        # --- Top row: welcome label, clear button, drop-down menu ---
        top_frame = tk.Frame(main_frame)
        top_frame.grid(row=0, column=0, sticky="ew")
        top_frame.columnconfigure(0, weight=1)

        # attach a menubar
        self.build_menu()

        # --- Middle Row: Avatar, Buttons (Left) + Chat Output (Right) ---
        middle_frame = tk.Frame(main_frame)
        middle_frame.grid(row=1, column=0, sticky="nsew", pady=5)

        # Left and right sections both expand properly
        middle_frame.columnconfigure(0, weight=1)   # left panel (avatar)
        middle_frame.columnconfigure(1, weight=3)   # right panel (chat)
        middle_frame.rowconfigure(0, weight=1)

        # ----- LEFT SIDE -----
        left_frame = tk.Frame(middle_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        middle_frame.rowconfigure(0, weight=1)
        middle_frame.columnconfigure(0, weight=1)

        self.config = ConfigManager().load()

        # Avatar
        avatar_path = self.config["app"]["avatar_image"]
        self.image_label = tk.Label(left_frame)
        self.image_label.pack(side="top", pady=(10, 10))
        if os.path.exists(avatar_path):
            self.original_avatar = Image.open(avatar_path)
            self._resize_avatar() 

        # Avatar sizing constraints
        self.avatar_min_size = 256    # minimum dimension
        self.avatar_max_size = 512    # maximum dimension

        # Track current displayed image separately
        self.avatar_image = self.config["app"]["avatar_image"]

        # Bind resize event
        self.root.bind("<Configure>", self._resize_avatar)

        # --- Bottom Controls Container ---
        controls_container = tk.Frame(left_frame)
        controls_container.pack(side="bottom", fill="x", pady=(8, 5))

        # --- Row 1: Audio + Clear + Settings + Generate ---
        row1 = tk.Frame(controls_container)
        row1.pack(fill="x", pady=(2, 3))

        # Audio checkbox
        self.audio_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            row1,
            text="Audio",
            variable=self.audio_var,
            command=lambda: self.toggle_audio(self.audio_var.get())
        ).pack(side="left", padx=(5, 8))

        # Clear Images button
        tk.Button(
            row1,
            text="🗑 Clear",
            width=7,
            command=self.clear_embedded_images
        ).pack(side="left", padx=(2, 6))

        # Image Settings button
        tk.Button(
            row1,
            text="⚙",
            width=4,
            command=self.open_image_settings
        ).pack(side="left", padx=(2, 6))

        # Generate button
        tk.Button(
            row1,
            text="🖼 Generate",
            width=9,
            command=self.generate_image_from_prompt
        ).pack(side="left", padx=(2, 6))

        # --- Row 2: Dropdown + checkboxes ---
        self.use_guidance_var = tk.BooleanVar(value=False)
        self.use_controlnet_var = tk.BooleanVar(value=False)
        self.use_multilayer_var = tk.BooleanVar(value=False)

        row2 = tk.Frame(controls_container)
        row2.pack(fill="x", pady=(2, 3))

        # Dropdown
        self.image_model_var = tk.StringVar(value="uncanny-valley-vpred-v1-sdxl")
        model_options = self.get_available_models()
        self.model_dropdown = ttk.Combobox(
            row2,
            textvariable=self.image_model_var,
            values=model_options,
            state="readonly",
            width=28
        )
        self.model_dropdown.pack(side="left", fill="x", expand=True, padx=(5, 10))

        # Use Current Image
        tk.Checkbutton(
            row2,
            text="Use Current Image",
            variable=self.use_guidance_var
        ).pack(side="left", padx=(5, 8))

        # ControlNet toggle
        tk.Checkbutton(
            row2,
            text="ControlNet",
            variable=self.use_controlnet_var
        ).pack(side="left", padx=(5, 8))

        # Multi-layer toggle
        tk.Checkbutton(
            row2,
            text="Multi-Layer",
            variable=self.use_multilayer_var
        ).pack(side="left", padx=(5, 8))

        # Progress bar (below both rows)
        self.progress = ttk.Progressbar(
            controls_container,
            orient="horizontal",
            mode="determinate"
        )
        self.progress.pack(fill="x", padx=(5, 5), pady=(5, 5))
        self.progress["value"] = 0

        # ----- RIGHT SIDE -----
        right_frame = tk.Frame(middle_frame)
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)

        # Scrollable AI chat output area
        self.ai_scroll = tk.Scrollbar(right_frame, orient="vertical")
        self.ai_scroll.grid(row=0, column=1, sticky="ns")

        self.ai_output_box = tk.Text(
            right_frame,
            width=75,
            height=20,
            wrap="word",
            font=("Segoe UI", 10),
            yscrollcommand=self.ai_scroll.set,
            state="normal",
            bg="#1e1e1e",
            fg="#dcdcdc"
        )
        self.ai_output_box.grid(row=0, column=0, sticky="nsew")
        self.ai_scroll.config(command=self.ai_output_box.yview)
        self.ai_output_box.insert(tk.END, "Chat output...\n")
        self.ai_output_box.config(state="disabled")

         # --- Bottom Chat Section (single-row layout) ---
        chat_frame = tk.Frame(main_frame, bg="#2b2b2b")
        chat_frame.grid(row=3, column=0, sticky="ew", pady=(5, 5), padx=(5, 5))
        main_frame.columnconfigure(0, weight=1)

        # Configure grid columns so entry expands but buttons stay fixed
        chat_frame.columnconfigure(0, weight=1)  # entry expands
        chat_frame.columnconfigure(1, weight=0)
        chat_frame.columnconfigure(2, weight=0)

        # Entry box — width roughly aligned with output box
        self.ai_entry = tk.Entry(chat_frame, font=("Segoe UI", 10))
        self.ai_entry.insert(0, "Chat...")
        self.ai_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5), pady=(0, 3))

        # Remove placeholder and bind Enter
        self.ai_entry.bind("<FocusIn>", self.clear_placeholder)
        self.ai_entry.bind("<Return>", self.submit_ai_text)
        self.ai_entry.bind("<KeyRelease>", self.focus_send_button)

        # Speak button — near middle-right side
        mic_button = tk.Button(
            chat_frame,
            text="🎙 Speak",
            width=8,
            command=self.speech_to_text
        )
        mic_button.grid(row=0, column=2, sticky="e", padx=(5, 0))

        # Send button — next to Speak
        self.ai_send_button = tk.Button(
            chat_frame,
            text="Send",
            width=8,
            command=lambda: self.display_ai_response(self.ai_entry.get())
        )
        self.ai_send_button.grid(row=0, column=1, sticky="e", padx=(5, 0))

        # ai output frame
        ai_output_frame = tk.Frame(main_frame)
        ai_output_frame.grid(row=4, column=0, sticky="w", pady=(2, 0))
        
        # timer label under AI output box
        # Bottom bar for response time and info
        bottom_bar = tk.Frame(main_frame)
        bottom_bar.grid(row=4, column=0, sticky="ew", padx=5, pady=(3, 3))
        bottom_bar.columnconfigure(1, weight=1)

        self.response_time_label = tk.Label(bottom_bar, text="Response time: —", fg="#aaa", bg="#1e1e1e")
        self.response_time_label.pack(side="right", padx=10)

        # insert placeholder text at start
        self.ai_output_box.insert(tk.END, "Chat output...\n")
        self.ai_output_box.config(state="disabled")

        # initial render from service
        # self.render_summary()

        self.img_settings = {
            "steps": 22,
            "guidance": 7.5,
            "width": 512,
            "height": 512,
            "model": "stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf",
            "style": "sexy"
        }

        # Allow full window resizing
        self.root.minsize(900, 600)  # or smaller if desired
        self.root.geometry("")  # auto-size on startup
        self.root.resizable(True, True)

    # ------------------- HELPERS -------------------
    def get_available_models(self):
        model_dir = ConfigManager().get_model_dir("image_model")
        model_list = []
        for item in os.listdir(model_dir):
            full = os.path.join(model_dir, item)
            if os.path.isdir(full) or item.endswith((".gguf", ".safetensors", ".ckpt")):
                model_list.append(item)
        return sorted(model_list)
    
    def _resize_avatar(self, event=None):
        """
        Dynamically resize avatar image within min/max limits,
        maintaining aspect ratio as the window resizes.
        """
        try:
            if not hasattr(self, "original_avatar"):
                return  # no image loaded yet

            # compute target width based on 20–25% of total window width
            total_w = self.root.winfo_width()
            target_w = max(self.avatar_min_size, min(self.avatar_max_size, total_w // 4))
            img = self.original_avatar.copy()
            ratio = target_w / img.width
            target_h = int(img.height * ratio)

            from PIL import ImageTk
            resized = img.resize((target_w, target_h), Image.LANCZOS)
            self.avatar_image = ImageTk.PhotoImage(resized)
            self.image_label.config(image=self.avatar_image,
                                    width=target_w,
                                    height=target_h)
        except Exception as e:
            print(f"[WARN] Avatar resize failed: {e}")

    def insert_chat_image(self, image_path: str):
        """
        Insert a generated image directly into the chat output box,
        scaled to 1/3 of its width (~250–300px).
        """
        from PIL import Image, ImageTk

        try:
            img = Image.open(image_path)
            chat_width = int(self.ai_output_box.winfo_width() or 900)
            target_width = chat_width // 3
            ratio = target_width / img.width
            target_height = int(img.height * ratio)
            img = img.resize((target_width, target_height), Image.LANCZOS)

            tk_img = ImageTk.PhotoImage(img)
            self.ai_output_box.image_create(tk.END, image=tk_img)
            self.ai_output_box.insert(tk.END, "\n")
            # Keep reference so Tk doesn’t garbage collect it
            if not hasattr(self, "_chat_images"):
                self._chat_images = []
            self._chat_images.append(tk_img)
            self.ai_output_box.see(tk.END)
        except Exception as e:
            self.custom_message_popup("Image Error", f"Could not insert image: {e}", msg_type="error")
            
    def clear_embedded_images(self):
        """
        Remove generated images displayed in the interface.
        """
        try:
            if hasattr(self, "generated_image_labels"):
                for lbl in self.generated_image_labels:
                    lbl.destroy()
                self.generated_image_labels.clear()
                print("[INFO] Cleared embedded images.")
            else:
                print("[INFO] No embedded images found to clear.")
        except Exception as e:
            print(f"[WARN] Failed to clear images: {e}")
            
    def clear_model_cache(self):
        """
        Clear cached pipelines to free GPU memory.
        Shows a popup confirmation once complete.
        """
        try:
            clear_pipeline_cache()
            self.custom_message_popup("Cache Cleared", "All cached models have been released from memory.")
        except Exception as e:
            self.show_async_error("Cache Clear Failed", e)

    def edit_positive_prompt(self):
        path = "positive_prompt.txt"
        current = open(path, "r", encoding="utf-8").read() if os.path.exists(path) else ""
        new = self.custom_input_popup("Edit Positive Prompt", current, "Save", show_cancel=True, multiline=True)
        if new:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new)
            self.custom_message_popup("Saved", "Positive prompt updated.")

    def edit_negative_prompt(self):
        path = "negative_prompt.txt"
        current = open(path, "r", encoding="utf-8").read() if os.path.exists(path) else ""
        new = self.custom_input_popup("Edit Negative Prompt", current, "Save", show_cancel=True, multiline=True)
        if new:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new)
            self.custom_message_popup("Saved", "Negative prompt updated.")

    def show_async_error(self, title: str, exception: Exception):
        """
        Safely display an exception message from a background thread.
        """
        import traceback
        tb = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        self.root.after(0, lambda: self.custom_message_popup(title, tb, msg_type="error"))
    
    def custom_input_popup(
    self,
    title: str,
    prompt: str = "",
    ok_text: str = "OK",
    show_cancel: bool = False,
    multiline: bool = False
) -> str | None:
        """
        Custom popup dialog for text input.
        - title: popup window title
        - prompt: text shown above or pre-filled inside the box
        - ok_text: label for the confirmation button ("OK" or "Save")
        - show_cancel: whether to display a Cancel button
        - multiline: whether to use a Text box (multi-line) or Entry (single line)
        Returns the entered string or None if canceled/closed.
        """
        popup = tk.Toplevel(self.root)
        popup.title(title)
        width, height = (400, 250) if multiline else (300, 150)
        self.center_popup(popup, width, height)

        # label for instruction or context
        if not multiline and prompt:
            tk.Label(
                popup,
                text=prompt,
                font=("Segoe UI", 9),
                wraplength=width - 40,
                justify="left"
            ).pack(pady=(10, 5), padx=10)

        result = {"value": None}

        # multi-line text or single-line entry
        if multiline:
            text_widget = tk.Text(
                popup,
                width=45,
                height=6,
                wrap="word",
                font=("Segoe UI", 10),
                bg="#1e1e1e",
                fg="#dcdcdc",
                insertbackground="#ffffff"
            )
            text_widget.insert("1.0", prompt or "")
            text_widget.pack(padx=10, pady=(10, 10), fill="both", expand=True)
            text_widget.focus_set()
        else:
            text_widget = tk.Entry(popup, width=40)
            text_widget.insert(0, prompt or "")
            text_widget.pack(pady=10)
            text_widget.focus_set()

        # button frame for OK / Save / Cancel
        btn_frame = tk.Frame(popup, bg="#2b2b2b")
        btn_frame.pack(pady=(5, 10))

        def on_ok(event=None):
            val = text_widget.get("1.0", "end-1c") if multiline else text_widget.get()
            result["value"] = val.strip()
            popup.destroy()

        def on_cancel():
            result["value"] = None
            popup.destroy()

        ok_btn = tk.Button(
            btn_frame,
            text=ok_text,
            bg="#444444",
            fg="#ffffff",
            activebackground="#666666",
            activeforeground="#ffffff",
            command=on_ok
        )
        ok_btn.pack(side="left", padx=5)

        if show_cancel:
            cancel_btn = tk.Button(
                btn_frame,
                text="Cancel",
                bg="#444444",
                fg="#ffffff",
                activebackground="#666666",
                activeforeground="#ffffff",
                command=on_cancel
            )
            cancel_btn.pack(side="left", padx=5)

        popup.bind("<Return>", on_ok)
        self.root.wait_window(popup)
        return result["value"]
    
    def custom_message_popup(self, title: str, message: str, msg_type: str = "info"):
        """
        Custom message popup to replace default messagebox dialogs.
        - title: window title text
        - message: main message content
        - msg_type: "info", "error", "warning" (changes color scheme)
        Enhanced version: wraps text and auto-sizes for readability.
        """

        popup = tk.Toplevel(self.root)
        popup.title(title)

        # determine size based on message length
        base_width = 350
        base_height = 180
        if len(message) > 150:
            base_height = 250
        if len(message) > 300:
            base_height = 320

        self.center_popup(popup, base_width, base_height)

        # choose colors based on message type
        if msg_type == "error":
            bg_color = "#5c2b2b"
            fg_color = "#ffcccc"
        elif msg_type == "warning":
            bg_color = "#5c5c2b"
            fg_color = "#ffffcc"
        else:  # info
            bg_color = "#2b2b2b"
            fg_color = "#ffffff"

        popup.configure(bg=bg_color)

        # scrollable frame for long messages
        import tkinter.scrolledtext as st
        text_area = st.ScrolledText(
            popup,
            wrap="word",
            height=6,
            font=("Segoe UI", 10),
            bg=bg_color,
            fg=fg_color,
            relief="flat",
            borderwidth=0
        )
        text_area.insert("1.0", message)
        text_area.config(state="disabled")
        text_area.pack(pady=(15, 10), padx=10, fill="both", expand=True)

        # OK button to close popup
        ok_button = tk.Button(
            popup,
            text="OK",
            bg="#444444",
            fg="#ffffff",
            activebackground="#666666",
            activeforeground="#ffffff",
            command=popup.destroy
        )
        ok_button.pack(pady=(0, 10))
        ok_button.focus_set()
        popup.bind("<Return>", lambda event=None: popup.destroy())
        return popup

    def clear_placeholder(self, event):
        """
        Remove placeholder text when user clicks into the entry box.
        """
        if self.ai_entry.get().strip() == "Chat...":
            self.ai_entry.delete(0, tk.END)
            self.ai_entry.unbind("<FocusIn>")

    def focus_send_button(self, event):
        """
        If the user has started typing real text,
        keep focus in the entry box until Enter is pressed.
        """
        current_text = self.ai_entry.get().strip()
        if current_text and current_text != "Chat...":
            # keep focus in the entry so user can continue typing
            self.ai_entry.focus_set()

    def _handle_ai_input(self, user_input: str):
        """
        Core logic for processing AI input and updating UI.
        Handles AI input: shows 'processing...', then replaces it with the LLM's reply.
        """
        user_input = user_input.strip()
        if not user_input or user_input == "Chat...":
            return

        # display the user's input in the chat box
        self.ai_output_box.config(state="normal")
        # clear the placeholder text the first time the user sends a message
        current_text = self.ai_output_box.get("1.0", tk.END).strip()
        if current_text.startswith("Chat"):
            self.ai_output_box.delete("1.0", tk.END)
        self.ai_output_box.insert(tk.END, f"You: {user_input}\n")

        # record the index before inserting placeholder
        insert_start = self.ai_output_box.index(tk.END)
        self.ai_output_box.see(tk.END)
        self.ai_output_box.config(state="disabled")

        import time
        start_time = time.time()
        self.root.after(0, lambda: self.response_time_label.config(text="Processing...", fg="#888888"))

        # ---------------- async LLM section ----------------
        def update_text_callback(reply_text: str):
            """Executed on the main Tkinter thread once the LLM reply is ready."""
            end_time = time.time()
            elapsed = end_time - start_time

            # save user/AI exchange to log file
            # sanitize and display assistant response
            reply_text = reply_text.lstrip().replace("Assistant:", "").replace("Response:", "").strip()
            self.service.save_session_log(user_input, reply_text)

            # update chat output
            self.ai_output_box.config(state="normal")

            # Clean up duplicate prefixes from LLM output
            reply_text = (
                reply_text
                .replace("Assistant:", "")
                .replace("Response:", "")
                .replace("User:", "")
                .replace("System Prompt:", "")
                .replace("Current Model:", "")
                .replace(self.service.get_prompt() or "", "")
                .replace(self.service.get_current_model() or "", "")
                .strip()
            )

            # Insert assistant name label
            self.ai_output_box.insert(insert_start, f"Verita: {reply_text}\n\n")

            if any(w in user_input.lower() for w in ["diagram", "image", "visualize", "show me", "picture"]):
                self.ai_output_box.insert(tk.END, "Tip: Click “🖼 Generate Image” to create a diagram.\n\n")

            self.ai_output_box.config(state="disabled")
            self.ai_output_box.see(tk.END)

            # determine color based on elapsed time
            if elapsed < 2:
                color = "green"
            elif elapsed < 5:
                color = "orange"
            else:
                color = "red"

            # display timing and backend info
            backend_info = self.service.responses.backend_info
            self.response_time_label.config(
                text=f"Response time: {elapsed:.2f} seconds   |   {backend_info}",
                fg=color,
                anchor="e"
            )

            # update persistent chat history
            full_text = self.ai_output_box.get("1.0", "end-1c")
            self.service.update_chat_log(full_text)

            # optional text-to-speech playback
            self.service.speak_if_enabled(reply_text)

        # submit background inference job (non-blocking)
        self.service.responses.reply_async(
            user_input,
            lambda reply_text: self.root.after(0, lambda: update_text_callback(reply_text))
        )

        # clear input field after sending
        self.ai_entry.delete(0, tk.END)
        self.ai_entry.focus_set()
        
    def speech_to_text(self):
        """
        Convert speech input to text and insert it into the entry box.
        """
        import threading

        def record_audio():
            try:
                import speech_recognition as sr
                r = sr.Recognizer()

                # adjust sensitivity to ambient noise
                with sr.Microphone() as source:
                    popup = self.custom_message_popup("Listening", "Speak now…", msg_type="info")

                    # optional ambient calibration (increases reliability in noisy rooms)
                    r.adjust_for_ambient_noise(source, duration=0.8)

                    # increase energy threshold for clarity and allow longer phrases
                    r.energy_threshold = 300
                    r.pause_threshold = 1.2  # seconds of silence before stopping
                    r.phrase_threshold = 0.3
                    r.non_speaking_duration = 0.5

                    # listen for speech up to 20 seconds total
                    audio = r.listen(source, timeout=8, phrase_time_limit=20)

                # recognize speech with Google Speech Recognition
                text = r.recognize_google(audio)
                
                # close the listening popup
                if popup and popup.winfo_exists():
                    popup.destroy()

                # update GUI input box
                self.ai_entry.delete(0, tk.END)
                self.ai_entry.insert(0, text)
                self._handle_ai_input(text)

            except sr.WaitTimeoutError:
                self.custom_message_popup("Timeout", "No speech detected. Please try again.", msg_type="warning")
            except sr.UnknownValueError:
                self.custom_message_popup("Error", "Speech was unclear. Please try again.", msg_type="error")
            except sr.RequestError as e:
                self.custom_message_popup("Error", f"Speech API unavailable: {e}", msg_type="error")
            except Exception as e:
                self.custom_message_popup(
                    "Speech Error",
                    f"Speech recognition failed: {e}\nTry reinstalling PyAudio or checking your microphone.",
                    msg_type="error"
                )

        threading.Thread(target=record_audio, daemon=True).start()

    def submit_ai_text(self, event=None):
        """
        Triggered when pressing Enter.
        """
        self._handle_ai_input(self.ai_entry.get())

    def display_ai_response(self, user_input: str):
        """
        Triggered when clicking Send.
        """
        self._handle_ai_input(user_input)

    def toggle_controlnet(self, enabled):
        self.use_controlnet_var.set(enabled)
        print(f"[INFO] ControlNet {'enabled' if enabled else 'disabled'}")

    def toggle_multilayer(self, enabled):
        self.use_multilayer_var.set(enabled)
        print(f"[INFO] Multi-layer {'enabled' if enabled else 'disabled'}")
    
    def update_progress(self, value):
        self.progress["value"] = value
        self.root.update_idletasks()

    # ------------------- EVENT HANDLERS -------------------
    def on_add_or_edit_entry(self, entry_type: EntryType):
        """
        Event handler for Goal/Skill/Session/Notes buttons.
        Steps:
            - Open custom popup for user input.
            - If Goal then create a GoalLog (with status).
            - If Notes then create a ReflectionLog (with mood analysis).
            - Otherwise then use normal LearningLog via service.
            - Re-render summary output.
        """
        text = self.custom_input_popup("Input", f"Enter your {entry_type.value}:")
        if not text:
            return  # user canceled

        # sentiment for all entries
        self.service.set_entry(entry_type, text)

        # self.render_summary()

    def clear_entries(self) -> None:
        """
        Clear all entries from the service and refresh display.
        """
        self.service.clear()
        self.render_summary()
        self.custom_message_popup("Cleared", "All entries have been cleared.", msg_type="info")

    # ------------------- MENU & FILE OPS -------------------
    def build_menu(self):
        """
        Build the top menubar with File menu options.
        """
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save", command=self.save_entries) # save data entries in json formatted file
        file_menu.add_command(label="Load", command=self.load_entries) # load data entries in json formatted file
        file_menu.add_command(label="Export CSV", command=self.export_csv)  # export history to Excel-readable format
        file_menu.add_command(label="View History", command=self.show_history)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        self.root.config(menu=menubar)
        
        # add llm model selection to menu bar
        llm_menu = tk.Menu(menubar, tearoff=0)
        
        # display current model name (disabled so not clickable)
        current_model = self.service.get_current_model() if hasattr(self.service, "get_current_model") else "Unknown"
        llm_menu.add_command(label=f"Current Model: {current_model}", state="disabled")

        # separator then load option
        llm_menu.add_separator()
        llm_menu.add_command(label="Select Model", command=self.load_llm) # change the llm in use
        llm_menu.add_command(label="Change Avatar", command=self.change_avatar) # add Change Avatar option
        menubar.add_cascade(label="LLM", menu=llm_menu)
        
        # store a reference so we can update the label later when model changes
        self.llm_menu = llm_menu
        
        # prompt menu for viewing/editing/saving prompt
        prompt_menu = tk.Menu(menubar, tearoff=0)
        prompt_menu.add_command(label="View/Edit Prompt", command=self.edit_prompt)
        prompt_menu.add_command(label="Save Prompt", command=self.save_prompt)
        prompt_menu.add_command(label="Current Prompt", command=self.load_prompt)
        prompt_menu.add_separator()
        prompt_menu.add_command(label="Edit Positive Prompt", command=self.edit_positive_prompt)
        prompt_menu.add_command(label="Edit Negative Prompt", command=self.edit_negative_prompt)
        menubar.add_cascade(label="Prompt", menu=prompt_menu)

        cache_menu = tk.Menu(menubar, tearoff=0) 
        cache_menu.add_command(label="Clear Model Cache", command=self.clear_model_cache)
        menubar.add_cascade(label="Cache", menu=cache_menu)

        # session controls
        session_menu = tk.Menu(menubar, tearoff=0)
        session_menu.add_command(label="Refresh Session", command=self.refresh_session)
        menubar.add_cascade(label="Session", menu=session_menu)

        # add Entries menu with entry-related actions
        entries_menu = tk.Menu(menubar, tearoff=0)
        entries_menu.add_command(label="Goal", command=lambda: self.on_add_or_edit_entry(EntryType.Goal))
        entries_menu.add_command(label="Skill", command=lambda: self.on_add_or_edit_entry(EntryType.Skill))
        entries_menu.add_command(label="Session", command=lambda: self.on_add_or_edit_entry(EntryType.Session))
        entries_menu.add_command(label="Notes", command=lambda: self.on_add_or_edit_entry(EntryType.Notes))
        entries_menu.add_separator()
        entries_menu.add_command(label="Clear", command=self.clear_entries)
        menubar.add_cascade(label="Entries", menu=entries_menu)

    def save_entries(self):
        """
        Save all current entries to a JSON file.
        Explicitly writes base attributes and subclass-specific ones.
        - LearningLog = entry_type, text, timestamp, mood
        - GoalLog = adds 'status'
        - ReflectionLog = keeps 'mood'
        """
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON files", "*.json")]
        )
        if not file_path:
            return

        history = self.service.snapshot().entries

        from memory.domain_models import GoalLog, ReflectionLog

        export_dict = {}

        for et, logs in history.items():
            if logs:
                export_dict[et.value] = []
                for rec in logs:
                    # base record dictionary
                    record_dict = {
                        "entry_type": et.value,
                        "text": rec.text,
                        "timestamp": rec.timestamp,
                        "mood": getattr(rec, "mood", "")
                    }

                    # add subclass-specific attributes
                    if isinstance(rec, GoalLog):
                        record_dict["status"] = rec.status
                    elif isinstance(rec, ReflectionLog):
                        record_dict["mood"] = rec.mood  # mood is present

                    export_dict[et.value].append(record_dict)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(export_dict, f, indent=4)

        self.custom_message_popup("Saved", f"Entries saved to {file_path}", msg_type="info")

    def load_entries(self):
        """
        Load entries from a JSON file.
        Reconstructs the correct class type:
        - GoalLog if 'status' field is present
        - ReflectionLog if entry_type == 'Notes'
        - LearningLog otherwise
        """
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not file_path:
            return

        from memory.domain_models import GoalLog, ReflectionLog, LearningLog, EntryType

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # reset current state before loading
            self.service._state.entries = {e: [] for e in EntryType}

            for etype_str, records in data.items():
                etype = EntryType(etype_str)
                for rec in records:
                    text = rec.get("text", "")
                    timestamp = rec.get("timestamp", "")
                    mood = rec.get("mood", "")

                    if "status" in rec:
                        # build GoalLog
                        entry = GoalLog(etype, text, timestamp=timestamp, mood=mood, status=rec["status"])
                    elif etype == EntryType.Notes:
                        # build ReflectionLog
                        entry = ReflectionLog(etype, text, timestamp=timestamp, mood=mood)
                    else:
                        # build base LearningLog
                        entry = LearningLog(etype, text, timestamp=timestamp, mood=mood)

                    self.service._state.entries[etype].append(entry)

            self.render_summary()
            self.custom_message_popup("Loaded", f"Entries loaded from {file_path}", msg_type="info")

        except Exception as e:
            self.custom_message_popup("Error", "Failed to load file!", msg_type="error")

    def show_history(self):
        """
        Display the AI chat history from chat_history.txt.
        Opens the saved transcript file (if it exists) in a scrollable popup.
        """
        import os

        log_file = "chat_history.txt"
        if not os.path.exists(log_file):
            self.custom_message_popup("Chat History", "No chat history found yet.")
            return

        popup = tk.Toplevel(self.root)
        popup.title("Chat History")
        self.center_popup(popup, 650, 450)

        scrollbar = tk.Scrollbar(popup)
        scrollbar.pack(side="right", fill="y")

        text_area = tk.Text(popup, wrap="word", yscrollcommand=scrollbar.set)
        text_area.pack(fill="both", expand=True)
        scrollbar.config(command=text_area.yview)

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()
            text_area.insert(tk.END, content)
        except Exception as e:
            text_area.insert(tk.END, f"Failed to read log: {e}")

        text_area.config(state="disabled")

    def load_llm(self):
        """
        Allows the user to select and load a different local LLM (GGUF file).
        """
        file_path = filedialog.askopenfilename(
            title="Select GGUF Model File",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
        )
        if not file_path:
            return
        result = self.service.set_llm(file_path)
        self.custom_message_popup("LLM Model", result, msg_type="info")
        
        # update menu label to show new model name
        current_model = self.service.get_current_model()
        self.llm_menu.entryconfig(0, label=f"Current Model: {current_model}")

    def center_popup(self, popup, width, height):
        """
        Center any popup relative to the main app window.
        """
        self.root.update_idletasks()

        # grab the info for the root frame
        main_x = self.root.winfo_x()
        main_y = self.root.winfo_y()
        main_w = self.root.winfo_width()
        main_h = self.root.winfo_height()

        # calculate horizontal offset: start from the root x position,
        # then add half the root width, and subtract half the popup width
        pos_x = main_x + (main_w // 2) - (width // 2)

        # calculate vertical offset: start from the root y position,
        # then add half the root height, and subtract half the popup height
        pos_y = main_y + (main_h // 2) - (height // 2)

        # apply the new location for the popup by changing it's geometry
        popup.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
        
    def change_avatar(self):
        """
        Allow user to choose a new avatar image, scale it to fixed bounds,
        update display immediately, and persist the path to settings.json.
        """
        file_path = filedialog.askopenfilename(
            title="Select Avatar Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")]
        )
        if not file_path:
            return

        try:
            from PIL import Image, ImageTk

            # open the selected image
            img = Image.open(file_path)

            # store as new original for resizing
            self.original_avatar = img.copy()

            # immediately render resized version
            self._resize_avatar()

            # update config for persistence
            self.config["app"]["avatar_image"] = file_path
            self.current_image_path = file_path

            # save config safely
            from app_core.config_manager import ConfigManager
            cfg = ConfigManager()
            cfg.data = self.config
            cfg.save()

            self.custom_message_popup("Avatar Changed", "AI avatar updated successfully.")
            # --- Sync avatar path to config and memory ---
            self.current_image_path = file_path
            cfg = ConfigManager().load()
            cfg["app"]["avatar_image"] = file_path
            ConfigManager().save(cfg)
            print(f"[INFO] Updated avatar path: {file_path}")

        except Exception as e:
            self.custom_message_popup("Error", f"Failed to change avatar: {e}", msg_type="error")
        
    def edit_prompt(self):
        """
        Open a popup to view or edit the current system prompt.
        """
        current_prompt = self.service.get_prompt() or "You are a friendly learning advisor who motivates students kindly."

        # use the enhanced custom_input_popup
        new_prompt = self.custom_input_popup(
            title="Edit Prompt",
            prompt=current_prompt,
            ok_text="Save",
            show_cancel=True,
            multiline=True
        )

        if new_prompt:
            self.service.set_prompt(new_prompt)
            self.custom_message_popup("Prompt Saved", "System prompt updated successfully.")
            # refresh LLM context to apply new system prompt immediately
            if hasattr(self.service, "responses") and hasattr(self.service.responses, "reset_context"):
                self.service.reset_context(system_prompt=new_prompt)
                self.custom_message_popup("Prompt Applied", "New prompt applied to current chat session.")

    def save_prompt(self):
        """
        Force-save current prompt text.
        """
        text = self.service.get_prompt()
        self.service.set_prompt(text)
        self.custom_message_popup("Prompt", "Prompt saved to prompt.txt.")

    def load_prompt(self):
        """
        Reload prompt from file.
        """
        text = self.service.get_prompt()
        self.custom_message_popup("Prompt Loaded", text or "No saved prompt found.")

    def refresh_session(self):
        """
        Reset LLM context and clear on-screen conversation.
        """
        msg = self.service.reset_context()
        self.ai_output_box.config(state="normal")
        self.ai_output_box.delete("1.0", tk.END)
        self.ai_output_box.insert(tk.END, "Session reset. Ready for a new conversation.\n")
        self.ai_output_box.config(state="disabled")
        self.custom_message_popup("Session", msg)

    def export_csv(self):
        """
        Export all entries (history) to a CSV file.
        Columns: EntryType, Timestamp, Text, Mood, Status
        - GoalLog adds Status
        - ReflectionLog adds Mood
        """
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            return

        history = self.service.snapshot().entries

        from memory.domain_models import GoalLog, ReflectionLog

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # write header row
            writer.writerow(["EntryType", "Timestamp", "Text", "Mood", "Status"])

            # write one row per log entry
            for etype, records in history.items():
                for rec in records:
                    mood = rec.mood if hasattr(rec, "mood") else ""
                    status = ""

                    # handle derived class specifics
                    if isinstance(rec, GoalLog):
                        status = rec.status
                    elif isinstance(rec, ReflectionLog):
                        mood = rec.mood  # reflectionLog should always carry mood

                    writer.writerow([
                        etype.value,
                        rec.timestamp,
                        rec.text,
                        mood,
                        status
                    ])
        self.custom_message_popup("Exported", f"Entries exported to {file_path}", msg_type="info")

    def open_image_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Image Settings")
        self.center_popup(win, 320, 240)

        # fields: steps, guidance, width, height, model
        labels = ["Steps", "Guidance", "Width", "Height", "Model", "Style"]
        keys   = ["steps", "guidance", "width", "height", "model", "style"]
        entries = {}

        for i, (lbl, key) in enumerate(zip(labels, keys)):
            tk.Label(win, text=lbl).grid(row=i, column=0, sticky="e", padx=8, pady=6)
            e = tk.Entry(win, width=22)
            e.grid(row=i, column=1, sticky="w", padx=8, pady=6)
            e.insert(0, str(self.img_settings[key]))
            entries[key] = e

        def save():
            try:
                self.img_settings["steps"]    = int(entries["steps"].get())
                self.img_settings["guidance"] = float(entries["guidance"].get())
                self.img_settings["width"]    = int(entries["width"].get())
                self.img_settings["height"]   = int(entries["height"].get())
                self.img_settings["model"]    = entries["model"].get().strip()
                self.img_settings["style"]    = entries["style"].get().strip()
                win.destroy()
                self.custom_message_popup("Saved", "Image settings updated.")
            except Exception as e:
                self.custom_message_popup("Error", f"Bad input: {e}", msg_type="error")

        tk.Button(win, text="Save", command=save).grid(row=len(labels), column=0, columnspan=2, pady=10)
            
    def generate_image_from_prompt(self):
        self.progress["value"] = 0
        def run():
            try:
                def on_progress(p):
                    self.root.after(0, lambda: self.progress.configure(value=p))

                prompt_text = self.ai_entry.get().strip()
                init_img = self.avatar_image if self.use_guidance_var.get() else None

                path = self.service.generate_concept_image(
                    user_text=prompt_text,
                    steps=self.img_settings["steps"],
                    guidance=self.img_settings["guidance"],
                    width=self.img_settings["width"],
                    height=self.img_settings["height"],
                    model_name=self.image_model_var.get(),
                    style=self.img_settings["style"],
                    init_image=init_img,
                    use_controlnet=self.use_controlnet_var.get(),
                    use_multilayer=self.use_multilayer_var.get(),
                    use_current_image=self.use_guidance_var.get(),
                    progress_callback=on_progress,
                )
                self.root.after(0, lambda: self.insert_chat_image(path))
                self.root.after(1000, lambda: self.progress.configure(value=0))
            except Exception as e:
                self.show_async_error("Image generation failed", e)
            finally:
                self.root.after(0, lambda: self.progress.configure(value=0))

        threading.Thread(target=run, daemon=True).start()
