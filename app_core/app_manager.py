from llm.llm_service import LlmService
from audio.tts_service import TTSService
from ui.ui_manager import App
from memory.domain_models import LlmState  # or domain if you keep it at root
from app_core.config_manager import ConfigManager

class AppManager:

    def __init__(self, root):
        # init service and app UI
        self.service = LlmService(LlmState())
        self.tts = TTSService()
        self.app = App(root, self.service)

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
        # self.root.destroy()
