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
            cfg_mgr = ConfigManager()
            cfg = cfg_mgr.load()

            # Save image model if available
            image_model = getattr(self.app, "image_model_var", None)
            if image_model:
                cfg["image_model"]["last_used"] = image_model.get()

            # Save avatar path if tracked
            if hasattr(self.app, "avatar_path"):
                cfg["app"]["avatar_image"] = self.app.avatar_path

            # Save current image generation settings
            if hasattr(self.app, "current_image_settings"):
                cfg["image_settings"] = self.app.current_image_settings

            cfg_mgr.data = cfg
            cfg_mgr.save()
            print("[INFO] Settings saved successfully.")

            self.root.quit

        except Exception as e:
            print(f"[WARN] Failed to save settings on exit: {e}")
