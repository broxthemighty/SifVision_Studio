# config_manager.py
import json, os
from pathlib import Path

_DEFAULT = {
    "app": {
        "avatar_image": "images/succ.jpeg",
        "theme": "dark"
    },
    "chat_model": {
        "folder": "chat_model",
        "last_used": "DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q5_K_M.gguf"
    },
    "image_model": {
        "folder": "image_model",
        "last_used": "stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf"
    },
    "image_settings": {
        "width": 512,
        "height": 512,
        "steps": 20,
        "guidance": 7.5
    },
    "prompts": {
        "positive": "config/positive_prompt.txt",
        "negative": "config/negative_prompt.txt"
    }
}

class ConfigManager:
    def __init__(self, path="config/settings.json"):
        self.path = Path(path)
        self.data = {}

    def get_model_dir(self, model_type: str) -> Path:
        """Return an absolute folder path for image_model or chat_model."""
        models_root = self.data.get("models", {}).get("base_folder", "models")
        sub_folder = self.data.get("models", {}).get(model_type, model_type)
        repo_root = Path(__file__).resolve().parents[1]
        path = (repo_root / models_root / sub_folder).resolve()
        os.makedirs(path, exist_ok=True)
        return path
    
    def get_last_used_model(self, model_type: str) -> str:
        return self.data.get("last_used", {}).get(model_type, "")

    def load(self):
        if not self.path.exists():
            self._write(_DEFAULT)
            return _DEFAULT
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Ensure all default keys exist
        for key, value in _DEFAULT.items():
            if key not in data:
                data[key] = value
            elif isinstance(value, dict):
                for subkey, subval in value.items():
                    data[key].setdefault(subkey, subval)

        self.data = data
        return self.data

    def save(self, data=None):
        """Save configuration safely to settings.json"""
        try:
            # Choose what to save
            config_to_save = data or self.data

            if not config_to_save or not isinstance(config_to_save, dict):
                print("[WARN] Skipping save: empty or invalid config object.")
                return

            if "app" not in config_to_save:
                print("[WARN] Skipping save: missing 'app' key.")
                return

            os.makedirs(self.path.parent, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(config_to_save, f, indent=4)
            print(f"[INFO] Config saved to {self.path}")
        except Exception as e:
            print(f"[ERROR] Failed to save config: {e}")

    def _write(self, obj):
        os.makedirs(self.path.parent, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4)