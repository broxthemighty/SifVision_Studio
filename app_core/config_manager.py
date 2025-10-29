# config_manager.py
import json, os
from pathlib import Path

_DEFAULT = {
    "app": {
        "avatar_image": "images/succ.jpeg",
        "theme": "dark"
    },
    "chat_model": {
        "folder": "models/chat_model",
        "last_used": "DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q5_K_M.gguf"
    },
    "image_model": {
        "folder": "models/image_model",
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

    def save(self):
        self._write(self.data)

    def _write(self, obj):
        os.makedirs(self.path.parent, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4)