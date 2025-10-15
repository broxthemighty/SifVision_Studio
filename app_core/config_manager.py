import json
import os

class ConfigManager:
    def __init__(self, path="config.json"):
        self.path = path

    def load(self):
        if not os.path.exists(self.path):
            self.save_default()
        with open(self.path, "r") as f:
            return json.load(f)

    def save_default(self):
        default = {
            "model_path": "./models/llama-3-8b.gguf",
            "tts_enabled": True,
            "auto_generate_image": True,
            "theme": "light"
        }
        with open(self.path, "w") as f:
            json.dump(default, f, indent=4)
