from llm.llm_service import LlmService
from audio.tts_service import TTSService
from ui.ui_manager import App
from memory.domain_models import LlmState  # or domain if you keep it at root

class AppManager:
    def __init__(self, root):
        # init service and app UI
        self.service = LlmService(LlmState())
        self.tts = TTSService()
        self.app = App(root, self.service)
