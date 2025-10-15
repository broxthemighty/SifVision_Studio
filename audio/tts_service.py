import pyttsx3
import threading

class TTSService:
    """
    Text-to-speech. Outs cleanly if pyttsx3 is not installed.
    Keeps UI separate from the business logic.
    """
    
    def __init__(self, enabled=True):
        self.enabled = enabled

    def set_enabled(self, value: bool):
        """Toggle TTS on/off from the checkbox in UI."""
        self.enabled = value

    def speak(self, text: str):
        """
        Speak text asynchronously using pyttsx3.
        A new engine instance is created for each call to avoid
        the Windows 'silent after first run' bug when threaded.
        """
        if not (self.enabled and text):
            return

        def run_tts():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 180)
                for v in engine.getProperty("voices"):
                    if "female" in v.name.lower() or "zira" in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"TTS error: {e}")

        # launch speech in a daemon thread so it doesn't block the GUI
        threading.Thread(target=run_tts, daemon=True).start()

