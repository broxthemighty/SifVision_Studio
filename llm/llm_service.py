from memory.domain_models import EntryType, LlmState, LearningLog
from textblob import TextBlob
from llama_cpp import Llama
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache
from audio.tts_service import TTSService
from typing import Optional, Dict          # if not present
from copy import deepcopy                  # snapshot()
import torch, gc                           # GPU suspend/resume
from vision.image_gen import generate_image, generate_image_diffusers # concept image
import os
from app_core.config_manager import ConfigManager
from pathlib import Path
import random

def resolve_model_path(model_name: str, model_type: str = "chat") -> str:
    """
    Resolves an absolute path to a given model file, ensuring no duplicate folder nesting.
    Example:
        model_type="image" and settings["image_model"]["folder"]="models/image_model"
        will yield: <repo_root>/models/image_model/<model_name>
    """
    cfg = ConfigManager().load()
    repo_root = Path(__file__).resolve().parents[1]

    folder = Path(cfg[f"{model_type}_model"]["folder"])
    base_dir = repo_root / folder

    # 1️⃣ If an absolute file path is provided
    if os.path.isabs(model_name) and os.path.exists(model_name):
        return model_name

    # 2️⃣ If the model exists in the configured folder
    candidate = base_dir / model_name
    if candidate.exists():
        return str(candidate)

    # 3️⃣ Search recursively in the configured folder
    if base_dir.exists():
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.lower() == model_name.lower():
                    return str(Path(root) / f)

    # 4️⃣ Final fallback — models/image_model or models/chat_model only ONCE
    fallback = repo_root / "models" / f"{model_type}_model" / model_name
    if fallback.exists():
        return str(fallback)

    # 5️⃣ Throw clear diagnostic
    checked = "\n  ".join(str(x) for x in [candidate, fallback])
    raise FileNotFoundError(
        f"Model file not found.\nChecked:\n  {checked}\nBase dir: {base_dir}\nCWD: {os.getcwd()}"
    )

class LlmService:
    """
    The LearnflowService class for all non-UI functionality.
    It operates on a LearnflowState object, which stores user data.
    """
    from audio.tts_service import TTSService

    def __init__(self, state: Optional[LlmState] = None, model_path: str = None):
        """
        Constructor initializes service with an existing state,
        or creates a new empty LearnflowState if none provided.
        """
        from llm.llm_service import resolve_model_path

        cfg_path = Path(__file__).resolve().parents[1] / "config" / "settings.json"
        print(f"[DEBUG] Expected config path: {cfg_path}")
        cfg = ConfigManager().load()
        print(f"[DEBUG] Loaded config data: {cfg}")

        cfg = ConfigManager().load()
        chat_model_name = cfg["chat_model"].get("last_used", "").strip()

        if not chat_model_name:
            raise ValueError("No chat model configured in settings.json (chat_model.last_used is empty).")

        model_path = resolve_model_path(chat_model_name, "chat")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Resolved chat model path is not a file: {model_path}")

        self._state = state or LlmState()
        self.responses = LlamaEngine(model_path=model_path)  # full ai replies
        prompt_text = self.get_prompt() # pull the prompt text
        if prompt_text:
            self.responses.system_prompt = prompt_text
        self.tts = TTSService(enabled=False)  # audio off by default
        last_model = self.load_last_model()
        if last_model:
            try:
                print(f"[INFO] Loading last model from file: {last_model}")
                self.responses = LlamaEngine(model_path=last_model, n_gpu_layers=100)
                if prompt_text:
                    self.responses.system_prompt = prompt_text
            except Exception as e:
                print(f"[WARN] Could not load saved model: {e}")

    # ------------------- COMMANDS (Mutate State) -------------------

    def set_entry(self, entry_type: EntryType, text: str) -> None:
        """
        Add a new entry to the state.
        Input:
            entry_type - The category of entry (Goal, Skill, Session, Notes).
            text       - The user-provided content string.
        Behavior:
            - Creates a new LearningLog object.
            - Appends it to the list for the given entry_type.
            - Calls write_log() stub for future logfile use.
        """
        # sanitize text
        clean_text = (text or "").strip()

        # delegated mood evaluation to a helper method
        mood = self.analyze_mood(clean_text)

        # create new log record object with mood
        record = LearningLog(entry_type, clean_text, mood=mood)

        # append to the appropriate list in state
        self._state.entries[entry_type].append(record)

        # placeholder hook for logfile writing
        self.write_log(record)

    def clear(self) -> None:
        """
        Reset the entire state back to empty lists.
        Useful for starting over without restarting the program.
        """
        for k in self._state.entries:
            self._state.entries[k] = []
            
    def set_llm(self, model_path: str, gpu_layers: int = 80):
        """
        Dynamically reload the Llama model with a new GGUF file.
        """
        try:
            print(f"Loading new model: {model_path} with {gpu_layers} GPU layers")
            self.responses = LlamaEngine(model_path=model_path, n_gpu_layers=gpu_layers)
            self.save_last_model(model_path)  # NEW
            return f"Model loaded with GPU acceleration: {model_path}"
        except Exception as e:
            return f"Error loading model: {e}"
        
    def get_current_model(self) -> str:
        """
        Return the current LLM model file name (for menu display).
        """
        try:
            import os
            return os.path.basename(self.responses.llm.model_path)
        except Exception:
            return "Unknown Model"

    # ------------------- QUERIES (Read State) -------------------

    def get_entry(self, entry_type: EntryType) -> str:
        """
        Retrieve the most recent entry for a given type.
        Returns:
            - The latest entry text if available.
            - Empty string if no entries exist for this type.
        """
        if self._state.entries[entry_type]:
            return self._state.entries[entry_type][-1].text
        return ""

    def summary(self) -> Dict[str, str]:
        """
        Build a dictionary summary of the most recent entries by type.
        Returns:
            { "Goal": "Finish Week 1", "Notes": "Felt motivated", ... }
        Each value comes from the LearningLog.summary() method.
        """
        result = {}
        for e, records in self._state.entries.items():
            if records:  # only include if there is at least one record
                result[e.value] = records[-1].summary()
        return result

    def snapshot(self) -> LlmState:
        """
        Return a deep copy of the current LearnflowState.
        This allows the GUI to display history safely without
        risking accidental modification of the underlying data.
        """
        return deepcopy(self._state)

    # ------------------- HELPER FUNCTIONS -------------------

    def speak_if_enabled(self, text: str) -> None:
        """
        Speak a line through TTS if the user has audio enabled.
        """
        self.tts.speak(text)

    def analyze_mood(self, text: str) -> str:
        """
        Run sentiment analysis on note text using TextBlob.
        Returns one of: "motivated", "stuck", or "neutral".
        Polarity amounts chosen for simplicity.
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.3:      # positive sentiment
            return "motivated"
        elif polarity < -0.3:   # negative sentiment
            return "stuck"
        else:                   # neutral sentiment
            return "neutral"

    def write_log(self, record: "LearningLog"):
        """
        Append a log entry to a persistent text file (learnflow.log).
        Subclass-aware:
        - GoalLog → includes Status
        - ReflectionLog → includes Mood
        - LearningLog → base summary
        """
        from memory.domain_models import GoalLog, ReflectionLog

        log_file = "learnflow.log"

        # base summary always includes entry type and text
        line = f"[{record.timestamp}] {record.entry_type.value}: {record.text}"

        # add subclass-specific info
        if isinstance(record, GoalLog):
            line += f" (Status: {record.status})"
        elif isinstance(record, ReflectionLog):
            if record.mood:
                line += f" (Mood: {record.mood})"
        elif record.mood:  # base LearningLog may still carry a mood
            line += f" (Mood: {record.mood})"

        # write line to logfile
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            
    # ------------------- SESSION LOGGING -------------------

    def save_session_log(self, user_input: str, ai_output: str):
        """
        Save each user/AI message pair to a session_log.json file.
        The log is later used to continue conversation context.
        """
        import json, os
        log_file = "session_log.json"
        entry = {"user": user_input, "ai": ai_output}

        # load existing log or start new one
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = []
        else:
            data = []

        data.append(entry)

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load_session_log(self) -> list:
        """
        Load the previous session log so the LLM can retain context.
        Returns a list of prior exchanges.
        """
        import json, os
        log_file = "session_log.json"
        if not os.path.exists(log_file):
            return []

        with open(log_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
            
    # --- Session Chat File Logging ---
    def update_chat_log(self, full_text: str, append: bool = False):
        """
        Write or append the complete AI chat text to a persistent log file.
        Called whenever the AI output box is updated.
        """
        log_path = "chat_history.txt"
        mode = "a" if append else "w"
        try:
            with open(log_path, mode, encoding="utf-8") as f:
                f.write(full_text if append else full_text.rstrip() + "\n")
        except Exception as e:
            print(f"[WARN] Failed to write chat log: {e}")
                
    # ------------------- PROMPT MANAGEMENT -------------------

    def set_prompt(self, text: str):
        """
        Save a custom system prompt to prompt.txt.
        """
        with open("prompt.txt", "w", encoding="utf-8") as f:
            f.write(text.strip())

        if hasattr(self, "responses"):
            self.responses.system_prompt = text.strip()

    def get_prompt(self) -> str:
        """
        Load the current system prompt from file or return default.
        """
        import os
        if os.path.exists("prompt.txt"):
            with open("prompt.txt", "r", encoding="utf-8") as f:
                return f.read().strip()
        return getattr(self.responses, "system_prompt", "")
    
    def reset_context(self):
        """
        Clear conversation log and reload the LLM context.
        """
        import os
        if os.path.exists("session_log.json"):
            os.remove("session_log.json")
        new_prompt = self.get_prompt()
        self.responses.reset_context()
        self.responses.system_prompt = new_prompt
        return "Session has been refreshed with updated system prompt."
    
    def save_last_model(self, model_path: str):
        try:
            with open("model.txt", "w", encoding="utf-8") as f:
                f.write(model_path.strip())
            print(f"[INFO] Saved model path to model.txt: {model_path}")
        except Exception as e:
            print(f"[WARN] Could not save model path: {e}")

    def load_last_model(self) -> str:
        import os
        if os.path.exists("model.txt"):
            try:
                with open("model.txt", "r", encoding="utf-8") as f:
                    val = f.read().strip()
                    if val:
                        print(f"[INFO] Loaded last model: {val}")
                        return val
            except Exception as e:
                print(f"[WARN] Could not read model.txt: {e}")
        return ""
    
    def suspend_llm(self):
        try:
            if hasattr(self, "responses") and getattr(self.responses, "llm", None):
                print("[INFO] Suspending LLM GPU context...")
                if hasattr(self.responses.llm, "free"):
                    try: self.responses.llm.free()
                    except Exception: pass
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            print(f"[WARN] Could not suspend LLM: {e}")

    def resume_llm(self):
        try:
            print("[INFO] Resuming LLM GPU context...")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[WARN] Could not resume LLM: {e}")

    def generate_concept_image(
        self,
        user_text: str | None = None,
        steps: int = 20,
        guidance: float = 7.5,
        width: int = 512,
        height: int = 512,
        model_name: str = "models\\Image_Gen\\stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf",
        style=None,
        progress_callback=None,
        init_image: str | None = None,
    ) -> str:
        text = (user_text or "").strip()
        # --- Make image prompt context-aware ---
        if not user_text:
            if getattr(self.responses, "context", None):
                # Combine the last 2–3 exchanges for richer semantic context
                context_snippets = [
                    turn.get("user", "") for turn in self.responses.context[-3:] if turn.get("user")
                ]
                user_text = " ".join(context_snippets).strip()
        if not user_text:
            user_text = "A sexy full-body image of Verita."
            
        if not style:
            style = "sexy"

        cfg = ConfigManager().load()
        positive_path = cfg["prompts"]["positive"]
        negative_path = cfg["prompts"]["negative"]

        def read_prompt(path):
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            return ""

        positive = f"{read_prompt(positive_path)}, Concept: {user_text}, full body, {style or 'realistic detailed'}"
        negative = f"{read_prompt(negative_path)}, cropped, close-up, duplicate, deformed"

        cfg = ConfigManager().load()
        avatar_path = cfg["app"].get("avatar_image")

        init_image = None
        if avatar_path and os.path.exists(avatar_path):
            init_image = avatar_path

        # optionally free VRAM before SD
        self.suspend_llm()
        strength = 0.45 if init_image else 0.0
        try:
            # Resolve model path dynamically
            try:
                model_path = resolve_model_path(model_name, model_type="image")
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Image model could not be resolved: {e}")
            if model_name.endswith(".gguf"):
                # Use C++ CLI generator (fast quantized)
                path = generate_image(
                    prompt=positive,
                    steps=steps,
                    guidance=guidance,
                    size=(width, height),
                    model_name=model_path,
                    progress_callback=progress_callback,
                    negative_prompt=negative,
                    seed = random.randint(0, 999999),
                    init_image=init_image,
                    style=style
                )
            else:
                # Use Diffusers pipeline (PyTorch)
                path = generate_image_diffusers(
                    prompt=positive,
                    steps=steps,
                    guidance=guidance,
                    size=(width, height),
                    model_name=model_path,
                    progress_callback=progress_callback,
                    negative_prompt=negative,
                    seed = random.randint(0, 999999),
                    init_image=init_image,
                    style=style
                )
        finally:
            self.resume_llm()

        print(f"[INFO] Image successfully generated at {path}")
        return path
    
    def reset_context(self, system_prompt: str = None):
        """Reset internal context and optionally update system prompt."""
        self.chat_history = []
        if system_prompt:
            self.system_prompt = system_prompt
        print("[INFO] LLM context reset with new prompt.")

class LlamaEngine:
    """
    Wrapper around llama-cpp to generate responses from a local GGUF model.
    Includes support for persistent prompt and session history.
    """
    _executor = ThreadPoolExecutor(max_workers=2)

    def __init__(self, model_path=None, n_gpu_layers=100):
        if not model_path:
            model_path = resolve_model_path("DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q5_K_M.gguf")

        self.model_path = model_path
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=8192,
            n_threads=16,               # CPU threads for mixed compute
            n_gpu_layers=-1,            # number of layers to offload to GPU
            n_batch=2048,               # process more tokens per pass
            use_mmap=True,
            use_mlock=False,
            flash_attn=True,
            verbose=False,
            temperature=0.7,         
            top_p=0.9,               
            repeat_penalty=1.15,
        )
        self.backend_info = self.get_backend_info() # detect CUDA/CPU backend
        self.system_prompt = "You are a friendly learning advisor named Verita who motivates students kindly."
        self.context = []
        self._lock = threading.Lock()  # guard llama_cpp thread safety
        
    def get_backend_info(self):
        """
        Detect whether CUDA-capable GPU is active and available.
        Combines torch (runtime GPU check) + llama.cpp system info for maximum accuracy.
        """
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_index = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(device_index)
                cuda_version = torch.version.cuda
                return f"GPU Active ({device_name}, CUDA {cuda_version})"
            else:
                # double-check llama.cpp build flags as fallback
                import llama_cpp
                info = llama_cpp.llama_cpp.llama_print_system_info().decode().lower()
                if any(x in info for x in ["cuda", "cublas", "gpu"]):
                    return "GPU Build Detected (Torch CUDA unavailable)"
                return "CPU Only"
        except Exception as e:
            print(f"[WARN] Could not determine backend info: {e}")
            return "Unknown Backend"
        
    @lru_cache(maxsize=128)
    def _cached_reply(self, prompt_key: str) -> str:
        """
        Internal LRU cache for repeated prompts.
        """
        # actual model inference (do not call directly from GUI thread)
        with self._lock:
            response = self.llm(prompt_key,
                        max_tokens=512,
                        temperature=0.7,
                        top_p=0.9,
                        repeat_penalty=1.15,
                        stop=["User:", "Assistant:", "\n\n", "Verita:"],
                        echo=False,
                        )
        return response["choices"][0]["text"].strip()

    def _build_prompt(self, user_text: str) -> str:
        """
        Combine limited recent context for concise replies.
        """
        convo = ""
        for msg in self.context[-2:]:
            convo += f"User: {msg['user']}\nAssistant: {msg['ai']}\n"

        return (
            f"System Prompt: {self.system_prompt.strip()}\n"
            f"{convo}"
            f"User: {user_text.strip()}\n"
            f"Assistant:"
        )
    
    def reply(self, user_text: str) -> str:
        """
        Blocking version of reply (used internally).
        Maintains context and uses caching.
        """
        prompt = self._build_prompt(user_text)
        cache_key = f"{self.model_path}:{self.system_prompt}:{prompt}"
        reply = self._cached_reply(cache_key)
        self.context.append({"user": user_text, "ai": reply})
        if len(self.context) > 10:       # roughly five full turns
            self.context = self.context[-4:]   # keep the last 2 user/ai pairs only
        reply = reply.strip()

        # Strip system prompt and model info if leaked
        for junk in [self.system_prompt, self.model_path, "Current Model:", "Model loaded with GPU acceleration"]:
            if junk in reply:
                reply = reply.replace(junk, "")

        # Clean up redundant phrases and self-introductions
        reply = reply.strip()
        for junk in [self.system_prompt, self.model_path, "Current Model:", "Model loaded with GPU acceleration"]:
            reply = reply.replace(junk, "")
        reply = reply.replace("Assistant:", "").replace("User:", "").strip()

        # Remove introductory fluff (common with instruction-tuned models)
        intro_markers = [
            "as verita", "as your learning advisor", "sure!", "of course", "hello", "hi", "let me explain",
            "i can explain", "here's an explanation", "allow me to", "let's talk about"
        ]
        lower = reply.lower()
        for marker in intro_markers:
            if lower.startswith(marker):
                parts = reply.split(".", 1)
                if len(parts) > 1:
                    reply = parts[1].strip()
                    break

        # Limit length if the model still rambles
        if len(reply.split()) > 120:
            reply = " ".join(reply.split()[:120]) + "..."

        return reply
    
    def reply_async(self, user_text: str, callback):
        """
        Run LLM inference asynchronously via ThreadPoolExecutor.
        The callback receives the generated reply when finished.
        """
        def task():
            try:
                reply = self.reply(user_text)
                callback(reply)
            except Exception as e:
                callback(f"[Error] LLM failed: {e}")

        self._executor.submit(task)

    def reset_context(self):
        """
        Clears in-memory conversation context.
        """
        self.context = []
        self._cached_reply.cache_clear()
    