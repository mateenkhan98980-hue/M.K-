from .deps import *
from .logger import StructuredLogger

class SpeechIO:
    """
    Microphone → text (speech_recognition)
    Text → speech (pyttsx3)
    """
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self._tts_engine = None

        if _TTS_OK:
            try:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty("rate", 175)
            except Exception as e:
                self.logger.warn(f"TTS init failed: {e}")

    # ── speech → text ────────────────────────────────────────
    def listen(self, timeout: int = 5) -> str:
        if not _SR_OK:
            return "speech_recognition not installed."
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                self.logger.info("Listening…")
                audio = recognizer.listen(source, timeout=timeout)
            text = recognizer.recognize_google(audio)
            self.logger.info(f"Heard: {text}")
            return text
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            return ""
        except Exception as e:
            self.logger.error(f"Listen error: {e}")
            return ""

    # ── text → speech ────────────────────────────────────────
    def speak(self, text: str):
        if self._tts_engine:
            try:
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
            except Exception as e:
                self.logger.error(f"Speak error: {e}")
        else:
            print(f"[TTS] {text}")

    # ── text → mp3 file ──────────────────────────────────────
    def save_audio(self, text: str, path: str = "/tmp/output.mp3"):
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang="en")
            tts.save(path)
            return path
        except Exception as e:
            return f"gTTS error: {e}"
