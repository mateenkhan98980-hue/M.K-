from .deps import *
from .logger import StructuredLogger

class GUIController:
    """
    Controls desktop GUI via pyautogui.
    All actions are logged and have safety delays.
    """
    SAFE_DELAY = 0.3  # seconds between actions

    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        if _GUI_OK:
            pyautogui.FAILSAFE = True   # move mouse to corner to abort
            pyautogui.PAUSE    = self.SAFE_DELAY

    def _check(self) -> bool:
        if not _GUI_OK:
            self.logger.warn("pyautogui not installed — GUI control unavailable.")
            return False
        return True

    def screenshot(self, save_path: str = "/tmp/gui_screenshot.png") -> str:
        if not self._check(): return ""
        img = ImageGrab.grab()
        img.save(save_path)
        self.logger.info(f"Screenshot saved: {save_path}")
        return save_path

    def click(self, x: int, y: int):
        if not self._check(): return
        self.logger.info(f"Click ({x}, {y})")
        pyautogui.click(x, y)

    def type_text(self, text: str):
        if not self._check(): return
        self.logger.info(f"Type: {text[:30]}…")
        pyautogui.typewrite(text, interval=0.05)

    def hotkey(self, *keys: str):
        if not self._check(): return
        self.logger.info(f"Hotkey: {keys}")
        pyautogui.hotkey(*keys)

    def find_and_click(self, image_path: str) -> bool:
        """Locate image on screen and click it."""
        if not self._check(): return False
        try:
            loc = pyautogui.locateCenterOnScreen(image_path, confidence=0.8)
            if loc:
                pyautogui.click(loc)
                return True
        except Exception as e:
            self.logger.error(f"find_and_click: {e}")
        return False

    def scroll(self, x: int, y: int, clicks: int = 3):
        if not self._check(): return
        pyautogui.scroll(clicks, x=x, y=y)

    def get_screen_size(self) -> Tuple[int, int]:
        if not self._check(): return (0, 0)
        return pyautogui.size()
