from .deps import *
from .logger import StructuredLogger

class VisionAgent:
    """
    BLIP-2 / CLIP for image captioning & classification.
    Falls back to PIL metadata if models not available.
    """
    def __init__(self, logger: StructuredLogger, device: str = "cpu"):
        self.logger = logger
        self.device = device
        self._blip_loaded = False
        self._clip_loaded = False

    def _load_blip(self):
        if self._blip_loaded:
            return
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self._blip_proc  = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self._blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self.device)
            self._blip_loaded = True
            self.logger.info("BLIP model loaded.")
        except Exception as e:
            self.logger.warn(f"BLIP load failed: {e}")

    def caption(self, image_path: str) -> str:
        """Generate a text caption for an image file."""
        if not _PIL_OK:
            return "PIL not installed — cannot process image."
        self._load_blip()
        try:
            img = PILImage.open(image_path).convert("RGB")
            if self._blip_loaded:
                inputs = self._blip_proc(img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self._blip_model.generate(**inputs, max_new_tokens=50)
                return self._blip_proc.decode(out[0], skip_special_tokens=True)
            else:
                return f"Image size: {img.size}, mode: {img.mode}"
        except Exception as e:
            return f"Caption error: {e}"

    def screenshot_and_describe(self) -> str:
        """Take a screenshot and describe it."""
        if not _GUI_OK:
            return "pyautogui/PIL not installed."
        try:
            screenshot = ImageGrab.grab()
            path = "/tmp/screenshot.png"
            screenshot.save(path)
            return self.caption(path)
        except Exception as e:
            return f"Screenshot error: {e}"

    def image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
