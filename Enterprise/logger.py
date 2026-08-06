from .deps import *

class StructuredLogger:
    def __init__(self, log_dir="./logs"):
        Path(log_dir).mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"train_{ts}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def info(self, msg):  self.logger.info(msg)
    def error(self, msg): self.logger.error(msg)
    def warn(self, msg):  self.logger.warning(msg)
