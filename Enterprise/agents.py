from .deps import *
from .logger import StructuredLogger
from .browser import BrowserAgent
from .sandbox import SecureSandbox
from .vision import VisionAgent

@dataclass
class AgentMessage:
    sender:    str
    recipient: str
    content:   str
    msg_type:  str = "task"       # task | result | error | broadcast
    timestamp: float = field(default_factory=time.time)


class MessageBus:
    """Thread-safe message bus for inter-agent communication."""
    def __init__(self):
        self._queues: Dict[str, queue.Queue] = {}
        self._lock = threading.Lock()

    def register(self, agent_id: str):
        with self._lock:
            self._queues[agent_id] = queue.Queue()

    def send(self, msg: AgentMessage):
        with self._lock:
            q = self._queues.get(msg.recipient)
            if q:
                q.put(msg)

    def receive(self, agent_id: str, timeout: float = 1.0) -> Optional[AgentMessage]:
        q = self._queues.get(agent_id)
        if q:
            try:
                return q.get(timeout=timeout)
            except queue.Empty:
                pass
        return None

    def broadcast(self, msg: AgentMessage):
        with self._lock:
            for aid, q in self._queues.items():
                if aid != msg.sender:
                    q.put(AgentMessage(msg.sender, aid, msg.content, "broadcast"))


class BaseAgent(threading.Thread):
    def __init__(self, agent_id: str, bus: MessageBus, logger: StructuredLogger):
        super().__init__(daemon=True)
        self.agent_id = agent_id
        self.bus      = bus
        self.logger   = logger
        self._running = threading.Event()
        bus.register(agent_id)

    def handle(self, msg: AgentMessage) -> Optional[str]:
        raise NotImplementedError

    def run(self):
        self._running.set()
        while self._running.is_set():
            msg = self.bus.receive(self.agent_id, timeout=0.5)
            if msg:
                try:
                    reply = self.handle(msg)
                    if reply and msg.msg_type == "task":
                        self.bus.send(AgentMessage(
                            self.agent_id, msg.sender, reply, "result"
                        ))
                except Exception as e:
                    self.logger.error(f"[{self.agent_id}] {e}")

    def stop(self):
        self._running.clear()


class ResearchAgent(BaseAgent):
    """Fetches and summarises web content."""
    def __init__(self, bus, logger, browser: BrowserAgent):
        super().__init__("research_agent", bus, logger)
        self.browser = browser

    def handle(self, msg: AgentMessage) -> Optional[str]:
        url_match = re.search(r'https?://\S+', msg.content)
        if url_match:
            text = self.browser.visit(url_match.group())
            return f"[Research] {text[:500]}"
        return f"[Research] No URL found in: {msg.content}"


class CodingAgent(BaseAgent):
    """Writes and executes code."""
    def __init__(self, bus, logger, sandbox: SecureSandbox):
        super().__init__("coding_agent", bus, logger)
        self.sandbox = sandbox

    def handle(self, msg: AgentMessage) -> Optional[str]:
        code_match = re.search(r'```python\n(.+?)```', msg.content, re.DOTALL)
        if code_match:
            result = self.sandbox.run_python(code_match.group(1))
            return f"[Code] stdout={result['stdout']} stderr={result['stderr']}"
        return f"[Code] No code block found."


class ReasoningAgent(BaseAgent):
    """Uses LLM for reasoning tasks."""
    def __init__(self, bus, logger, inference):
        super().__init__("reasoning_agent", bus, logger)
        self.inference = inference

    def handle(self, msg: AgentMessage) -> Optional[str]:
        answer = self.inference.generate(
            f"Reason step-by-step:\n{msg.content}", max_length=300
        )
        return f"[Reasoning] {answer}"


class VisionAgentWorker(BaseAgent):
    """Handles image understanding tasks."""
    def __init__(self, bus, logger, vision: VisionAgent):
        super().__init__("vision_agent", bus, logger)
        self.vision = vision

    def handle(self, msg: AgentMessage) -> Optional[str]:
        path_match = re.search(r'(/[\w/._-]+\.(png|jpg|jpeg))', msg.content)
        if path_match:
            caption = self.vision.caption(path_match.group(1))
            return f"[Vision] {caption}"
        if "screenshot" in msg.content.lower():
            return f"[Vision] {self.vision.screenshot_and_describe()}"
        return "[Vision] No image path found."


class MultiAgentOrchestrator:
    """
    Routes tasks to specialized agents and collects results.
    """
    def __init__(
        self,
        logger:    StructuredLogger,
        inference,
        browser:   BrowserAgent,
        sandbox:   SecureSandbox,
        vision:    VisionAgent,
    ):
        self.logger = logger
        self.bus    = MessageBus()
        self.bus.register("orchestrator")

        self.agents = [
            ResearchAgent(self.bus, logger, browser),
            CodingAgent(self.bus, logger, sandbox),
            ReasoningAgent(self.bus, logger, inference),
            VisionAgentWorker(self.bus, logger, vision),
        ]
        for a in self.agents:
            a.start()

    def route(self, task: str) -> str:
        task_lower = task.lower()

        if any(k in task_lower for k in ["http", "url", "web", "search"]):
            target = "research_agent"
        elif any(k in task_lower for k in ["code", "python", "script", "run"]):
            target = "coding_agent"
        elif any(k in task_lower for k in ["image", "photo", "screenshot", "vision"]):
            target = "vision_agent"
        else:
            target = "reasoning_agent"

        self.logger.info(f"Routing to {target}: {task[:60]}")
        self.bus.send(AgentMessage("orchestrator", target, task, "task"))

        # Wait for reply
        for _ in range(20):
            msg = self.bus.receive("orchestrator", timeout=0.5)
            if msg:
                return msg.content
            time.sleep(0.1)
        return "No reply from agents."

    def broadcast_task(self, task: str) -> List[str]:
        """Send to ALL agents and collect all replies."""
        self.bus.broadcast(AgentMessage("orchestrator", "*", task, "broadcast"))
        results = []
        deadline = time.time() + 5
        while time.time() < deadline:
            msg = self.bus.receive("orchestrator", timeout=0.5)
            if msg:
                results.append(msg.content)
        return results

    def shutdown(self):
        for a in self.agents:
            a.stop()
