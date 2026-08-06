from .deps import *
from .logger import StructuredLogger
from .vision import VisionAgent
from .gui import GUIController

class ComputerUseAgent:
    """
    Screenshot → Vision describe → plan action → GUI execute loop.
    """
    def __init__(
        self,
        vision:  VisionAgent,
        gui:     GUIController,
        inference,           # InferencePipeline (defined below)
        logger:  StructuredLogger,
    ):
        self.vision    = vision
        self.gui       = gui
        self.inference = inference
        self.logger    = logger

    def observe(self) -> str:
        """Take screenshot and get description."""
        path        = self.gui.screenshot()
        description = self.vision.caption(path) if path else "No screenshot available."
        self.logger.info(f"Observation: {description}")
        return description

    def plan(self, goal: str, observation: str) -> str:
        """Use LLM to plan next GUI action."""
        prompt = (
            f"Goal: {goal}\n"
            f"Current screen: {observation}\n\n"
            "What is the single next GUI action to take?\n"
            "Reply in format: ACTION: <click|type|hotkey|scroll> PARAMS: <details>"
        )
        return self.inference.generate(prompt, max_length=80)

    def execute_plan(self, plan_text: str):
        """Parse and execute a planned action."""
        plan_text = plan_text.lower()

        if "click" in plan_text:
            coords = re.findall(r"\d+", plan_text)
            if len(coords) >= 2:
                self.gui.click(int(coords[0]), int(coords[1]))

        elif "type" in plan_text:
            m = re.search(r'params:\s*(.+)', plan_text)
            if m:
                self.gui.type_text(m.group(1).strip())

        elif "hotkey" in plan_text:
            m = re.search(r'params:\s*(.+)', plan_text)
            if m:
                keys = m.group(1).strip().split("+")
                self.gui.hotkey(*keys)

    def run_task(self, goal: str, max_steps: int = 5) -> List[Dict]:
        """Full observe → plan → act loop."""
        history = []
        for step in range(max_steps):
            obs    = self.observe()
            plan   = self.plan(goal, obs)
            self.execute_plan(plan)
            history.append({"step": step, "obs": obs, "plan": plan})
            self.logger.info(f"Step {step}: {plan}")
        return history
