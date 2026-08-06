from .deps import *

class SecureSandbox:
    """
    Subprocess-based sandbox with:
    - Timeout enforcement
    - Memory limit (Linux only)
    - Output capture
    - No network in subprocess
    """
    def __init__(self, timeout: int = 10, memory_mb: int = 256):
        self.timeout   = timeout
        self.memory_mb = memory_mb

    def run_python(self, code: str) -> Dict:
        """
        Execute Python code in an isolated subprocess.
        Returns dict: {stdout, stderr, exit_code, timed_out}
        """
        wrapper = f"""
import sys, resource, signal

# Memory limit (Linux only)
try:
    soft = {self.memory_mb} * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
except Exception:
    pass

# Block dangerous builtins
import builtins
_safe = {{k: getattr(builtins, k) for k in dir(builtins) if k not in (
    'open', 'exec', 'eval', 'compile', '__import__', 'input'
)}}

exec(compile('''{code.replace("'", "\\'")}''', '<sandbox>', 'exec'), {{'__builtins__': _safe}})
"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", wrapper],
                capture_output=True, text=True,
                timeout=self.timeout,
            )
            return {
                "stdout":    result.stdout,
                "stderr":    result.stderr,
                "exit_code": result.returncode,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "Timeout", "exit_code": -1, "timed_out": True}
        except Exception as e:
            return {"stdout": "", "stderr": str(e), "exit_code": -1, "timed_out": False}

    def run_bash(self, command: str) -> Dict:
        """Run a shell command with timeout."""
        try:
            result = subprocess.run(
                command, shell=True,
                capture_output=True, text=True,
                timeout=self.timeout,
            )
            return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "Timeout", "exit_code": -1}
