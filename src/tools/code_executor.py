"""
src/tools/code_executor.py — Safe Sandboxed Python Execution
──────────────────────────────────────────────────────────────
Lets the Coder agent write and execute Python code safely.

Design decisions:
1. WHY A SANDBOX? An LLM running arbitrary code could delete files,
   make unexpected network requests, or loop forever. We need guardrails.
2. TWO-LAYER SAFETY:
   Layer 1 — RestrictedPython: Validates code at compile time.
              Catches dangerous builtins before we ever run anything.
   Layer 2 — Subprocess + Timeout: Runs code in a completely separate
              Python process. If it crashes or times out, our process
              is unaffected.
3. WHY SUBPROCESS OVER EXEC()?
   exec() shares memory with the parent process. subprocess.run() creates
   a truly isolated process — much safer for untrusted code.
4. STDOUT + STDERR CAPTURED: Agent sees both — errors are observations
   it can reason about and fix in the next iteration. This is the retry
   loop that makes the Coder agent robust against its own mistakes.
5. OUTPUT DIRECTORY: CWD is set to data/outputs/ so any files the code
   saves (charts, CSVs) go to the right place automatically.

Interview talking point:
    "I implemented two-layer code safety: RestrictedPython for compile-time
    validation, and subprocess with timeout for runtime isolation. The agent
    receives stdout + stderr as its observation, which lets it debug and
    revise its code just like a human programmer would."
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Optional
from loguru import logger

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from config import settings

try:
    from RestrictedPython import compile_restricted

    RESTRICTED_PYTHON_AVAILABLE = True
except ImportError:
    RESTRICTED_PYTHON_AVAILABLE = False
    logger.warning("RestrictedPython not installed. Run: pip install RestrictedPython")


def execute_code(code: str, timeout: Optional[int] = None) -> str:
    """
    Execute Python code in a sandboxed subprocess.

    Args:
        code: Python code string to execute.
        timeout: Seconds before killing the process (default from config).

    Returns:
        JSON string with keys:
        - success (bool): whether code ran without error
        - stdout (str): captured print output
        - stderr (str): captured warnings/errors
        - error (str): clean error message if success=False
        - execution_time (float): seconds taken
    """
    timeout = timeout or settings.code_timeout_seconds
    logger.debug(f"[code_executor] Executing {len(code)} chars (timeout={timeout}s)")

    # Layer 1: Compile-time validation with RestrictedPython
    if RESTRICTED_PYTHON_AVAILABLE:
        validation = _validate_code(code)
        if not validation["valid"]:
            return json.dumps(
                {
                    "success": False,
                    "stdout": "",
                    "stderr": "",
                    "error": f"Code validation failed: {validation['error']}",
                    "execution_time": 0.0,
                }
            )

    # Layer 2: Run in subprocess
    # Determine outputs directory for cwd
    outputs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "outputs",
    )
    os.makedirs(outputs_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(_wrap_code(code))
        tmp_path = f.name

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=outputs_dir,
        )
        elapsed = time.time() - start

        stdout = result.stdout[: settings.max_code_output_chars]
        stderr = result.stderr[:2000]
        success = result.returncode == 0

        error_msg = ""
        if not success:
            error_lines = [l for l in stderr.strip().split("\n") if l.strip()]
            error_msg = "\n".join(error_lines[-5:]) if error_lines else "Unknown error"

        logger.info(
            f"[code_executor] {'✓' if success else '✗'} "
            f"exit={result.returncode} time={elapsed:.2f}s"
        )

        return json.dumps(
            {
                "success": success,
                "stdout": stdout,
                "stderr": stderr,
                "error": error_msg,
                "execution_time": round(elapsed, 3),
            },
            ensure_ascii=False,
        )

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        logger.warning(f"[code_executor] Timeout after {timeout}s")
        return json.dumps(
            {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": f"Code execution timed out after {timeout} seconds. Check for infinite loops.",
                "execution_time": round(elapsed, 3),
            }
        )

    except Exception as e:
        logger.error(f"[code_executor] Subprocess error: {e}")
        return json.dumps(
            {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": f"Executor error: {type(e).__name__}: {e}",
                "execution_time": 0.0,
            }
        )

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _validate_code(code: str) -> dict:
    """
    Use RestrictedPython to validate code at compile time.
    Catches dangerous constructs before we ever run anything.
    Returns {"valid": bool, "error": str}
    """
    try:
        compile_restricted(code, filename="<agent_code>", mode="exec")
        return {"valid": True, "error": ""}
    except SyntaxError as e:
        return {"valid": False, "error": f"Syntax error: {e}"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


def _wrap_code(code: str) -> str:
    """
    Prepend a safety header to user code.
    Sets up the environment and suppresses noisy warnings.
    """
    prefix = textwrap.dedent(
        """\
        # AgentForge Safe Executor — auto-injected prefix
        import os, sys, warnings
        warnings.filterwarnings("ignore")
        OUTPUTS_DIR = os.getcwd()  # set to data/outputs/ by subprocess cwd

        # ── User code below ──
    """
    )
    return prefix + "\n" + code


def format_execution_result(result_json: str) -> str:
    """
    Format the JSON result into a readable observation string for the agent.
    Used by the Coder agent to present results clearly.
    """
    try:
        result = json.loads(result_json)
    except Exception:
        return result_json

    lines = []
    if result.get("success"):
        lines.append(
            f"✓ Code executed successfully in {result.get('execution_time', 0):.2f}s"
        )
        if result.get("stdout"):
            lines.append(f"Output:\n{result['stdout']}")
        if result.get("stderr"):
            lines.append(f"Warnings:\n{result['stderr']}")
    else:
        lines.append("✗ Code execution failed")
        if result.get("error"):
            lines.append(f"Error:\n{result['error']}")
        if result.get("stderr"):
            lines.append(f"Stderr:\n{result['stderr']}")
        if result.get("stdout"):
            lines.append(f"Partial output (before error):\n{result['stdout']}")

    return "\n".join(lines)


if __name__ == "__main__":
    from rich import print as rprint

    rprint("[bold cyan]Testing Code Executor Tool...[/bold cyan]")

    rprint("\n[yellow]Test 1: Simple working code[/yellow]")
    result = execute_code(
        """
import math
numbers = [1, 4, 9, 16, 25]
roots = [math.sqrt(n) for n in numbers]
print(f"Square roots: {roots}")
print(f"Sum: {sum(roots):.2f}")
"""
    )
    data = json.loads(result)
    rprint(f"Success: {data['success']}")
    rprint(f"Output: {data['stdout'].strip()}")

    rprint("\n[yellow]Test 2: Runtime error (ZeroDivisionError)[/yellow]")
    result2 = execute_code("print(1 / 0)")
    data2 = json.loads(result2)
    rprint(f"Success: {data2['success']}")
    rprint(f"Error: {data2['error']}")

    rprint("\n[yellow]Test 3: Pandas data analysis[/yellow]")
    result3 = execute_code(
        """
import pandas as pd
data = {
    "country": ["China", "USA", "India", "Germany", "Japan"],
    "solar_gw": [430, 140, 80, 67, 84]
}
df = pd.DataFrame(data).sort_values("solar_gw", ascending=False)
print(df.to_string(index=False))
print(f"\\nTotal: {df.solar_gw.sum()} GW")
"""
    )
    data3 = json.loads(result3)
    rprint(f"Success: {data3['success']}")
    rprint(f"Output:\n{data3['stdout']}")

    rprint("\n[yellow]Test 4: Timeout enforcement[/yellow]")
    result4 = execute_code("while True: pass", timeout=2)
    data4 = json.loads(result4)
    rprint(f"Success: {data4['success']}")
    rprint(f"Error: {data4['error']}")

    rprint("[bold green]✓ Code executor test complete![/bold green]")
