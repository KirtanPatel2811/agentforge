import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.base_agent import BaseAgent, Tool
from src.core.memory import ChunkType
from src.tools.code_executor import execute_code
from src.tools.url_reader import read_url
from loguru import logger


class CoderAgent(BaseAgent):
    """
    Coder agent — writes, executes, and debugs Python code.

    Interview talking point:
        The Coder has an error recovery loop built into its ReAct cycle.
        When code fails, the error becomes an OBSERVATION the agent reasons
        about. It then rewrites and retries — exactly like a human developer
        debugging. The sandbox timeout prevents infinite loops.
    """

    agent_name = "coder"

    system_prompt = (
        "You are an expert Python Coder Agent in a multi-agent AI system.\n\n"
        "Your job is to write correct, clean Python code and ALWAYS execute it to verify it works.\n\n"
        "Available packages: pandas, numpy, scipy, plotly, math, statistics, json, re, datetime, collections\n\n"
        "CRITICAL RULES:\n"
        "1. ALWAYS execute your code with execute_code before giving FINAL_ANSWER\n"
        "2. If code fails, READ the error, fix it, and execute again\n"
        "3. Never return code you have not successfully run\n"
        "4. Maximum 3 attempts to fix errors\n\n"
        "FINAL_ANSWER format:\n"
        "CODE:\n[the final working code]\n\n"
        "OUTPUT:\n[what the code printed]\n\n"
        "EXPLANATION:\n[brief explanation of what the code does]\n\n"
        "Code quality standards:\n"
        "- Use descriptive variable names\n"
        "- Add print() for every key result\n"
        "- Handle edge cases\n"
        "- Save files to current directory\n"
        "- For charts: fig.write_html('chart_name.html')"
    )

    def __init__(self):
        self.tools = {
            "execute_code": Tool(
                name="execute_code",
                description=(
                    "Execute Python code in a safe sandbox. "
                    "Returns JSON with success, stdout, stderr, error. "
                    "Available: pandas, numpy, scipy, plotly, math, statistics."
                ),
                func=execute_code,
                example='ACTION_INPUT: {"code": "import pandas as pd\nprint(pd.__version__)"}',
            ),
            "read_url": Tool(
                name="read_url",
                description="Read documentation if you need to look something up.",
                func=read_url,
                example='ACTION_INPUT: {"url": "https://pandas.pydata.org/docs/"}',
            ),
        }
        super().__init__()

    def _chunk_type(self):
        return ChunkType.CODE

    def _post_process(self, result, subtask):
        if "CODE:" not in result:
            return "CODE RESULT\n" + "=" * 40 + "\n" + result
        return result
