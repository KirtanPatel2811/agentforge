import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.base_agent import BaseAgent, Tool
from src.core.memory import ChunkType
from src.tools.code_executor import execute_code
from src.tools.chart_generator import generate_chart, get_charts_for_task
from loguru import logger


class AnalystAgent(BaseAgent):
    """
    Analyst agent — interprets data, generates statistics, creates charts.

    Improvement: execute_code tool description now explicitly states
    that ACTION_INPUT must contain a 'code' key as a string, preventing
    the LLM from sending empty {} inputs.
    """

    agent_name = "analyst"

    system_prompt = (
        "You are an expert Data Analyst Agent in a multi-agent AI system.\n\n"
        "Your job is to analyse data, compute statistics, and create clear visualisations.\n\n"
        "Workflow:\n"
        "1. Understand the data from context provided\n"
        "2. Write and execute pandas/numpy code to compute key statistics\n"
        "3. Create at least one chart using generate_chart\n"
        "4. Write a data-driven narrative explaining what the numbers mean\n\n"
        "Available packages for code: pandas, numpy, scipy, statistics, json, math\n\n"
        "Chart types available: bar, line, scatter, pie, histogram, heatmap\n\n"
        "CRITICAL: When using execute_code, ACTION_INPUT must ALWAYS contain the key 'code'\n"
        'Example: ACTION_INPUT: {"code": "import pandas as pd\nprint(42)"}\n\n'
        "FINAL_ANSWER format:\n"
        "DATA ANALYSIS SUMMARY\n"
        "=====================\n"
        "KEY STATISTICS:\n"
        "- [statistic with value]\n\n"
        "TRENDS AND INSIGHTS:\n"
        "- [insight]\n\n"
        "CHARTS CREATED:\n"
        "- [chart title]: [what it shows]\n\n"
        "INTERPRETATION:\n"
        "[2-3 paragraph narrative]\n\n"
        "Standards:\n"
        "- Always include specific numbers\n"
        "- Explain what numbers MEAN not just what they are\n"
        "- Reference charts by title in narrative"
    )

    def __init__(self):
        self._current_task_id = None
        self.tools = {
            "execute_code": Tool(
                name="execute_code",
                description=(
                    "Run Python for data analysis. "
                    "REQUIRED: ACTION_INPUT must have key 'code' as a string. "
                    "Use pandas, numpy, scipy for statistics. "
                    "Always print() your results so they appear in output. "
                    "Available: pandas, numpy, scipy, math, statistics, json."
                ),
                func=execute_code,
                example='ACTION_INPUT: {"code": "import numpy as np\ndata=[430,140,80]\nprint(f'mean={np.mean(data):.1f}')"}',
            ),
            "generate_chart": Tool(
                name="generate_chart",
                description=(
                    "Create a Plotly chart saved as HTML. "
                    "REQUIRED fields: data (JSON string), chart_type, x_column, y_column, title. "
                    "chart_type options: bar, line, scatter, pie, histogram, heatmap. "
                    "data format: JSON string of list-of-dicts like "
                    '[{"country":"China","gw":430},{"country":"USA","gw":140}]'
                ),
                func=self._chart_wrapper,
                example='ACTION_INPUT: {"data": "[{\"country\":\"China\",\"gw\":430}]", "chart_type": "bar", "x_column": "country", "y_column": "gw", "title": "Solar Capacity by Country"}',
            ),
        }
        super().__init__()

    def _chart_wrapper(self, data, chart_type, x_column, y_column, title,
                       x_label=None, y_label=None, color_column=None, description=None):
        return generate_chart(
            data=data, chart_type=chart_type,
            x_column=x_column, y_column=y_column,
            title=title, task_id=self._current_task_id or "unknown",
            x_label=x_label, y_label=y_label,
            color_column=color_column, description=description,
        )

    def run(self, subtask):
        self._current_task_id = subtask.task_id
        return super().run(subtask)

    def _chunk_type(self):
        return ChunkType.ANALYSIS

    def _post_process(self, result, subtask):
        if "DATA ANALYSIS" not in result and "KEY STATISTICS" not in result:
            return "ANALYSIS RESULT\n" + "=" * 40 + "\n" + result
        return result
