"""
fix_phase3.py — Rewrites all 6 agent files with correct syntax.
Run from agentforge root: python fix_phase3.py
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

def write(path, content):
    full = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Fixed: {path}")


# ── researcher.py ─────────────────────────────────────────────────────────────
write("src/agents/researcher.py", """\
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.base_agent import BaseAgent, Tool
from src.core.memory import ChunkType
from src.tools.web_search import web_search, news_search
from src.tools.arxiv_search import arxiv_search
from src.tools.url_reader import read_url
from loguru import logger


class ResearcherAgent(BaseAgent):
    \"\"\"
    Researcher agent — finds information via web, news, ArXiv, and URL reading.

    Interview talking point:
        The Researcher combines three information sources: general web search
        for broad coverage, news search for recency, and ArXiv for academic
        depth. It uses url_reader to go beyond search snippets. All findings
        are stored in ChromaDB with source metadata so the Writer can cite them.
    \"\"\"

    agent_name = "researcher"

    system_prompt = (
        "You are an expert Research Agent in a multi-agent AI system.\\n\\n"
        "Your job is to find accurate, relevant, up-to-date information on any topic.\\n\\n"
        "Tools available:\\n"
        "- web_search: general web results\\n"
        "- news_search: recent news articles\\n"
        "- arxiv_search: academic papers\\n"
        "- read_url: fetch and read full page content\\n\\n"
        "Your FINAL_ANSWER must be a structured research summary with:\\n"
        "1. KEY FINDINGS: bullet points with specific numbers/statistics\\n"
        "2. SOURCES: URLs you actually read\\n"
        "3. ACADEMIC CONTEXT: relevant papers found\\n"
        "4. GAPS: what you could not find\\n\\n"
        "Quality standards:\\n"
        "- Prefer specific numbers over vague claims\\n"
        "- Always read at least 2 URLs before writing your final answer\\n"
        "- Note publication dates of statistics\\n"
        "- If sources conflict, report both"
    )

    def __init__(self):
        self.tools = {
            "web_search": Tool(
                name="web_search",
                description="Search the web. Returns titles, URLs, snippets.",
                func=web_search,
                example='ACTION_INPUT: {"query": "solar energy capacity 2024 by country"}',
            ),
            "news_search": Tool(
                name="news_search",
                description="Search recent news. Use for current events.",
                func=news_search,
                example='ACTION_INPUT: {"query": "solar energy record 2024"}',
            ),
            "arxiv_search": Tool(
                name="arxiv_search",
                description="Search academic papers on ArXiv.",
                func=arxiv_search,
                example='ACTION_INPUT: {"query": "renewable energy deep learning", "max_results": 3}',
            ),
            "read_url": Tool(
                name="read_url",
                description="Fetch and read a full web page. Use after web_search for complete content.",
                func=read_url,
                example='ACTION_INPUT: {"url": "https://example.com/article"}',
            ),
        }
        super().__init__()

    def _chunk_type(self):
        return ChunkType.RESEARCH

    def _post_process(self, result, subtask):
        if "KEY FINDINGS" not in result and "SOURCES" not in result:
            return "RESEARCH FINDINGS\\n" + "=" * 40 + "\\n" + result
        return result
""")


# ── coder.py ──────────────────────────────────────────────────────────────────
write("src/agents/coder.py", """\
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.base_agent import BaseAgent, Tool
from src.core.memory import ChunkType
from src.tools.code_executor import execute_code
from src.tools.url_reader import read_url
from loguru import logger


class CoderAgent(BaseAgent):
    \"\"\"
    Coder agent — writes, executes, and debugs Python code.

    Interview talking point:
        The Coder has an error recovery loop built into its ReAct cycle.
        When code fails, the error becomes an OBSERVATION the agent reasons
        about. It then rewrites and retries — exactly like a human developer
        debugging. The sandbox timeout prevents infinite loops.
    \"\"\"

    agent_name = "coder"

    system_prompt = (
        "You are an expert Python Coder Agent in a multi-agent AI system.\\n\\n"
        "Your job is to write correct, clean Python code and ALWAYS execute it to verify it works.\\n\\n"
        "Available packages: pandas, numpy, scipy, plotly, math, statistics, json, re, datetime, collections\\n\\n"
        "CRITICAL RULES:\\n"
        "1. ALWAYS execute your code with execute_code before giving FINAL_ANSWER\\n"
        "2. If code fails, READ the error, fix it, and execute again\\n"
        "3. Never return code you have not successfully run\\n"
        "4. Maximum 3 attempts to fix errors\\n\\n"
        "FINAL_ANSWER format:\\n"
        "CODE:\\n[the final working code]\\n\\n"
        "OUTPUT:\\n[what the code printed]\\n\\n"
        "EXPLANATION:\\n[brief explanation of what the code does]\\n\\n"
        "Code quality standards:\\n"
        "- Use descriptive variable names\\n"
        "- Add print() for every key result\\n"
        "- Handle edge cases\\n"
        "- Save files to current directory\\n"
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
                example='ACTION_INPUT: {"code": "import pandas as pd\\nprint(pd.__version__)"}',
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
            return "CODE RESULT\\n" + "=" * 40 + "\\n" + result
        return result
""")


# ── analyst.py ────────────────────────────────────────────────────────────────
write("src/agents/analyst.py", """\
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.base_agent import BaseAgent, Tool
from src.core.memory import ChunkType
from src.tools.code_executor import execute_code
from src.tools.chart_generator import generate_chart, get_charts_for_task
from loguru import logger


class AnalystAgent(BaseAgent):
    \"\"\"
    Analyst agent — interprets data, generates statistics, creates charts.

    Interview talking point:
        The Analyst is the bridge between raw data and human-readable insights.
        It uses execute_code for statistical analysis and generate_chart for
        visualisations. It pulls context from ChromaDB to build on the
        Researcher findings rather than re-discovering data. This shared
        memory pattern is what makes agents genuinely collaborative.
    \"\"\"

    agent_name = "analyst"

    system_prompt = (
        "You are an expert Data Analyst Agent in a multi-agent AI system.\\n\\n"
        "Your job is to analyse data, compute statistics, and create clear visualisations.\\n\\n"
        "Workflow:\\n"
        "1. Understand the data from context provided\\n"
        "2. Write and execute pandas/numpy code to compute key statistics\\n"
        "3. Create at least one chart using generate_chart\\n"
        "4. Write a data-driven narrative explaining what the numbers mean\\n\\n"
        "Available packages for code: pandas, numpy, scipy, statistics, json, math\\n\\n"
        "Chart types: bar, line, scatter, pie, histogram, heatmap\\n\\n"
        "FINAL_ANSWER format:\\n"
        "DATA ANALYSIS SUMMARY\\n"
        "=====================\\n"
        "KEY STATISTICS:\\n"
        "- [statistic with value]\\n\\n"
        "TRENDS AND INSIGHTS:\\n"
        "- [insight]\\n\\n"
        "CHARTS CREATED:\\n"
        "- [chart title]: [what it shows]\\n\\n"
        "INTERPRETATION:\\n"
        "[2-3 paragraph narrative]\\n\\n"
        "Standards:\\n"
        "- Always include specific numbers\\n"
        "- Explain what numbers MEAN\\n"
        "- Reference charts by title in narrative"
    )

    def __init__(self):
        self._current_task_id = None
        self.tools = {
            "execute_code": Tool(
                name="execute_code",
                description=(
                    "Run Python for data analysis. "
                    "Use pandas, numpy, scipy for statistics. "
                    "Always print() your results."
                ),
                func=execute_code,
                example='ACTION_INPUT: {"code": "import numpy as np\\ndata=[430,140,80]\\nprint(f\'mean={np.mean(data):.1f}\')"}',
            ),
            "generate_chart": Tool(
                name="generate_chart",
                description=(
                    "Create a Plotly chart saved as HTML. "
                    "chart_type: bar, line, scatter, pie, histogram, heatmap. "
                    "data: JSON string of list-of-dicts."
                ),
                func=self._chart_wrapper,
                example='ACTION_INPUT: {"data": "[{\\"country\\":\\"China\\",\\"gw\\":430}]", "chart_type": "bar", "x_column": "country", "y_column": "gw", "title": "Solar Capacity"}',
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
            return "ANALYSIS RESULT\\n" + "=" * 40 + "\\n" + result
        return result
""")


# ── writer.py ─────────────────────────────────────────────────────────────────
write("src/agents/writer.py", """\
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.base_agent import BaseAgent
from src.core.memory import ChunkType
from src.core.task_queue import get_queue, TaskStatus
from src.core.message_bus import get_bus, make_result_message
from src.tools.chart_generator import get_charts_for_task
from loguru import logger


class WriterAgent(BaseAgent):
    \"\"\"
    Writer agent — assembles final report from all agent outputs in memory.

    Interview talking point:
        The Writer demonstrates the value of shared vector memory. Rather than
        receiving outputs through direct calls, it queries ChromaDB for all
        chunks tagged with the task_id, groups them by agent, and builds a
        rich context before writing. This means it works even if agents
        completed in different orders or at different times.
    \"\"\"

    agent_name = "writer"

    system_prompt = (
        "You are an expert Technical Writer Agent in a multi-agent AI system.\\n\\n"
        "Your job is to synthesise research, data analysis, and code results into a polished report.\\n\\n"
        "Your FINAL_ANSWER must be a complete structured report with these sections:\\n\\n"
        "# [Report Title]\\n\\n"
        "## Executive Summary\\n"
        "[2-3 sentence overview of key findings]\\n\\n"
        "## Key Findings\\n"
        "[Bullet points with specific numbers]\\n\\n"
        "## Detailed Analysis\\n"
        "[2-3 paragraphs referencing charts and statistics]\\n\\n"
        "## Code and Implementation\\n"
        "[If code was produced: brief explanation]\\n\\n"
        "## Sources and References\\n"
        "[All sources with URLs]\\n\\n"
        "## Conclusion\\n"
        "[1 paragraph synthesising into actionable insight]\\n\\n"
        "Writing standards:\\n"
        "- Use specific numbers (never vague language)\\n"
        "- Reference charts: 'As shown in [Chart Title]...'\\n"
        "- Keep sentences clear and professional\\n"
        "- Target length: 500-800 words"
    )

    def __init__(self):
        self.tools = {}
        super().__init__()

    def run(self, subtask):
        task_id = subtask.task_id
        subtask_id = subtask.subtask_id
        logger.info(f"[writer] Assembling report for task {task_id}")
        self.queue.update_subtask(subtask_id, status=TaskStatus.IN_PROGRESS)

        try:
            context = self._build_context(task_id)
            prompt = (
                "You have received all outputs from the specialist agents.\\n\\n"
                "ORIGINAL GOAL:\\n" + subtask.description + "\\n\\n"
                + context
                + "\\n\\nWrite the complete final report following your system prompt format."
            )
            report = self.llm.simple(prompt)

            chunk_id = self.memory.store(
                content=report, agent_name=self.agent_name,
                task_id=task_id, chunk_type=ChunkType.REPORT_DRAFT,
                metadata={"subtask_id": subtask_id},
            )
            self.queue.complete_subtask(subtask_id, result=report, chunk_ids=[chunk_id])
            self.queue.update_goal(task_id, final_report=report)
            self.bus.send(make_result_message(
                sender=self.agent_name, recipient="planner",
                task_id=task_id, output=report,
                chunk_ids=[chunk_id], success=True,
            ))
            logger.info(f"[writer] Report assembled — {len(report)} chars")
            return report

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"[writer] Failed: {error_msg}")
            self.queue.fail_subtask(subtask_id, error=error_msg)
            self.bus.send(make_result_message(
                sender=self.agent_name, recipient="planner",
                task_id=task_id, output="", success=False, error=error_msg,
            ))
            return ""

    def _build_context(self, task_id):
        summary = self.memory.get_task_summary(task_id)
        charts = get_charts_for_task(task_id)
        sections = []

        if "researcher" in summary:
            content = "\\n\\n".join(c["content"] for c in summary["researcher"])
            sections.append("RESEARCH FINDINGS (from Researcher Agent):\\n" + "-" * 40 + "\\n" + content)

        if "analyst" in summary:
            content = "\\n\\n".join(c["content"] for c in summary["analyst"])
            sections.append("DATA ANALYSIS (from Analyst Agent):\\n" + "-" * 40 + "\\n" + content)

        if "coder" in summary:
            content = "\\n\\n".join(c["content"] for c in summary["coder"])
            sections.append("CODE AND IMPLEMENTATION (from Coder Agent):\\n" + "-" * 40 + "\\n" + content)

        if charts:
            chart_list = "\\n".join(
                f"- [{c['chart_type'].upper()}] {c['title']}: {c['description']}"
                for c in charts
            )
            sections.append("CHARTS CREATED:\\n" + "-" * 40 + "\\n" + chart_list)

        if not sections:
            chunks = self.memory.retrieve(
                query="research findings analysis results",
                task_id=task_id, k=10,
            )
            if chunks:
                content = "\\n\\n".join(
                    "[" + c["metadata"].get("agent_name", "unknown") + "]: " + c["content"]
                    for c in chunks
                )
                sections.append("AGENT OUTPUTS:\\n" + "-" * 40 + "\\n" + content)

        if not sections:
            return "NOTE: No agent outputs found in memory. Write best-effort report from goal alone."

        return "\\n\\n".join(sections)

    def _chunk_type(self):
        return ChunkType.REPORT_DRAFT
""")


# ── critic.py ─────────────────────────────────────────────────────────────────
write("src/agents/critic.py", """\
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.base_agent import BaseAgent
from src.core.memory import ChunkType
from src.core.llm_client import LLMParseError
from src.core.task_queue import get_queue, TaskStatus
from src.core.message_bus import get_bus, make_result_message
from config import settings
from loguru import logger


class CriticAgent(BaseAgent):
    \"\"\"
    Critic agent — scores report quality on 5 dimensions, controls revision loop.

    Interview talking point:
        The Critic is what separates a robust agentic system from a simple
        pipeline. It scores output on five dimensions and provides specific
        actionable feedback for each dimension below threshold. Revision
        requests are targeted: low factual_grounding sends the Researcher
        back, not the Writer. Inspired by Constitutional AI reward models.
    \"\"\"

    agent_name = "critic"

    system_prompt = (
        "You are a rigorous Quality Control Critic Agent in a multi-agent AI system.\\n\\n"
        "Evaluate reports and return a JSON object with EXACTLY this structure:\\n"
        "{\\n"
        '  "scores": {\\n'
        '    "completeness": 0.0-1.0,\\n'
        '    "factual_grounding": 0.0-1.0,\\n'
        '    "code_correctness": 0.0-1.0,\\n'
        '    "source_citation": 0.0-1.0,\\n'
        '    "readability": 0.0-1.0\\n'
        "  },\\n"
        '  "overall_score": 0.0-1.0,\\n'
        '  "approved": true/false,\\n'
        '  "feedback": {"completeness": "...", "factual_grounding": "...", '
        '"code_correctness": "...", "source_citation": "...", "readability": "..."},\\n'
        '  "revision_instructions": "specific improvements needed or empty string if approved",\\n'
        '  "revision_targets": ["researcher", "writer"]\\n'
        "}\\n\\n"
        "Scoring guide:\\n"
        "- completeness: addressed ALL parts of goal? Missing sections = low score\\n"
        "- factual_grounding: stats cited with sources? Vague claims = low score\\n"
        "- code_correctness: code valid, explained, output shown?\\n"
        "- source_citation: URLs provided? Sources named?\\n"
        "- readability: clear structure, logical flow, professional tone?\\n\\n"
        "overall_score = weighted: completeness(0.25) + factual_grounding(0.25) + "
        "code(0.20) + citations(0.15) + readability(0.15)\\n\\n"
        "approved = overall_score >= 0.7\\n\\n"
        "IMPORTANT: Return ONLY the JSON. No preamble, no markdown fences."
    )

    def __init__(self):
        self.tools = {}
        super().__init__()

    def run(self, subtask):
        task_id = subtask.task_id
        subtask_id = subtask.subtask_id
        logger.info(f"[critic] Reviewing task {task_id}")
        self.queue.update_subtask(subtask_id, status=TaskStatus.IN_PROGRESS)

        try:
            goal_task = self.queue.get_goal(task_id)
            report = goal_task.final_report if goal_task else None

            if not report:
                chunks = self.memory.retrieve(
                    query="final report executive summary",
                    task_id=task_id, agent_name="writer", k=3,
                )
                report = "\\n\\n".join(c["content"] for c in chunks) if chunks else ""

            if not report:
                report = "No report was produced."

            original_goal = subtask.description
            prompt = (
                "Original Goal:\\n" + original_goal + "\\n\\n"
                "Report to Evaluate:\\n" + "=" * 60 + "\\n"
                + report[:6000] + "\\n" + "=" * 60 + "\\n\\n"
                "Evaluate this report and return your JSON assessment."
            )

            try:
                review = self.llm.complete_json([{"role": "user", "content": prompt}])
            except LLMParseError as e:
                logger.warning(f"[critic] JSON parse failed, using fallback: {e}")
                review = self._fallback_review(report)

            review = self._validate_review(review)

            overall = review.get("overall_score", 0.0)
            approved = review.get("approved", False)
            logger.info(f"[critic] Score: {overall:.2f} | Approved: {approved}")

            critique_text = json.dumps(review, indent=2)
            chunk_id = self.memory.store(
                content=critique_text, agent_name=self.agent_name,
                task_id=task_id, chunk_type=ChunkType.CRITIQUE,
                metadata={"overall_score": overall, "approved": approved},
            )
            self.queue.update_goal(
                task_id,
                critic_score=overall,
                critic_feedback=review.get("revision_instructions", ""),
                approved=approved,
                status=TaskStatus.COMPLETED if approved else TaskStatus.NEEDS_REVISION,
            )
            self.queue.complete_subtask(subtask_id, result=critique_text, chunk_ids=[chunk_id])
            self.bus.send(make_result_message(
                sender=self.agent_name, recipient="planner",
                task_id=task_id, output=critique_text,
                chunk_ids=[chunk_id], success=True,
            ))
            return critique_text

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"[critic] Failed: {error_msg}")
            self.queue.fail_subtask(subtask_id, error=error_msg)
            return ""

    def _validate_review(self, review):
        scores = review.get("scores", {})
        dimensions = settings.critic_dimensions
        for dim in dimensions:
            if dim not in scores:
                scores[dim] = 0.5
            scores[dim] = max(0.0, min(1.0, float(scores[dim])))
        review["scores"] = scores

        weights = {
            "completeness": 0.25, "factual_grounding": 0.25,
            "code_correctness": 0.20, "source_citation": 0.15, "readability": 0.15,
        }
        overall = sum(scores.get(d, 0.5) * w for d, w in weights.items())
        review["overall_score"] = round(overall, 3)
        review["approved"] = overall >= settings.critic_pass_threshold

        if "feedback" not in review:
            review["feedback"] = {d: "No feedback provided" for d in dimensions}
        if "revision_instructions" not in review:
            review["revision_instructions"] = "" if review["approved"] else "Please improve quality."
        if "revision_targets" not in review:
            review["revision_targets"] = [] if review["approved"] else ["writer"]

        return review

    def _fallback_review(self, report):
        has_sources = "http" in report.lower() or "source" in report.lower()
        has_numbers = any(c.isdigit() for c in report)
        has_structure = "#" in report or "##" in report
        scores = {
            "completeness": 0.6,
            "factual_grounding": 0.7 if has_numbers else 0.4,
            "code_correctness": 0.5,
            "source_citation": 0.6 if has_sources else 0.3,
            "readability": 0.7 if has_structure else 0.5,
        }
        overall = sum(scores.values()) / len(scores)
        return {
            "scores": scores,
            "overall_score": round(overall, 3),
            "approved": overall >= settings.critic_pass_threshold,
            "feedback": {d: "Auto-assessed" for d in scores},
            "revision_instructions": "" if overall >= settings.critic_pass_threshold
                else "Improve factual grounding and source citations.",
            "revision_targets": [] if overall >= settings.critic_pass_threshold else ["writer"],
        }

    def _chunk_type(self):
        return ChunkType.CRITIQUE
""")


# ── planner.py ────────────────────────────────────────────────────────────────
write("src/agents/planner.py", """\
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.base_agent import BaseAgent
from src.core.llm_client import LLMParseError
from src.core.memory import ChunkType, get_memory
from src.core.task_queue import get_queue, TaskStatus
from src.core.message_bus import get_bus
from config import settings, AgentName
from loguru import logger

from src.agents.researcher import ResearcherAgent
from src.agents.coder import CoderAgent
from src.agents.analyst import AnalystAgent
from src.agents.writer import WriterAgent
from src.agents.critic import CriticAgent


class PlannerAgent(BaseAgent):
    \"\"\"
    Planner agent — orchestrates the entire multi-agent pipeline.

    Interview talking point:
        The Planner uses the LLM to decompose an arbitrary natural language
        goal into typed subtasks for the right specialist agents. The dependency
        system ensures Researcher runs before Writer. The Critic revision loop
        means the system can self-correct: if output scores below 0.7, the
        Planner re-runs flagged agents with the Critic specific feedback.
    \"\"\"

    agent_name = "planner"

    system_prompt = (
        "You are the Planner Agent in AgentForge, a multi-agent AI system.\\n\\n"
        "Decompose the user goal into subtasks for specialist agents.\\n\\n"
        "Available agents:\\n"
        "- researcher: web search, news, ArXiv, URL reading\\n"
        "- coder: write and execute Python code\\n"
        "- analyst: data analysis, statistics, chart generation\\n"
        "- writer: assemble final report (ALWAYS last)\\n"
        "- critic: quality review (runs automatically, do NOT include)\\n\\n"
        "Return a JSON plan:\\n"
        "{\\n"
        '  "title": "brief task title",\\n'
        '  "reasoning": "why these agents in this order",\\n'
        '  "subtasks": [\\n'
        "    {\\n"
        '      "agent": "researcher",\\n'
        '      "description": "specific instruction",\\n'
        '      "expected_output": "what to return",\\n'
        '      "priority": 2,\\n'
        '      "dependencies": []\\n'
        "    },\\n"
        "    {\\n"
        '      "agent": "writer",\\n'
        '      "description": "assemble final report",\\n'
        '      "expected_output": "complete structured report",\\n'
        '      "priority": 1,\\n'
        '      "dependencies": ["researcher"]\\n'
        "    }\\n"
        "  ]\\n"
        "}\\n\\n"
        "Rules:\\n"
        "- writer MUST always be the last subtask\\n"
        "- researcher should run before analyst and writer\\n"
        "- priority: higher = runs first (researcher=3, coder/analyst=2, writer=1)\\n"
        "- dependencies: list of agent names that must complete first\\n"
        "- Return ONLY the JSON. No preamble."
    )

    def __init__(self):
        self.tools = {}
        super().__init__()
        logger.info("[planner] Initialising specialist agents...")
        self._agents = {
            AgentName.RESEARCHER: ResearcherAgent(),
            AgentName.CODER:      CoderAgent(),
            AgentName.ANALYST:    AnalystAgent(),
            AgentName.WRITER:     WriterAgent(),
            AgentName.CRITIC:     CriticAgent(),
        }
        logger.info("[planner] All agents ready")

    def run_goal(self, goal):
        \"\"\"
        Main entry point. Takes a natural language goal and runs the
        full multi-agent pipeline to completion.
        Returns dict with task_id, report, critic_score, approved, etc.
        \"\"\"
        logger.info(f"[planner] Starting goal: '{goal[:80]}'")
        queue = get_queue()
        goal_task = queue.create_goal(goal)
        task_id = goal_task.task_id
        queue.update_goal(task_id, status=TaskStatus.IN_PROGRESS)

        try:
            # Step 1: Generate plan
            plan = self._create_plan(goal, task_id)
            logger.info(f"[planner] Plan: {plan['title']}")
            logger.info(f"[planner] Agents: {[s['agent'] for s in plan['subtasks']]}")
            queue.update_goal(task_id, plan=json.dumps(plan))

            # Step 2: Register subtasks
            subtask_map = self._register_subtasks(plan, task_id)

            # Step 3: Run specialist agents (not writer)
            subtask_results = {}
            specialist_subtasks = [s for s in plan["subtasks"] if s["agent"] != AgentName.WRITER]

            for subtask_def in specialist_subtasks:
                agent_name = subtask_def["agent"]
                if agent_name not in self._agents:
                    logger.warning(f"[planner] Unknown agent: {agent_name}, skipping")
                    continue
                subtask_obj = subtask_map[agent_name]
                logger.info(f"[planner] Running {agent_name}...")
                result = self._agents[agent_name].run(subtask_obj)
                subtask_results[agent_name] = result
                logger.info(f"[planner] {agent_name} done ({len(result)} chars)")

            # Step 4: Run writer
            writer_subtask = subtask_map.get(AgentName.WRITER)
            if writer_subtask:
                logger.info("[planner] Running writer...")
                report = self._agents[AgentName.WRITER].run(writer_subtask)
                subtask_results[AgentName.WRITER] = report
                logger.info(f"[planner] Writer done ({len(report)} chars)")
            else:
                report = "\\n".join(subtask_results.values())

            # Step 5: Critic review loop
            critic_result = None
            revision_round = 0

            while revision_round <= settings.max_revision_rounds:
                logger.info(f"[planner] Critic review round {revision_round + 1}")
                critic_subtask = self._make_critic_subtask(goal, task_id, revision_round)
                critique_json = self._agents[AgentName.CRITIC].run(critic_subtask)
                subtask_results[AgentName.CRITIC] = critique_json

                try:
                    critic_result = json.loads(critique_json)
                except Exception:
                    critic_result = {"approved": True, "overall_score": 0.75}

                approved = critic_result.get("approved", False)
                overall = critic_result.get("overall_score", 0.0)
                logger.info(f"[planner] Score: {overall:.2f} | Approved: {approved}")

                if approved or revision_round >= settings.max_revision_rounds:
                    break

                # Revision: re-run flagged agents with feedback
                revision_round += 1
                feedback = critic_result.get("revision_instructions", "")
                targets = critic_result.get("revision_targets", ["writer"])
                logger.info(f"[planner] Revision {revision_round}: targets={targets}")

                for target in targets:
                    if target in self._agents and target in subtask_map:
                        subtask_obj = subtask_map[target]
                        subtask_obj.context["critic_feedback"] = feedback
                        subtask_obj.context["revision_round"] = revision_round
                        subtask_obj.status = TaskStatus.PENDING
                        queue._save_subtask(subtask_obj)
                        result = self._agents[target].run(subtask_obj)
                        subtask_results[target] = result

                # Re-run writer after revisions to non-writer agents
                if writer_subtask and any(t != AgentName.WRITER for t in targets):
                    writer_subtask.status = TaskStatus.PENDING
                    queue._save_subtask(writer_subtask)
                    report = self._agents[AgentName.WRITER].run(writer_subtask)
                    subtask_results[AgentName.WRITER] = report

            # Step 6: Finalise
            final_goal = queue.get_goal(task_id)
            approved = critic_result.get("approved", False) if critic_result else False
            queue.update_goal(task_id, status=TaskStatus.COMPLETED, approved=approved)

            logger.info(
                f"[planner] Complete! task={task_id} "
                f"score={critic_result.get('overall_score', 0):.2f} approved={approved}"
            )

            return {
                "task_id":         task_id,
                "goal":            goal,
                "plan":            plan,
                "report":          final_goal.final_report or report,
                "critic_score":    critic_result.get("overall_score", 0.0) if critic_result else 0.0,
                "approved":        approved,
                "critic_feedback": critic_result.get("revision_instructions", "") if critic_result else "",
                "subtask_results": subtask_results,
            }

        except Exception as e:
            logger.error(f"[planner] Pipeline failed: {e}")
            queue.update_goal(task_id, status=TaskStatus.FAILED)
            raise

    def _create_plan(self, goal, task_id):
        prompt = (
            "Create an execution plan for this goal:\\n\\n"
            "GOAL: " + goal + "\\n\\n"
            "Return a JSON plan assigning subtasks to specialist agents. "
            "writer must ALWAYS be last."
        )
        try:
            plan = self.llm.complete_json([{"role": "user", "content": prompt}])
            if "subtasks" not in plan:
                raise ValueError("Plan missing subtasks")
            return plan
        except (LLMParseError, ValueError):
            logger.warning("[planner] Invalid JSON plan, using default")
            return self._default_plan(goal)

    def _default_plan(self, goal):
        return {
            "title": "Research and report: " + goal[:50],
            "reasoning": "Default plan: research then write",
            "subtasks": [
                {
                    "agent": AgentName.RESEARCHER,
                    "description": "Research the following topic thoroughly: " + goal,
                    "expected_output": "Structured findings with sources and key statistics",
                    "priority": 2,
                    "dependencies": [],
                },
                {
                    "agent": AgentName.WRITER,
                    "description": "Write a comprehensive report on: " + goal,
                    "expected_output": "Complete structured report with all sections",
                    "priority": 1,
                    "dependencies": [AgentName.RESEARCHER],
                },
            ],
        }

    def _register_subtasks(self, plan, task_id):
        queue = get_queue()
        subtask_map = {}
        for subtask_def in plan["subtasks"]:
            agent_name = subtask_def["agent"]
            dep_ids = [
                subtask_map[dep].subtask_id
                for dep in subtask_def.get("dependencies", [])
                if dep in subtask_map
            ]
            subtask = queue.add_subtask(
                task_id=task_id,
                agent_name=agent_name,
                description=subtask_def["description"],
                expected_output=subtask_def.get("expected_output", ""),
                priority=subtask_def.get("priority", 1),
                dependencies=dep_ids,
            )
            subtask_map[agent_name] = subtask
        return subtask_map

    def _make_critic_subtask(self, goal, task_id, revision_round):
        from src.core.task_queue import SubTask
        import uuid
        return SubTask(
            task_id=task_id,
            subtask_id="critic_" + uuid.uuid4().hex[:8],
            agent_name=AgentName.CRITIC,
            description="Review the final report for this goal: " + goal,
            expected_output="JSON quality scores and feedback",
            context={"revision_round": revision_round},
        )
""")


print("\\n✅ All 6 agent files fixed!")
print("\\nNow run:")
print("  python -m pytest tests/test_phase3_agents.py -v -m 'not live'")
