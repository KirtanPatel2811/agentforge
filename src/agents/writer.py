import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.base_agent import BaseAgent
from src.core.memory import ChunkType
from src.core.task_queue import get_queue, TaskStatus
from src.core.message_bus import get_bus, make_result_message
from src.tools.chart_generator import get_charts_for_task
from loguru import logger


class WriterAgent(BaseAgent):
    """
    Writer agent — assembles final report from all agent outputs in memory.

    Interview talking point:
        The Writer demonstrates the value of shared vector memory. Rather than
        receiving outputs through direct calls, it queries ChromaDB for all
        chunks tagged with the task_id, groups them by agent, and builds a
        rich context before writing. This means it works even if agents
        completed in different orders or at different times.
    """

    agent_name = "writer"

    system_prompt = (
        "You are an expert Technical Writer Agent in a multi-agent AI system.\n\n"
        "Your job is to synthesise research, data analysis, and code results into a polished report.\n\n"
        "Your FINAL_ANSWER must be a complete structured report with these sections:\n\n"
        "# [Report Title]\n\n"
        "## Executive Summary\n"
        "[2-3 sentence overview of key findings]\n\n"
        "## Key Findings\n"
        "[Bullet points with specific numbers]\n\n"
        "## Detailed Analysis\n"
        "[2-3 paragraphs referencing charts and statistics]\n\n"
        "## Code and Implementation\n"
        "[If code was produced: brief explanation]\n\n"
        "## Sources and References\n"
        "[All sources with URLs]\n\n"
        "## Conclusion\n"
        "[1 paragraph synthesising into actionable insight]\n\n"
        "Writing standards:\n"
        "- Use specific numbers (never vague language)\n"
        "- Reference charts: 'As shown in [Chart Title]...'\n"
        "- Keep sentences clear and professional\n"
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
                "You have received all outputs from the specialist agents.\n\n"
                "ORIGINAL GOAL:\n" + subtask.description + "\n\n"
                + context
                + "\n\nWrite the complete final report following your system prompt format."
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
            content = "\n\n".join(c["content"] for c in summary["researcher"])
            sections.append("RESEARCH FINDINGS (from Researcher Agent):\n" + "-" * 40 + "\n" + content)

        if "analyst" in summary:
            content = "\n\n".join(c["content"] for c in summary["analyst"])
            sections.append("DATA ANALYSIS (from Analyst Agent):\n" + "-" * 40 + "\n" + content)

        if "coder" in summary:
            content = "\n\n".join(c["content"] for c in summary["coder"])
            sections.append("CODE AND IMPLEMENTATION (from Coder Agent):\n" + "-" * 40 + "\n" + content)

        if charts:
            chart_list = "\n".join(
                f"- [{c['chart_type'].upper()}] {c['title']}: {c['description']}"
                for c in charts
            )
            sections.append("CHARTS CREATED:\n" + "-" * 40 + "\n" + chart_list)

        if not sections:
            chunks = self.memory.retrieve(
                query="research findings analysis results",
                task_id=task_id, k=10,
            )
            if chunks:
                content = "\n\n".join(
                    "[" + c["metadata"].get("agent_name", "unknown") + "]: " + c["content"]
                    for c in chunks
                )
                sections.append("AGENT OUTPUTS:\n" + "-" * 40 + "\n" + content)

        if not sections:
            return "NOTE: No agent outputs found in memory. Write best-effort report from goal alone."

        return "\n\n".join(sections)

    def _chunk_type(self):
        return ChunkType.REPORT_DRAFT
