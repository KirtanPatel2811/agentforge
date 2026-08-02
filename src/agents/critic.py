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
    """
    Critic agent — scores report quality on 5 dimensions, controls revision loop.

    Interview talking point:
        The Critic is what separates a robust agentic system from a simple
        pipeline. It scores output on five dimensions and provides specific
        actionable feedback for each dimension below threshold. Revision
        requests are targeted: low factual_grounding sends the Researcher
        back, not the Writer. Inspired by Constitutional AI reward models.
    """

    agent_name = "critic"

    system_prompt = (
        "You are a rigorous Quality Control Critic Agent in a multi-agent AI system.\n\n"
        "Evaluate reports and return a JSON object with EXACTLY this structure:\n"
        "{\n"
        '  "scores": {\n'
        '    "completeness": 0.0-1.0,\n'
        '    "factual_grounding": 0.0-1.0,\n'
        '    "code_correctness": 0.0-1.0,\n'
        '    "source_citation": 0.0-1.0,\n'
        '    "readability": 0.0-1.0\n'
        "  },\n"
        '  "overall_score": 0.0-1.0,\n'
        '  "approved": true/false,\n'
        '  "feedback": {"completeness": "...", "factual_grounding": "...", '
        '"code_correctness": "...", "source_citation": "...", "readability": "..."},\n'
        '  "revision_instructions": "specific improvements needed or empty string if approved",\n'
        '  "revision_targets": ["researcher", "writer"]\n'
        "}\n\n"
        "Scoring guide:\n"
        "- completeness: addressed ALL parts of goal? Missing sections = low score\n"
        "- factual_grounding: stats cited with sources? Vague claims = low score\n"
        "- code_correctness: code valid, explained, output shown?\n"
        "- source_citation: URLs provided? Sources named?\n"
        "- readability: clear structure, logical flow, professional tone?\n\n"
        "overall_score = weighted: completeness(0.25) + factual_grounding(0.25) + "
        "code(0.20) + citations(0.15) + readability(0.15)\n\n"
        "approved = overall_score >= 0.7\n\n"
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
                report = "\n\n".join(c["content"] for c in chunks) if chunks else ""

            if not report:
                report = "No report was produced."

            original_goal = subtask.description
            prompt = (
                "Original Goal:\n" + original_goal + "\n\n"
                "Report to Evaluate:\n" + "=" * 60 + "\n"
                + report[:6000] + "\n" + "=" * 60 + "\n\n"
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
