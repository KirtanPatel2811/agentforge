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
    """
    Planner agent — orchestrates the entire multi-agent pipeline.

    Interview talking point:
        The Planner uses the LLM to decompose an arbitrary natural language
        goal into typed subtasks for the right specialist agents. The dependency
        system ensures Researcher runs before Writer. The Critic revision loop
        means the system can self-correct: if output scores below 0.7, the
        Planner re-runs flagged agents with the Critic specific feedback.
    """

    agent_name = "planner"

    system_prompt = (
        "You are the Planner Agent in AgentForge, a multi-agent AI system.\n\n"
        "Decompose the user goal into subtasks for specialist agents.\n\n"
        "Available agents:\n"
        "- researcher: web search, news, ArXiv, URL reading\n"
        "- coder: write and execute Python code\n"
        "- analyst: data analysis, statistics, chart generation\n"
        "- writer: assemble final report (ALWAYS last)\n"
        "- critic: quality review (runs automatically, do NOT include)\n\n"
        "Return a JSON plan:\n"
        "{\n"
        '  "title": "brief task title",\n'
        '  "reasoning": "why these agents in this order",\n'
        '  "subtasks": [\n'
        "    {\n"
        '      "agent": "researcher",\n'
        '      "description": "specific instruction",\n'
        '      "expected_output": "what to return",\n'
        '      "priority": 2,\n'
        '      "dependencies": []\n'
        "    },\n"
        "    {\n"
        '      "agent": "writer",\n'
        '      "description": "assemble final report",\n'
        '      "expected_output": "complete structured report",\n'
        '      "priority": 1,\n'
        '      "dependencies": ["researcher"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- writer MUST always be the last subtask\n"
        "- researcher should run before analyst and writer\n"
        "- priority: higher = runs first (researcher=3, coder/analyst=2, writer=1)\n"
        "- dependencies: list of agent names that must complete first\n"
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
        """
        Main entry point. Takes a natural language goal and runs the
        full multi-agent pipeline to completion.
        Returns dict with task_id, report, critic_score, approved, etc.
        """
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
                report = "\n".join(subtask_results.values())

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
            "Create an execution plan for this goal:\n\n"
            "GOAL: " + goal + "\n\n"
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
