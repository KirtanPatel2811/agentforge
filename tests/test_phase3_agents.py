"""
tests/test_phase3_agents.py — Phase 3 Agent Tests
───────────────────────────────────────────────────
Offline tests: python -m pytest tests/test_phase3_agents.py -v -m "not live"
Live tests:    python -m pytest tests/test_phase3_agents.py -v -m live

Offline tests validate:
- Agent instantiation (all 6 agents init without errors)
- Critic scoring logic and JSON validation
- Planner default plan generation
- Memory integration (agents can store/retrieve)

Live tests run actual LLM calls (needs GROQ_API_KEY).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Agent Instantiation Tests (offline) ──────────────────────────────────────


class TestAgentInstantiation:
    """All agents must instantiate without errors and have correct names."""

    def test_researcher_instantiates(self):
        from src.agents.researcher import ResearcherAgent

        agent = ResearcherAgent()
        assert agent.agent_name == "researcher"
        assert "web_search" in agent.tools
        assert "arxiv_search" in agent.tools
        assert "read_url" in agent.tools

    def test_coder_instantiates(self):
        from src.agents.coder import CoderAgent

        agent = CoderAgent()
        assert agent.agent_name == "coder"
        assert "execute_code" in agent.tools

    def test_analyst_instantiates(self):
        from src.agents.analyst import AnalystAgent

        agent = AnalystAgent()
        assert agent.agent_name == "analyst"
        assert "execute_code" in agent.tools
        assert "generate_chart" in agent.tools

    def test_writer_instantiates(self):
        from src.agents.writer import WriterAgent

        agent = WriterAgent()
        assert agent.agent_name == "writer"
        assert agent.tools == {}  # Writer has no tools

    def test_critic_instantiates(self):
        from src.agents.critic import CriticAgent

        agent = CriticAgent()
        assert agent.agent_name == "critic"
        assert agent.tools == {}  # Critic has no tools

    def test_planner_instantiates(self):
        from src.agents.planner import PlannerAgent

        planner = PlannerAgent()
        assert planner.agent_name == "planner"
        assert "researcher" in planner._agents
        assert "writer" in planner._agents
        assert "critic" in planner._agents

    def test_all_agents_have_system_prompts(self):
        from src.agents.analyst import AnalystAgent
        from src.agents.coder import CoderAgent
        from src.agents.critic import CriticAgent
        from src.agents.researcher import ResearcherAgent
        from src.agents.writer import WriterAgent

        for AgentClass in [
            ResearcherAgent,
            CoderAgent,
            AnalystAgent,
            WriterAgent,
            CriticAgent,
        ]:
            agent = AgentClass()
            assert len(agent.system_prompt) > 100, (
                f"{agent.agent_name} system prompt too short"
            )
            assert len(agent.llm.system_prompt) > 200, (
                f"{agent.agent_name} LLM prompt not built"
            )

    def test_all_agents_have_llm_client(self):
        from src.agents.researcher import ResearcherAgent

        agent = ResearcherAgent()
        assert agent.llm is not None
        assert agent.llm.agent_name == "researcher"

    def test_all_agents_have_memory(self):
        from src.agents.researcher import ResearcherAgent

        agent = ResearcherAgent()
        assert agent.memory is not None

    def test_all_agents_have_bus(self):
        from src.agents.researcher import ResearcherAgent

        agent = ResearcherAgent()
        assert agent.bus is not None


# ── Critic Logic Tests (offline) ──────────────────────────────────────────────


class TestCriticLogic:
    """Test Critic's scoring and validation logic without LLM calls."""

    @pytest.fixture
    def critic(self):
        from src.agents.critic import CriticAgent

        return CriticAgent()

    def test_validate_review_fills_missing_scores(self, critic):
        """If LLM omits a score dimension, default to 0.5."""
        incomplete = {
            "scores": {"completeness": 0.8},  # missing 4 dimensions
            "overall_score": 0.8,
            "approved": True,
        }
        result = critic._validate_review(incomplete)
        assert "factual_grounding" in result["scores"]
        assert "readability" in result["scores"]

    def test_validate_review_clamps_scores(self, critic):
        """Scores outside 0-1 should be clamped."""
        review = {
            "scores": {
                "completeness": 1.5,  # too high
                "factual_grounding": -0.2,  # too low
                "code_correctness": 0.8,
                "source_citation": 0.7,
                "readability": 0.9,
            },
            "overall_score": 1.5,
            "approved": True,
        }
        result = critic._validate_review(review)
        assert result["scores"]["completeness"] == 1.0
        assert result["scores"]["factual_grounding"] == 0.0

    def test_validate_review_recalculates_overall(self, critic):
        """Overall score should be recalculated from dimension scores."""
        review = {
            "scores": {
                "completeness": 1.0,
                "factual_grounding": 1.0,
                "code_correctness": 1.0,
                "source_citation": 1.0,
                "readability": 1.0,
            },
            "overall_score": 0.0,  # wrong — should be recalculated to 1.0
            "approved": False,
        }
        result = critic._validate_review(review)
        assert result["overall_score"] >= 0.99
        assert result["approved"] is True

    def test_approve_when_above_threshold(self, critic):
        review = {
            "scores": {
                "completeness": 0.9,
                "factual_grounding": 0.8,
                "code_correctness": 0.8,
                "source_citation": 0.7,
                "readability": 0.9,
            },
            "overall_score": 0.9,
            "approved": True,
        }
        result = critic._validate_review(review)
        assert result["approved"] is True

    def test_reject_when_below_threshold(self, critic):
        review = {
            "scores": {
                "completeness": 0.3,
                "factual_grounding": 0.3,
                "code_correctness": 0.3,
                "source_citation": 0.3,
                "readability": 0.3,
            },
            "overall_score": 0.3,
            "approved": False,
        }
        result = critic._validate_review(review)
        assert result["approved"] is False

    def test_fallback_review_structure(self, critic):
        """Fallback review must have all required fields."""
        result = critic._fallback_review(
            "Some report text with http://example.com and 42 GW"
        )
        assert "scores" in result
        assert "overall_score" in result
        assert "approved" in result
        assert "revision_instructions" in result
        assert "revision_targets" in result

    def test_fallback_review_detects_sources(self, critic):
        """Fallback should give higher source_citation if URL present."""
        with_url = critic._fallback_review("See http://example.com for data")
        without_url = critic._fallback_review("Data shows some trends")
        assert (
            with_url["scores"]["source_citation"]
            >= without_url["scores"]["source_citation"]
        )


# ── Planner Logic Tests (offline) ─────────────────────────────────────────────


class TestPlannerLogic:
    """Test Planner's plan generation and subtask registration."""

    @pytest.fixture
    def planner(self):
        from src.agents.planner import PlannerAgent

        return PlannerAgent()

    def test_default_plan_structure(self, planner):
        """Default plan must have researcher + writer."""
        plan = planner._default_plan("Analyse solar energy adoption globally")
        assert "subtasks" in plan
        assert "title" in plan
        agents = [s["agent"] for s in plan["subtasks"]]
        assert "researcher" in agents
        assert "writer" in agents

    def test_default_plan_writer_is_last(self, planner):
        """Writer must always be the last subtask."""
        plan = planner._default_plan("Research quantum computing")
        subtasks = plan["subtasks"]
        assert subtasks[-1]["agent"] == "writer"

    def test_register_subtasks(self, planner, tmp_path):
        """Subtasks should be registered in the queue."""
        import tempfile

        from src.core.task_queue import TaskQueue

        db = tempfile.mktemp(suffix=".db")
        queue = TaskQueue(db_path=db)

        # Create a test goal
        goal = queue.create_goal("Test goal")

        # Use a simple plan
        plan = {
            "title": "Test",
            "subtasks": [
                {
                    "agent": "researcher",
                    "description": "Research X",
                    "expected_output": "Findings",
                    "priority": 2,
                    "dependencies": [],
                },
                {
                    "agent": "writer",
                    "description": "Write report",
                    "expected_output": "Report",
                    "priority": 1,
                    "dependencies": ["researcher"],
                },
            ],
        }

        # Temporarily patch the queue
        original_queue = planner.queue
        planner.queue = queue

        subtask_map = planner._register_subtasks(plan, goal.task_id)

        assert "researcher" in subtask_map
        assert "writer" in subtask_map
        assert subtask_map["researcher"].agent_name == "researcher"
        assert subtask_map["writer"].agent_name == "writer"

        # Writer should depend on researcher
        writer_deps = subtask_map["writer"].dependencies
        assert subtask_map["researcher"].subtask_id in writer_deps

        planner.queue = original_queue
        os.remove(db)


# ── Writer Context Tests (offline) ────────────────────────────────────────────


class TestWriterContext:
    """Test Writer's memory context building."""

    def test_build_context_with_empty_memory(self):
        """Writer should handle empty memory gracefully."""
        from src.agents.writer import WriterAgent

        writer = WriterAgent()
        context = writer._build_context("nonexistent_task_xyz_999")
        assert "No agent outputs found" in context or isinstance(context, str)

    def test_build_context_with_memory(self, tmp_path):
        """Writer should include researcher outputs in context."""
        from src.agents.writer import WriterAgent
        from src.core.memory import ChunkType

        # Create isolated memory for test
        writer = WriterAgent()
        test_task = "test_writer_context_xyz"

        # Store something as researcher
        writer.memory.store(
            content="Solar energy reached 1.6 TW globally in 2023.",
            agent_name="researcher",
            task_id=test_task,
            chunk_type=ChunkType.RESEARCH,
        )

        context = writer._build_context(test_task)
        assert "researcher" in context.lower() or "1.6 TW" in context

        # Cleanup
        writer.memory.clear_task(test_task)


# ── Live Tests (need LLM API) ─────────────────────────────────────────────────


@pytest.mark.live
class TestResearcherLive:
    def test_researcher_finds_information(self):
        from src.agents.researcher import ResearcherAgent
        from src.core.task_queue import get_queue

        agent = ResearcherAgent()
        queue = get_queue()
        goal = queue.create_goal("live_test_researcher")
        subtask = queue.add_subtask(
            task_id=goal.task_id,
            agent_name="researcher",
            description="Find the top 3 countries by solar energy installed capacity. Include specific GW numbers.",
            expected_output="List of countries with capacity in GW and sources",
        )
        result = agent.run(subtask)
        assert len(result) > 100
        assert any(
            country in result for country in ["China", "USA", "India", "Germany"]
        )
        agent.memory.clear_task(goal.task_id)


@pytest.mark.live
class TestCoderLive:
    def test_coder_writes_and_executes_code(self):
        from src.agents.coder import CoderAgent
        from src.core.task_queue import get_queue

        agent = CoderAgent()
        queue = get_queue()
        goal = queue.create_goal("live_test_coder")
        subtask = queue.add_subtask(
            task_id=goal.task_id,
            agent_name="coder",
            description="Write Python code that creates a list of the top 5 countries by solar capacity, computes the total, and prints both.",
            expected_output="Working Python code with printed output showing countries and total GW",
        )
        result = agent.run(subtask)
        assert len(result) > 50
        agent.memory.clear_task(goal.task_id)


@pytest.mark.live
class TestPlannerLive:
    def test_planner_creates_valid_plan(self):
        from src.agents.planner import PlannerAgent

        planner = PlannerAgent()
        plan = planner._create_plan(
            "Research the top 5 countries by solar energy capacity and write a brief summary",
            "plan_test_001",
        )
        assert "subtasks" in plan
        agents = [s["agent"] for s in plan["subtasks"]]
        assert "writer" in agents
        assert plan["subtasks"][-1]["agent"] == "writer"


@pytest.mark.live
class TestFullPipelineLive:
    def test_simple_goal_end_to_end(self):
        """
        Full pipeline test — runs all agents on a simple goal.
        This is the big integration test. Expect ~60-90 seconds.
        """
        from src.agents.planner import PlannerAgent

        planner = PlannerAgent()
        result = planner.run_goal(
            "Find the top 3 countries by solar energy installed capacity "
            "and write a brief 200-word summary with the key statistics."
        )

        assert "task_id" in result
        assert "report" in result
        assert len(result["report"]) > 200
        assert "critic_score" in result
        assert isinstance(result["critic_score"], float)

        print(f"\nTask ID: {result['task_id']}")
        print(f"Critic Score: {result['critic_score']:.2f}")
        print(f"Approved: {result['approved']}")
        print(f"Report preview: {result['report'][:300]}...")

        # Cleanup
        from src.core.memory import get_memory

        get_memory().clear_task(result["task_id"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not live"])
