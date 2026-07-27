"""
tests/test_phase1_core.py — Phase 1 Integration Tests
Run: pytest tests/test_phase1_core.py -v

All tests are OFFLINE (no API calls needed).
Tests: Config, MessageBus, TaskQueue, AgentMemory, LLMClient JSON parsing.
"""

import os, sys, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings, AgentName, TaskStatus


class TestConfig:
    def test_config_loads(self):
        assert settings is not None

    def test_agent_names(self):
        assert len(AgentName.ALL) == 6
        assert AgentName.PLANNER in AgentName.ALL

    def test_task_statuses(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.COMPLETED == "completed"

    def test_data_dirs_exist(self):
        from config import CACHE_DIR, OUTPUTS_DIR, CHROMA_DIR
        assert CACHE_DIR.exists()
        assert OUTPUTS_DIR.exists()
        assert CHROMA_DIR.exists()


class TestMessageBus:
    def setup_method(self):
        from src.core.message_bus import MessageBus
        self.bus = MessageBus()

    def test_send_and_receive(self):
        from src.core.message_bus import make_task_message
        self.bus.send(make_task_message("planner", "researcher", "t1", "Find solar data"))
        received = self.bus.receive("researcher")
        assert len(received) == 1
        assert received[0].content["description"] == "Find solar data"

    def test_inbox_empty_after_receive(self):
        from src.core.message_bus import make_task_message
        self.bus.send(make_task_message("a", "b", "t1", "do something"))
        self.bus.receive("b")
        assert self.bus.pending_count("b") == 0

    def test_message_history(self):
        from src.core.message_bus import make_task_message, make_result_message
        self.bus.send(make_task_message("planner", "coder", "t2", "write code"))
        self.bus.send(make_result_message("coder", "planner", "t2", "here is code"))
        assert len(self.bus.get_history(task_id="t2")) == 2

    def test_broadcast(self):
        from src.core.message_bus import MessageType
        self.bus.broadcast("planner", ["researcher","coder","analyst"],
                           MessageType.STATUS_UPDATE, "t3", {"msg": "start"})
        assert self.bus.pending_count("researcher") == 1
        assert self.bus.pending_count("coder") == 1
        assert self.bus.pending_count("analyst") == 1

    def test_receive_by_type(self):
        from src.core.message_bus import make_task_message, make_result_message, MessageType
        self.bus.send(make_task_message("planner", "planner", "t4", "internal task"))
        self.bus.send(make_result_message("coder", "planner", "t4", "code output"))
        results = self.bus.receive("planner", message_type=MessageType.RESULT)
        assert len(results) == 1
        assert results[0].content["output"] == "code output"


class TestTaskQueue:
    def setup_method(self):
        import tempfile
        from src.core.task_queue import TaskQueue
        self.db = tempfile.mktemp(suffix=".db")
        self.queue = TaskQueue(db_path=self.db)

    def teardown_method(self):
        if os.path.exists(self.db):
            os.remove(self.db)

    def test_create_goal(self):
        goal = self.queue.create_goal("Test goal")
        assert goal.task_id.startswith("task_")
        assert goal.status == TaskStatus.PENDING

    def test_add_subtask(self):
        goal = self.queue.create_goal("Test")
        st = self.queue.add_subtask(goal.task_id, AgentName.RESEARCHER, "Find data")
        assert st.agent_name == AgentName.RESEARCHER
        assert st.status == TaskStatus.PENDING

    def test_dependency_ordering(self):
        goal = self.queue.create_goal("Dep test")
        st_r = self.queue.add_subtask(goal.task_id, AgentName.RESEARCHER, "Research")
        st_c = self.queue.add_subtask(goal.task_id, AgentName.CODER, "Code",
                                       dependencies=[st_r.subtask_id])
        ready = self.queue.get_ready_subtasks(goal.task_id)
        assert len(ready) == 1 and ready[0].agent_name == AgentName.RESEARCHER
        self.queue.complete_subtask(st_r.subtask_id, result="data found")
        ready = self.queue.get_ready_subtasks(goal.task_id)
        assert len(ready) == 1 and ready[0].agent_name == AgentName.CODER

    def test_complete_subtask(self):
        goal = self.queue.create_goal("Test")
        st = self.queue.add_subtask(goal.task_id, AgentName.CODER, "Write code")
        self.queue.complete_subtask(st.subtask_id, result="print('hello')")
        updated = self.queue.get_subtask(st.subtask_id)
        assert updated.status == TaskStatus.COMPLETED

    def test_all_subtasks_complete(self):
        goal = self.queue.create_goal("Test")
        st1 = self.queue.add_subtask(goal.task_id, AgentName.RESEARCHER, "R")
        st2 = self.queue.add_subtask(goal.task_id, AgentName.CODER, "C")
        self.queue.complete_subtask(st1.subtask_id, result="r")
        assert not self.queue.all_subtasks_complete(goal.task_id)
        self.queue.complete_subtask(st2.subtask_id, result="c")
        assert self.queue.all_subtasks_complete(goal.task_id)

    def test_persistence(self):
        goal = self.queue.create_goal("Persist test")
        gid = goal.task_id
        from src.core.task_queue import TaskQueue
        q2 = TaskQueue(db_path=self.db)
        loaded = q2.get_goal(gid)
        assert loaded is not None and loaded.goal == "Persist test"


class TestAgentMemory:
    def setup_method(self):
        import tempfile, chromadb
        from src.core.memory import AgentMemory
        self.temp_dir = tempfile.mkdtemp()
        self.mem = AgentMemory.__new__(AgentMemory)
        client = chromadb.PersistentClient(path=self.temp_dir)
        self.mem._client = client
        self.mem._collection = client.get_or_create_collection(
            name="test_col", metadata={"hnsw:space": "cosine"})

    def test_store_and_count(self):
        from src.core.memory import ChunkType
        self.mem.store("Solar energy reached 1.6 TW", "researcher", "t1", ChunkType.RESEARCH)
        assert self.mem.count == 1

    def test_semantic_retrieval(self):
        from src.core.memory import ChunkType
        self.mem.store("China has 430 GW of solar power", "researcher", "t1", ChunkType.RESEARCH)
        self.mem.store("Python code for bar charts", "coder", "t1", ChunkType.CODE)
        results = self.mem.retrieve("solar energy China", task_id="t1", k=2)
        assert len(results) >= 1

    def test_empty_returns_empty(self):
        assert self.mem.retrieve("anything", k=5) == []

    def test_task_summary(self):
        from src.core.memory import ChunkType
        self.mem.store("R1", "researcher", "t2", ChunkType.RESEARCH)
        self.mem.store("R2", "researcher", "t2", ChunkType.RESEARCH)
        self.mem.store("C1", "coder", "t2", ChunkType.CODE)
        summary = self.mem.get_task_summary("t2")
        assert len(summary["researcher"]) == 2
        assert len(summary["coder"]) == 1


class TestLLMParsingOffline:
    def test_clean_json(self):
        from src.core.llm_client import parse_json
        assert parse_json('{"a": 1}')["a"] == 1

    def test_markdown_fences(self):
        from src.core.llm_client import parse_json
        assert parse_json('```json\n{"k": "v"}\n```')["k"] == "v"

    def test_json_with_preamble(self):
        from src.core.llm_client import parse_json
        assert parse_json('Here you go:\n{"x": 42}')["x"] == 42

    def test_invalid_raises(self):
        from src.core.llm_client import parse_json, LLMParseError
        with pytest.raises(LLMParseError):
            parse_json("not json at all")

    def test_empty_raises(self):
        from src.core.llm_client import parse_json, LLMParseError
        with pytest.raises(LLMParseError):
            parse_json("")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
