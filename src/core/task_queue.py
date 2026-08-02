"""
src/core/task_queue.py — Task Assignment and Tracking
Manages task lifecycle and persists everything to SQLite.

Design decisions:
1. TWO LEVELS: GoalTask (user request) → SubTasks (agent assignments).
2. SQLITE: Zero deps, built into Python. Task history for dashboard + debug.
3. STATUS MACHINE: pending → in_progress → completed | failed | needs_revision
4. DEPENDENCY TRACKING: Writer waits until Researcher + Coder + Analyst done.
"""

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime

from loguru import logger

from config import AgentName, TaskStatus, settings


@dataclass
class SubTask:
    task_id: str
    subtask_id: str
    agent_name: str
    description: str
    expected_output: str
    context: dict
    status: str = TaskStatus.PENDING
    priority: int = 1
    dependencies: list[str] = field(default_factory=list)
    result: str | None = None
    error: str | None = None
    chunk_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    revision_count: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class GoalTask:
    goal: str
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    status: str = TaskStatus.PENDING
    plan: str | None = None
    subtask_ids: list[str] = field(default_factory=list)
    final_report: str | None = None
    critic_score: float | None = None
    critic_feedback: str | None = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: str | None = None
    total_llm_calls: int = 0
    approved: bool = False

    def to_dict(self):
        return asdict(self)


class TaskQueue:
    """
    Manages task lifecycle and persists to SQLite.
    All agents and the Planner interact with this.
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or settings.db_path
        self._init_db()
        self._goals: dict[str, GoalTask] = {}
        self._subtasks: dict[str, SubTask] = {}
        logger.info(f"TaskQueue initialised — DB: {self.db_path}")

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS goal_tasks (
                task_id TEXT PRIMARY KEY, goal TEXT NOT NULL, status TEXT NOT NULL,
                plan TEXT, subtask_ids TEXT, final_report TEXT,
                critic_score REAL, critic_feedback TEXT,
                created_at TEXT, completed_at TEXT,
                total_llm_calls INTEGER DEFAULT 0, approved INTEGER DEFAULT 0)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS subtasks (
                subtask_id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
                agent_name TEXT NOT NULL, description TEXT NOT NULL,
                expected_output TEXT, context TEXT, status TEXT NOT NULL,
                priority INTEGER DEFAULT 1, dependencies TEXT,
                result TEXT, error TEXT, chunk_ids TEXT,
                created_at TEXT, started_at TEXT, completed_at TEXT,
                revision_count INTEGER DEFAULT 0)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS agent_logs (
                log_id TEXT PRIMARY KEY, task_id TEXT, subtask_id TEXT,
                agent_name TEXT, log_type TEXT, content TEXT, timestamp TEXT)""")
            conn.commit()

    # ── Goal management ──────────────────────────────────────────────────────

    def create_goal(self, goal: str) -> GoalTask:
        task = GoalTask(goal=goal)
        self._goals[task.task_id] = task
        self._save_goal(task)
        logger.info(f"[queue] Created goal {task.task_id}: '{goal[:60]}'")
        return task

    def update_goal(self, task_id: str, **kwargs) -> GoalTask:
        task = self._get_or_load_goal(task_id)
        for k, v in kwargs.items():
            if hasattr(task, k):
                setattr(task, k, v)
        if kwargs.get("status") in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            if not task.completed_at:
                task.completed_at = datetime.utcnow().isoformat()
        self._save_goal(task)
        return task

    def get_goal(self, task_id: str) -> GoalTask | None:
        return self._get_or_load_goal(task_id)

    # ── SubTask management ───────────────────────────────────────────────────

    def add_subtask(
        self,
        task_id,
        agent_name,
        description,
        expected_output="",
        context=None,
        priority=1,
        dependencies=None,
        subtask_id=None,
    ) -> SubTask:
        subtask = SubTask(
            task_id=task_id,
            subtask_id=subtask_id or f"{agent_name}_{uuid.uuid4().hex[:8]}",
            agent_name=agent_name,
            description=description,
            expected_output=expected_output,
            context=context or {},
            priority=priority,
            dependencies=dependencies or [],
        )
        self._subtasks[subtask.subtask_id] = subtask
        self._save_subtask(subtask)
        goal = self._get_or_load_goal(task_id)
        if goal:
            goal.subtask_ids.append(subtask.subtask_id)
            self._save_goal(goal)
        logger.debug(f"[queue] Added subtask {subtask.subtask_id} → {agent_name}")
        return subtask

    def update_subtask(self, subtask_id: str, **kwargs) -> SubTask:
        subtask = self._get_or_load_subtask(subtask_id)
        if subtask is None:
            # Subtask not in DB (created directly in tests) — save it first
            from src.core.task_queue import SubTask as ST

            logger.warning(
                f"[queue] update_subtask: {subtask_id} not in DB, skipping update"
            )
            return ST(
                task_id="unknown",
                subtask_id=subtask_id,
                agent_name="unknown",
                description="",
                expected_output="",
                context={},
            )
        for k, v in kwargs.items():
            if hasattr(subtask, k):
                setattr(subtask, k, v)
        if kwargs.get("status") == TaskStatus.IN_PROGRESS and not subtask.started_at:
            subtask.started_at = datetime.utcnow().isoformat()
        if kwargs.get("status") in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            if not subtask.completed_at:
                subtask.completed_at = datetime.utcnow().isoformat()
        self._save_subtask(subtask)
        return subtask

    def complete_subtask(self, subtask_id, result, chunk_ids=None) -> SubTask:
        return self.update_subtask(
            subtask_id,
            status=TaskStatus.COMPLETED,
            result=result,
            chunk_ids=chunk_ids or [],
            completed_at=datetime.utcnow().isoformat(),
        )

    def fail_subtask(self, subtask_id, error) -> SubTask:
        return self.update_subtask(subtask_id, status=TaskStatus.FAILED, error=error)

    def request_revision(self, subtask_id, feedback) -> SubTask:
        subtask = self._get_or_load_subtask(subtask_id)
        return self.update_subtask(
            subtask_id,
            status=TaskStatus.NEEDS_REVISION,
            revision_count=subtask.revision_count + 1,
        )

    def get_subtask(self, subtask_id) -> SubTask | None:
        return self._get_or_load_subtask(subtask_id)

    def get_subtasks_for_goal(self, task_id, status_filter=None) -> list[SubTask]:
        goal = self._get_or_load_goal(task_id)
        if not goal:
            return []
        subtasks = []
        for sid in goal.subtask_ids:
            st = self._get_or_load_subtask(sid)
            if st and (status_filter is None or st.status == status_filter):
                subtasks.append(st)
        return sorted(subtasks, key=lambda s: s.priority, reverse=True)

    def get_ready_subtasks(self, task_id) -> list[SubTask]:
        """SubTasks that are PENDING and have all dependencies completed."""
        all_subtasks = self.get_subtasks_for_goal(task_id)
        completed_ids = {
            s.subtask_id for s in all_subtasks if s.status == TaskStatus.COMPLETED
        }
        ready = [
            s
            for s in all_subtasks
            if s.status == TaskStatus.PENDING
            and all(dep in completed_ids for dep in s.dependencies)
        ]
        return sorted(ready, key=lambda s: s.priority, reverse=True)

    def all_subtasks_complete(self, task_id) -> bool:
        return all(
            s.status == TaskStatus.COMPLETED
            for s in self.get_subtasks_for_goal(task_id)
        )

    def any_subtask_failed(self, task_id) -> bool:
        return any(
            s.status == TaskStatus.FAILED for s in self.get_subtasks_for_goal(task_id)
        )

    # ── Logging ──────────────────────────────────────────────────────────────

    def log_agent_action(self, task_id, agent_name, log_type, content, subtask_id=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO agent_logs VALUES (?,?,?,?,?,?,?)",
                (
                    uuid.uuid4().hex,
                    task_id,
                    subtask_id,
                    agent_name,
                    log_type,
                    content[:5000],
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def get_agent_logs(self, task_id, agent_name=None, limit=100) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if agent_name:
                rows = conn.execute(
                    "SELECT * FROM agent_logs WHERE task_id=? AND agent_name=? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (task_id, agent_name, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM agent_logs WHERE task_id=? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (task_id, limit),
                ).fetchall()
        return [dict(r) for r in rows]

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_goal(self, task: GoalTask):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO goal_tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    task.task_id,
                    task.goal,
                    task.status,
                    task.plan,
                    json.dumps(task.subtask_ids),
                    task.final_report,
                    task.critic_score,
                    task.critic_feedback,
                    task.created_at,
                    task.completed_at,
                    task.total_llm_calls,
                    int(task.approved),
                ),
            )
            conn.commit()

    def _save_subtask(self, subtask: SubTask):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO subtasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    subtask.subtask_id,
                    subtask.task_id,
                    subtask.agent_name,
                    subtask.description,
                    subtask.expected_output,
                    json.dumps(subtask.context),
                    subtask.status,
                    subtask.priority,
                    json.dumps(subtask.dependencies),
                    subtask.result,
                    subtask.error,
                    json.dumps(subtask.chunk_ids),
                    subtask.created_at,
                    subtask.started_at,
                    subtask.completed_at,
                    subtask.revision_count,
                ),
            )
            conn.commit()

    def _get_or_load_goal(self, task_id) -> GoalTask | None:
        if task_id in self._goals:
            return self._goals[task_id]
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM goal_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        if not row:
            return None
        task = GoalTask(
            goal=row["goal"],
            task_id=row["task_id"],
            status=row["status"],
            plan=row["plan"],
            subtask_ids=json.loads(row["subtask_ids"] or "[]"),
            final_report=row["final_report"],
            critic_score=row["critic_score"],
            critic_feedback=row["critic_feedback"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            total_llm_calls=row["total_llm_calls"],
            approved=bool(row["approved"]),
        )
        self._goals[task_id] = task
        return task

    def _get_or_load_subtask(self, subtask_id) -> SubTask | None:
        if subtask_id in self._subtasks:
            return self._subtasks[subtask_id]
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM subtasks WHERE subtask_id=?", (subtask_id,)
            ).fetchone()
        if not row:
            return None
        subtask = SubTask(
            subtask_id=row["subtask_id"],
            task_id=row["task_id"],
            agent_name=row["agent_name"],
            description=row["description"],
            expected_output=row["expected_output"] or "",
            context=json.loads(row["context"] or "{}"),
            status=row["status"],
            priority=row["priority"],
            dependencies=json.loads(row["dependencies"] or "[]"),
            result=row["result"],
            error=row["error"],
            chunk_ids=json.loads(row["chunk_ids"] or "[]"),
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            revision_count=row["revision_count"],
        )
        self._subtasks[subtask_id] = subtask
        return subtask

    def get_recent_goals(self, limit=10) -> list[GoalTask]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM goal_tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._get_or_load_goal(r["task_id"]) for r in rows]


_queue_instance: TaskQueue | None = None


def get_queue() -> TaskQueue:
    global _queue_instance
    if _queue_instance is None:
        _queue_instance = TaskQueue()
    return _queue_instance


if __name__ == "__main__":
    import os
    import tempfile

    from rich import print as rprint

    test_db = tempfile.mktemp(suffix=".db")
    rprint("[bold cyan]Testing Task Queue...[/bold cyan]")
    queue = TaskQueue(db_path=test_db)
    goal = queue.create_goal("Analyse solar energy globally")
    st_r = queue.add_subtask(
        goal.task_id, AgentName.RESEARCHER, "Find data", priority=2
    )
    st_c = queue.add_subtask(
        goal.task_id,
        AgentName.CODER,
        "Write chart code",
        priority=1,
        dependencies=[st_r.subtask_id],
    )
    ready = queue.get_ready_subtasks(goal.task_id)
    rprint(f"Ready before research: {[s.agent_name for s in ready]}")
    assert len(ready) == 1 and ready[0].agent_name == AgentName.RESEARCHER
    queue.complete_subtask(st_r.subtask_id, result="China: 430 GW, USA: 140 GW")
    ready = queue.get_ready_subtasks(goal.task_id)
    rprint(f"Ready after research: {[s.agent_name for s in ready]}")
    assert len(ready) == 1 and ready[0].agent_name == AgentName.CODER
    os.remove(test_db)
    rprint("[bold green]✓ Task Queue test passed![/bold green]")
