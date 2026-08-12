"""
src/api/main.py — FastAPI REST API for AgentForge
"""

import sys
import os
import threading
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings, TaskStatus
from src.core.task_queue import get_queue
from src.core.memory import get_memory
from src.tools.chart_generator import get_charts_for_task

app = FastAPI(
    title="AgentForge API",
    description="Multi-Agent Collaborative Task System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_running_tasks: dict = {}


class RunRequest(BaseModel):
    goal: str


class RunResponse(BaseModel):
    task_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    task_id: str
    goal: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    approved: Optional[bool] = None
    critic_score: Optional[float] = None


class ResultResponse(BaseModel):
    task_id: str
    goal: str
    status: str
    report: Optional[str] = None
    critic_score: Optional[float] = None
    approved: Optional[bool] = None
    critic_feedback: Optional[str] = None
    charts: list = []
    plan: Optional[str] = None


def _run_pipeline_background(task_id: str, goal: str):
    try:
        from src.agents.planner import PlannerAgent
        planner = PlannerAgent()
        planner.run_goal(goal)
    except Exception as e:
        queue = get_queue()
        queue.update_goal(task_id, status=TaskStatus.FAILED)
        print(f"[api] Pipeline failed for {task_id}: {e}")
    finally:
        _running_tasks.pop(task_id, None)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "AgentForge API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/run", response_model=RunResponse)
def run_goal(request: RunRequest, background_tasks: BackgroundTasks):
    if not request.goal or not request.goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty")

    if len(request.goal) > 2000:
        raise HTTPException(status_code=400, detail="Goal too long (max 2000 chars)")

    queue = get_queue()
    goal_task = queue.create_goal(request.goal.strip())
    task_id = goal_task.task_id

    thread = threading.Thread(
        target=_run_pipeline_background,
        args=(task_id, request.goal.strip()),
        daemon=True,
    )
    _running_tasks[task_id] = thread
    thread.start()

    return RunResponse(
        task_id=task_id,
        status="running",
        message=f"Pipeline started. Poll /status/{task_id} for updates.",
    )


@app.get("/status/{task_id}", response_model=StatusResponse)
def get_status(task_id: str):
    queue = get_queue()
    task = queue.get_goal(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return StatusResponse(
        task_id=task.task_id,
        goal=task.goal,
        status=task.status,
        created_at=task.created_at,
        completed_at=task.completed_at,
        approved=task.approved,
        critic_score=task.critic_score,
    )


@app.get("/result/{task_id}", response_model=ResultResponse)
def get_result(task_id: str):
    queue = get_queue()
    task = queue.get_goal(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task.status not in [TaskStatus.COMPLETED, TaskStatus.NEEDS_REVISION]:
        raise HTTPException(
            status_code=202,
            detail=f"Task not yet complete. Current status: {task.status}",
        )

    charts = get_charts_for_task(task_id)

    return ResultResponse(
        task_id=task.task_id,
        goal=task.goal,
        status=task.status,
        report=task.final_report,
        critic_score=task.critic_score,
        approved=task.approved,
        critic_feedback=task.critic_feedback,
        charts=charts,
        plan=task.plan,
    )


@app.get("/logs/{task_id}")
def get_logs(task_id: str, agent: Optional[str] = None, limit: int = 100):
    queue = get_queue()
    task = queue.get_goal(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    logs = queue.get_agent_logs(task_id, agent_name=agent, limit=limit)
    return {"task_id": task_id, "logs": logs, "count": len(logs)}


@app.get("/history")
def get_history(limit: int = 20):
    queue = get_queue()
    tasks = queue.get_recent_goals(limit=limit)
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "goal": t.goal[:100],
                "status": t.status,
                "critic_score": t.critic_score,
                "approved": t.approved,
                "created_at": t.created_at,
            }
            for t in tasks if t
        ]
    }


@app.delete("/task/{task_id}")
def delete_task(task_id: str):
    memory = get_memory()
    deleted = memory.clear_task(task_id)
    return {"task_id": task_id, "chunks_deleted": deleted}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
    )
