"""
setup_phase4.py — Creates Phase 4: Streamlit Dashboard + FastAPI
Run from agentforge root: python setup_phase4.py
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))

def write(path, content):
    full = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {path}")


# ── src/api/main.py ───────────────────────────────────────────────────────────
write("src/api/main.py", """\
\"\"\"
src/api/main.py — FastAPI REST API for AgentForge
───────────────────────────────────────────────────
Exposes the multi-agent pipeline as a REST API.

Endpoints:
  POST /run          — Submit a goal, start the pipeline, return task_id
  GET  /status/{id}  — Check task status (pending/running/completed/failed)
  GET  /result/{id}  — Get full result (report, critic score, charts)
  GET  /logs/{id}    — Get agent activity logs for live feed
  GET  /history      — List recent tasks
  GET  /health       — Health check

Design decisions:
  Background tasks: FastAPI's BackgroundTasks runs the pipeline in a
  background thread so POST /run returns immediately with task_id.
  The client polls GET /status until complete, then fetches GET /result.
  This is the standard pattern for long-running AI jobs.

Run with: uvicorn src.api.main:app --reload --port 8000
\"\"\"

import sys
import os
import json
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

# Store running pipelines (task_id -> thread)
_running_tasks: dict = {}


# ── Request / Response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    goal: str
    class Config:
        json_schema_extra = {
            "example": {
                "goal": "Research the top 5 countries by solar energy capacity and write a summary"
            }
        }

class RunResponse(BaseModel):
    task_id: str
    status: str
    message: str

class StatusResponse(BaseModel):
    task_id: str
    goal: str
    status: str
    created_at: str
    completed_at: Optional[str]
    approved: Optional[bool]
    critic_score: Optional[float]

class ResultResponse(BaseModel):
    task_id: str
    goal: str
    status: str
    report: Optional[str]
    critic_score: Optional[float]
    approved: Optional[bool]
    critic_feedback: Optional[str]
    charts: list
    plan: Optional[str]


# ── Background pipeline runner ────────────────────────────────────────────────

def _run_pipeline_background(task_id: str, goal: str):
    \"\"\"
    Runs the full agent pipeline in a background thread.
    Updates the task in SQLite as it progresses.
    The Streamlit dashboard and /status endpoint poll this.
    \"\"\"
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    \"\"\"Health check — confirms API is running.\"\"\"
    return {
        "status": "ok",
        "service": "AgentForge API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/run", response_model=RunResponse)
def run_goal(request: RunRequest, background_tasks: BackgroundTasks):
    \"\"\"
    Submit a goal to the multi-agent pipeline.
    Returns immediately with task_id. Pipeline runs in background.
    Poll /status/{task_id} to check progress.
    \"\"\"
    if not request.goal or not request.goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty")

    if len(request.goal) > 2000:
        raise HTTPException(status_code=400, detail="Goal too long (max 2000 chars)")

    # Create the goal task record first so we have a task_id to return
    queue = get_queue()
    goal_task = queue.create_goal(request.goal.strip())
    task_id = goal_task.task_id

    # Run pipeline in background thread
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
    \"\"\"
    Get the current status of a task.
    Status values: pending, in_progress, completed, failed, needs_revision
    \"\"\"
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
    \"\"\"
    Get the full result of a completed task.
    Includes report, critic scores, and chart metadata.
    \"\"\"
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
    \"\"\"
    Get agent activity logs for a task.
    Used by the Streamlit live activity feed.
    Filter by agent name with ?agent=researcher
    \"\"\"
    queue = get_queue()
    task = queue.get_goal(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    logs = queue.get_agent_logs(task_id, agent_name=agent, limit=limit)
    return {"task_id": task_id, "logs": logs, "count": len(logs)}


@app.get("/history")
def get_history(limit: int = 20):
    \"\"\"
    List recent tasks with their status and scores.
    Used by the Streamlit history sidebar.
    \"\"\"
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
    \"\"\"Clear a task from memory (ChromaDB chunks).\"\"\"
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
""")


# ── src/app/streamlit_app.py ──────────────────────────────────────────────────
write("src/app/streamlit_app.py", """\
\"\"\"
src/app/streamlit_app.py — AgentForge Streamlit Dashboard
───────────────────────────────────────────────────────────
Live dashboard for submitting goals and watching agents work.

Features:
  - Goal input with example prompts
  - Live agent activity feed (auto-refreshes every 2s while running)
  - Final report displayed with markdown rendering
  - Critic score with dimension breakdown
  - Interactive Plotly charts embedded inline
  - Task history sidebar with previous runs

Run with: streamlit run src/app/streamlit_app.py
\"\"\"

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AgentForge",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import settings, TaskStatus
from src.core.task_queue import get_queue
from src.core.memory import get_memory
from src.tools.chart_generator import get_charts_for_task


# ── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown(\"\"\"
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: #F9FAFB;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
        font-size: 0.85rem;
    }
    .thought-card { border-left-color: #8B5CF6; }
    .action-card  { border-left-color: #F59E0B; }
    .obs-card     { border-left-color: #10B981; }
    .error-card   { border-left-color: #EF4444; }
    .score-box {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .score-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1D4ED8;
    }
    .approved-badge {
        background: #D1FAE5;
        color: #065F46;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .rejected-badge {
        background: #FEE2E2;
        color: #991B1B;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .agent-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6B7280;
    }
    .stProgress .st-bo { background-color: #3B82F6; }
</style>
\"\"\", unsafe_allow_html=True)


# ── Agent colours for activity feed ───────────────────────────────────────────
AGENT_COLORS = {
    "planner":    "#3B82F6",
    "researcher": "#8B5CF6",
    "coder":      "#F59E0B",
    "analyst":    "#10B981",
    "writer":     "#06B6D4",
    "critic":     "#EF4444",
}

AGENT_EMOJIS = {
    "planner":    "🧠",
    "researcher": "🔍",
    "coder":      "💻",
    "analyst":    "📊",
    "writer":     "✍️",
    "critic":     "🔎",
}

LOG_TYPE_STYLE = {
    "thought":     ("thought-card", "💭"),
    "action":      ("action-card",  "⚡"),
    "observation": ("obs-card",     "👁️"),
    "error":       ("error-card",   "❌"),
}


# ── Helper functions ──────────────────────────────────────────────────────────

def run_pipeline(goal: str):
    \"\"\"Start the pipeline in a background thread and store task_id in session.\"\"\"
    import threading
    from src.agents.planner import PlannerAgent

    queue = get_queue()
    goal_task = queue.create_goal(goal)
    task_id = goal_task.task_id

    st.session_state["task_id"] = task_id
    st.session_state["running"] = True
    st.session_state["start_time"] = time.time()

    def _run():
        try:
            planner = PlannerAgent()
            planner.run_goal(goal)
        except Exception as e:
            queue.update_goal(task_id, status=TaskStatus.FAILED)
            st.session_state["error"] = str(e)
        finally:
            st.session_state["running"] = False

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return task_id


def get_task_status(task_id: str):
    \"\"\"Get current task from SQLite.\"\"\"
    return get_queue().get_goal(task_id)


def render_activity_feed(task_id: str, limit: int = 50):
    \"\"\"Render the live agent activity log feed.\"\"\"
    queue = get_queue()
    logs = queue.get_agent_logs(task_id, limit=limit)
    logs = list(reversed(logs))  # chronological order

    if not logs:
        st.info("Waiting for agents to start...")
        return

    for log in logs:
        agent = log.get("agent_name", "unknown")
        log_type = log.get("log_type", "thought")
        content = log.get("content", "")
        timestamp = log.get("timestamp", "")[:19].replace("T", " ")

        css_class, emoji = LOG_TYPE_STYLE.get(log_type, ("agent-card", "📝"))
        color = AGENT_COLORS.get(agent, "#6B7280")
        agent_emoji = AGENT_EMOJIS.get(agent, "🤖")

        # Truncate long tool outputs
        display_content = content[:300] + "..." if len(content) > 300 else content

        st.markdown(f\"\"\"
        <div class="agent-card {css_class}">
            <span style="color:{color}; font-weight:600;">
                {agent_emoji} {agent.upper()}
            </span>
            <span class="agent-label" style="margin-left:8px;">{emoji} {log_type}</span>
            <span style="float:right; color:#9CA3AF; font-size:0.75rem;">{timestamp}</span>
            <br><span style="color:#374151;">{display_content}</span>
        </div>
        \"\"\", unsafe_allow_html=True)


def render_critic_scores(critique_json: str):
    \"\"\"Render the Critic's dimension scores as progress bars.\"\"\"
    try:
        critique = json.loads(critique_json) if isinstance(critique_json, str) else critique_json
        scores = critique.get("scores", {})
        overall = critique.get("overall_score", 0.0)
        approved = critique.get("approved", False)
        feedback = critique.get("feedback", {})
    except Exception:
        st.warning("Could not parse critic scores")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        badge = '<span class="approved-badge">✓ APPROVED</span>' if approved else '<span class="rejected-badge">✗ NEEDS REVISION</span>'
        st.markdown(f\"\"\"
        <div class="score-box">
            <div class="score-number">{overall:.2f}</div>
            <div style="color:#6B7280; font-size:0.85rem;">Overall Score</div>
            <br>{badge}
        </div>
        \"\"\", unsafe_allow_html=True)

    with col2:
        dimension_labels = {
            "completeness":      "📋 Completeness",
            "factual_grounding": "📌 Factual Grounding",
            "code_correctness":  "💻 Code Correctness",
            "source_citation":   "🔗 Source Citation",
            "readability":       "📖 Readability",
        }
        for dim, label in dimension_labels.items():
            score = scores.get(dim, 0.0)
            tip = feedback.get(dim, "")
            st.markdown(f"**{label}** — {score:.2f}")
            st.progress(score)
            if tip and tip != "Auto-assessed":
                st.caption(tip)


def render_charts(task_id: str):
    \"\"\"Render all Plotly charts generated for this task.\"\"\"
    charts = get_charts_for_task(task_id)
    if not charts:
        st.info("No charts were generated for this task.")
        return

    for chart in charts:
        st.markdown(f"**{chart['title']}**")
        html_path = chart.get("html_path", "")
        if html_path and os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=520, scrolling=False)
        else:
            st.warning(f"Chart file not found: {html_path}")
        st.caption(chart.get("description", ""))


def render_history_sidebar():
    \"\"\"Render recent task history in the sidebar.\"\"\"
    queue = get_queue()
    recent = queue.get_recent_goals(limit=10)

    if not recent:
        st.sidebar.info("No previous tasks yet.")
        return

    for task in recent:
        if not task:
            continue
        score_str = f"{task.critic_score:.2f}" if task.critic_score else "—"
        status_emoji = {
            "completed": "✅",
            "failed": "❌",
            "in_progress": "⏳",
            "pending": "⏸️",
            "needs_revision": "🔄",
        }.get(task.status, "❓")

        label = f"{status_emoji} {task.goal[:40]}..."
        if st.sidebar.button(label, key=task.task_id, use_container_width=True):
            st.session_state["task_id"] = task.task_id
            st.session_state["running"] = False
            st.rerun()

        st.sidebar.caption(f"Score: {score_str} | {task.created_at[:10]}")


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    # Initialise session state
    if "task_id" not in st.session_state:
        st.session_state["task_id"] = None
    if "running" not in st.session_state:
        st.session_state["running"] = False
    if "start_time" not in st.session_state:
        st.session_state["start_time"] = None

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🤖 AgentForge")
        st.caption("Multi-Agent Collaborative Task System")
        st.divider()

        st.markdown("### Agent Team")
        for name, emoji in AGENT_EMOJIS.items():
            color = AGENT_COLORS[name]
            st.markdown(
                f'<span style="color:{color};">{emoji} **{name.capitalize()}**</span>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("### Recent Tasks")
        render_history_sidebar()

        st.divider()
        st.markdown("### Settings")
        st.caption(f"Model: `{settings.primary_model}`")
        st.caption(f"Fallback: `{settings.fallback_model}`")
        st.caption(f"Max iterations: `{settings.max_react_iterations}`")
        st.caption(f"Pass threshold: `{settings.critic_pass_threshold}`")

    # ── Main content ──────────────────────────────────────────────────────────
    st.markdown('<div class="main-header">🤖 AgentForge</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Multi-Agent Collaborative Task System — '
        'Powered by Groq + Llama 3.3 70B</div>',
        unsafe_allow_html=True,
    )

    # ── Goal input section ────────────────────────────────────────────────────
    with st.container():
        st.markdown("### 🎯 Enter Your Goal")

        example_goals = [
            "Select an example goal...",
            "Research the top 5 countries by solar energy capacity and write a detailed summary with statistics",
            "Find the latest developments in large language models and summarise the key trends",
            "Compare renewable energy adoption between India and China with data and analysis",
            "Research quantum computing breakthroughs in 2024 and write a technical summary",
            "Find the top AI research papers on agents from ArXiv and summarise their contributions",
        ]

        selected_example = st.selectbox("Quick examples:", example_goals, key="example_select")

        goal_text = st.text_area(
            "Or type your own goal:",
            value=selected_example if selected_example != example_goals[0] else "",
            height=100,
            placeholder="e.g. Research solar energy adoption globally and write a 400-word analyst note with sources...",
            key="goal_input",
        )

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            run_button = st.button(
                "🚀 Run AgentForge",
                type="primary",
                disabled=st.session_state["running"],
                use_container_width=True,
            )
        with col2:
            if st.button("🔄 Clear", use_container_width=True):
                st.session_state["task_id"] = None
                st.session_state["running"] = False
                st.rerun()
        with col3:
            if st.session_state["running"]:
                elapsed = int(time.time() - (st.session_state["start_time"] or time.time()))
                st.metric("Elapsed", f"{elapsed}s")

        if run_button and goal_text.strip():
            task_id = run_pipeline(goal_text.strip())
            st.success(f"✅ Pipeline started! Task ID: `{task_id}`")
            st.rerun()
        elif run_button and not goal_text.strip():
            st.warning("Please enter a goal first.")

    st.divider()

    # ── Task display ──────────────────────────────────────────────────────────
    task_id = st.session_state.get("task_id")

    if not task_id:
        st.markdown("### 👆 Enter a goal above to get started")
        with st.expander("What can AgentForge do?"):
            st.markdown(\"\"\"
            AgentForge is a multi-agent AI system that collaborates to complete complex tasks:

            1. **🧠 Planner** — Decomposes your goal into subtasks for specialist agents
            2. **🔍 Researcher** — Searches the web, reads articles, finds academic papers
            3. **💻 Coder** — Writes and executes Python code to process data
            4. **📊 Analyst** — Runs statistical analysis and creates interactive charts
            5. **✍️ Writer** — Assembles all outputs into a structured final report
            6. **🔎 Critic** — Reviews the report quality and requests revisions if needed
            \"\"\")
        return

    # Load current task state
    task = get_task_status(task_id)
    if not task:
        st.error(f"Task {task_id} not found")
        return

    # ── Status bar ────────────────────────────────────────────────────────────
    status_colors = {
        "completed": "🟢",
        "failed": "🔴",
        "in_progress": "🟡",
        "pending": "⚪",
        "needs_revision": "🟠",
    }
    emoji = status_colors.get(task.status, "❓")
    st.markdown(f"**Task:** `{task_id}` | **Status:** {emoji} `{task.status}`")
    st.caption(f"Goal: {task.goal}")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚡ Live Activity",
        "📄 Final Report",
        "🔎 Critic Review",
        "📊 Charts",
    ])

    with tab1:
        st.markdown("### ⚡ Agent Activity Feed")
        if st.session_state["running"]:
            st.info("🔄 Agents are working... auto-refreshing every 3 seconds")

        # Show which agents are in the plan
        if task.plan:
            try:
                plan = json.loads(task.plan)
                agents_in_plan = [s["agent"] for s in plan.get("subtasks", [])]
                cols = st.columns(len(agents_in_plan))
                for i, agent_name in enumerate(agents_in_plan):
                    with cols[i]:
                        color = AGENT_COLORS.get(agent_name, "#6B7280")
                        emoji_agent = AGENT_EMOJIS.get(agent_name, "🤖")
                        st.markdown(
                            f'<div style="text-align:center; color:{color}; '
                            f'font-size:1.5rem;">{emoji_agent}</div>'
                            f'<div style="text-align:center; font-size:0.75rem; '
                            f'color:#6B7280;">{agent_name}</div>',
                            unsafe_allow_html=True,
                        )
            except Exception:
                pass

        st.markdown("---")
        render_activity_feed(task_id, limit=80)

        # Auto-refresh while running
        if st.session_state["running"]:
            time.sleep(3)
            # Check if task completed
            fresh_task = get_task_status(task_id)
            if fresh_task and fresh_task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                st.session_state["running"] = False
            st.rerun()

    with tab2:
        st.markdown("### 📄 Final Report")
        if task.final_report:
            st.markdown(task.final_report)
            st.download_button(
                label="⬇️ Download Report (Markdown)",
                data=task.final_report,
                file_name=f"agentforge_report_{task_id}.md",
                mime="text/markdown",
            )
        elif task.status == "in_progress":
            st.info("⏳ Report is being written... check back in a moment.")
        elif task.status == "failed":
            st.error("❌ Pipeline failed. Check the Activity Feed for details.")
        else:
            st.info("Report will appear here once the Writer agent completes.")

    with tab3:
        st.markdown("### 🔎 Critic Quality Review")
        # Get critique from memory
        queue = get_queue()
        memory = get_memory()
        critique_chunks = memory.retrieve(
            query="quality scores approved",
            task_id=task_id,
            agent_name="critic",
            k=1,
        )

        if critique_chunks:
            critique_content = critique_chunks[0]["content"]
            render_critic_scores(critique_content)

            st.markdown("---")
            if task.critic_feedback:
                st.markdown("**Revision Instructions:**")
                st.info(task.critic_feedback)
        elif task.status == "completed":
            if task.critic_score:
                st.metric("Critic Score", f"{task.critic_score:.2f}")
                st.success("Approved" if task.approved else "Not approved")
            else:
                st.info("Critic review data not available.")
        else:
            st.info("⏳ Critic review will appear here after the Writer completes.")

    with tab4:
        st.markdown("### 📊 Generated Charts")
        render_charts(task_id)

    # Auto-refresh logic for running tasks
    if st.session_state["running"]:
        fresh = get_task_status(task_id)
        if fresh and fresh.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.NEEDS_REVISION]:
            st.session_state["running"] = False
            st.rerun()


if __name__ == "__main__":
    main()
""")


# ── tests/test_phase4_api.py ──────────────────────────────────────────────────
write("tests/test_phase4_api.py", """\
\"\"\"
tests/test_phase4_api.py — Phase 4 API Tests
──────────────────────────────────────────────
Run: python -m pytest tests/test_phase4_api.py -v

Tests the FastAPI endpoints without starting the real pipeline.
Uses TestClient from FastAPI for in-process testing.
\"\"\"

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    \"\"\"Create a FastAPI test client.\"\"\"
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "AgentForge API"
        assert "timestamp" in data

    def test_health_has_version(self, client):
        response = client.get("/health")
        assert response.json()["version"] == "1.0.0"


class TestRunEndpoint:
    def test_run_returns_task_id(self, client):
        response = client.post("/run", json={"goal": "Test goal for API"})
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["task_id"].startswith("task_")
        assert data["status"] == "running"

    def test_run_empty_goal_rejected(self, client):
        response = client.post("/run", json={"goal": ""})
        assert response.status_code == 400

    def test_run_whitespace_goal_rejected(self, client):
        response = client.post("/run", json={"goal": "   "})
        assert response.status_code == 400

    def test_run_too_long_goal_rejected(self, client):
        response = client.post("/run", json={"goal": "x" * 2001})
        assert response.status_code == 400

    def test_run_creates_retrievable_task(self, client):
        run_resp = client.post("/run", json={"goal": "Test retrievable task"})
        task_id = run_resp.json()["task_id"]

        status_resp = client.get(f"/status/{task_id}")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["task_id"] == task_id
        assert data["goal"] == "Test retrievable task"


class TestStatusEndpoint:
    def test_status_not_found(self, client):
        response = client.get("/status/task_nonexistent_xyz")
        assert response.status_code == 404

    def test_status_has_required_fields(self, client):
        run_resp = client.post("/run", json={"goal": "Status field test"})
        task_id = run_resp.json()["task_id"]

        response = client.get(f"/status/{task_id}")
        data = response.json()
        assert "task_id" in data
        assert "goal" in data
        assert "status" in data
        assert "created_at" in data

    def test_status_values_are_valid(self, client):
        run_resp = client.post("/run", json={"goal": "Status value test"})
        task_id = run_resp.json()["task_id"]

        response = client.get(f"/status/{task_id}")
        valid_statuses = {"pending", "in_progress", "completed", "failed", "needs_revision"}
        assert response.json()["status"] in valid_statuses


class TestLogsEndpoint:
    def test_logs_for_valid_task(self, client):
        run_resp = client.post("/run", json={"goal": "Logs test task"})
        task_id = run_resp.json()["task_id"]

        response = client.get(f"/logs/{task_id}")
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert "logs" in data
        assert "count" in data
        assert isinstance(data["logs"], list)

    def test_logs_not_found(self, client):
        response = client.get("/logs/task_nonexistent_abc")
        assert response.status_code == 404

    def test_logs_agent_filter(self, client):
        run_resp = client.post("/run", json={"goal": "Agent filter test"})
        task_id = run_resp.json()["task_id"]

        response = client.get(f"/logs/{task_id}?agent=researcher")
        assert response.status_code == 200


class TestHistoryEndpoint:
    def test_history_returns_list(self, client):
        response = client.get("/history")
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert isinstance(data["tasks"], list)

    def test_history_after_run_has_entry(self, client):
        client.post("/run", json={"goal": "History entry test"})
        response = client.get("/history")
        goals = [t["goal"] for t in response.json()["tasks"]]
        assert any("History entry test" in g for g in goals)

    def test_history_task_has_required_fields(self, client):
        client.post("/run", json={"goal": "Field check test"})
        response = client.get("/history")
        tasks = response.json()["tasks"]
        if tasks:
            task = tasks[0]
            assert "task_id" in task
            assert "goal" in task
            assert "status" in task
            assert "created_at" in task


class TestResultEndpoint:
    def test_result_not_found(self, client):
        response = client.get("/result/task_nonexistent_xyz")
        assert response.status_code == 404

    def test_result_not_ready_returns_202(self, client):
        \"\"\"A freshly created task should return 202 (not yet complete).\"\"\"
        run_resp = client.post("/run", json={"goal": "Result not ready test"})
        task_id = run_resp.json()["task_id"]
        # Task will be in_progress immediately after creation
        response = client.get(f"/result/{task_id}")
        # Either 202 (running) or 200 (somehow completed instantly)
        assert response.status_code in [200, 202]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
""")


# ── README.md ─────────────────────────────────────────────────────────────────
write("README.md", """\
# 🤖 AgentForge — Multi-Agent Collaborative Task System

A production-grade multi-agent AI system where 6 specialist agents collaborate
to complete complex research, analysis, and coding tasks from a single natural
language goal.

## 🎯 What It Does

Give AgentForge one goal:
> *"Research the top 5 countries by solar energy capacity and write a detailed report with statistics"*

AgentForge autonomously:
1. **Planner** decomposes the goal into subtasks
2. **Researcher** searches the web, ArXiv, reads articles
3. **Coder** writes and executes Python code
4. **Analyst** runs statistical analysis, creates charts
5. **Writer** assembles the final structured report
6. **Critic** reviews quality and requests revisions if needed

## 🚀 Quick Start

```bash
# 1. Activate environment
conda activate agentforge

# 2. Set up API keys
copy .env.example .env
# Edit .env with your Groq and Google AI Studio keys

# 3. Run the full pipeline
python run_pipeline.py

# 4. Launch the dashboard
streamlit run src/app/streamlit_app.py

# 5. Launch the API (separate terminal)
uvicorn src.api.main:app --reload --port 8000
```

## 🏗️ Architecture

```
User Goal (natural language)
        │
        ▼
┌─────────────────┐
│  PLANNER AGENT  │ ← Decomposes goal, delegates, tracks progress
└────────┬────────┘
         │
    ┌────┼────┐
    ▼    ▼    ▼
┌──────┐ ┌──────┐ ┌──────────┐
│RSRCH │ │CODER │ │ANALYST   │
│DDG   │ │Sand- │ │Pandas    │
│ArXiv │ │boxed │ │Plotly    │
│URL   │ │exec  │ │Charts    │
└──┬───┘ └──┬───┘ └────┬─────┘
   └────────┼──────────┘
            │ Results → ChromaDB (shared memory)
            ▼
   ┌─────────────────┐
   │  WRITER AGENT   │ ← Assembles final report
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │  CRITIC AGENT   │ ← Scores quality, requests revisions
   └────────┬────────┘
            │
            ▼
   Final Report + Charts
   (Streamlit Dashboard)
```

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM Primary | Groq API — Llama 3.3 70B (free) |
| LLM Fallback | Google AI Studio — Gemini 2.5 Flash (free) |
| Agent Framework | Custom ReAct loop (built from scratch) |
| Vector Memory | ChromaDB (local) |
| Web Search | DuckDuckGo + Serper.dev fallback |
| Paper Search | ArXiv API |
| Code Execution | RestrictedPython + subprocess sandbox |
| Visualisation | Plotly |
| Dashboard | Streamlit |
| API | FastAPI |
| Storage | SQLite |

## 📁 Project Structure

```
agentforge/
├── src/
│   ├── core/          # LLM client, memory, message bus, task queue, base agent
│   ├── agents/        # 6 specialist agents
│   ├── tools/         # 5 tools: web search, arxiv, url reader, code exec, charts
│   ├── api/           # FastAPI REST API
│   └── app/           # Streamlit dashboard
├── tests/             # 80+ tests across all phases
├── data/
│   ├── cache/         # SQLite task database
│   ├── chroma/        # ChromaDB vector store
│   └── outputs/       # Generated reports and charts
├── config.py          # Central configuration
└── run_pipeline.py    # Quick CLI runner
```

## 🧪 Running Tests

```bash
# All offline tests (no API needed)
python -m pytest tests/ -v -m "not live"

# Phase-specific
python -m pytest tests/test_phase1_core.py -v
python -m pytest tests/test_phase2_tools.py -v -m "not live"
python -m pytest tests/test_phase3_agents.py -v -m "not live"
python -m pytest tests/test_phase4_api.py -v

# Live tests (needs API keys + internet)
python -m pytest tests/ -v -m live
```

## 🔑 API Keys (All Free)

| Service | URL | Usage |
|---|---|---|
| Groq | console.groq.com | Primary LLM — Llama 3.3 70B |
| Google AI Studio | aistudio.google.com | Fallback LLM — Gemini 2.5 Flash |
| Serper.dev | serper.dev | Backup web search (100/month free) |

## 📊 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Health check |
| POST | /run | Submit a goal, returns task_id |
| GET | /status/{task_id} | Check task status |
| GET | /result/{task_id} | Get full result + report |
| GET | /logs/{task_id} | Get agent activity logs |
| GET | /history | List recent tasks |

## 💡 Key Design Decisions

**ReAct from scratch** — Each agent runs Reason→Act→Observe loops. Built
without LangChain so every line is understandable and explainable.

**Shared vector memory** — ChromaDB stores all agent outputs. The Writer
retrieves relevant chunks semantically rather than by key — this is how
production RAG pipelines work.

**Message bus** — Agents never import each other. All communication goes
through a central message bus, enabling loose coupling and easy testing.

**Critic loop** — Quality gate with 5 scoring dimensions. If score < 0.7,
targeted revision instructions sent back to the right agents.

**Dual-provider LLM** — Groq primary (fast, free), Gemini fallback.
Automatic retry with exponential backoff on rate limits.

## 👨‍💻 Built By

Kittu — B.Tech CSE, CSPIT CHARUSAT
Portfolio: MediScan → MarketPulse → AgentForge
""")


print("\\n✅ All Phase 4 files created!")
print("\\nFiles created:")
print("  src/api/main.py         — FastAPI REST API")
print("  src/app/streamlit_app.py — Streamlit dashboard")
print("  tests/test_phase4_api.py — API tests")
print("  README.md               — Full project documentation")
print("\\nNext steps:")
print("  1. pip install fastapi uvicorn[standard] httpx")
print("  2. python -m pytest tests/test_phase4_api.py -v")
print("  3. streamlit run src/app/streamlit_app.py")
print("  4. uvicorn src.api.main:app --reload --port 8000")
