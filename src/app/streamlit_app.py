"""
src/app/streamlit_app.py — AgentForge Streamlit Dashboard
Run: streamlit run src/app/streamlit_app.py
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

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

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(135deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { font-size: 1rem; color: #6B7280; margin-bottom: 2rem; }
    .agent-card {
        background: #F9FAFB; border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem; border-radius: 0 8px 8px 0;
        margin: 0.4rem 0; font-size: 0.85rem;
    }
    .thought-card { border-left-color: #8B5CF6; }
    .action-card  { border-left-color: #F59E0B; }
    .obs-card     { border-left-color: #10B981; }
    .error-card   { border-left-color: #EF4444; }
    .score-box {
        background: #EFF6FF; border: 1px solid #BFDBFE;
        border-radius: 8px; padding: 1rem; text-align: center;
    }
    .score-number { font-size: 2.5rem; font-weight: 700; color: #1D4ED8; }
    .approved-badge {
        background: #D1FAE5; color: #065F46;
        padding: 0.25rem 0.75rem; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600;
    }
    .rejected-badge {
        background: #FEE2E2; color: #991B1B;
        padding: 0.25rem 0.75rem; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

AGENT_COLORS = {
    "planner": "#3B82F6", "researcher": "#8B5CF6",
    "coder": "#F59E0B", "analyst": "#10B981",
    "writer": "#06B6D4", "critic": "#EF4444",
}
AGENT_EMOJIS = {
    "planner": "🧠", "researcher": "🔍",
    "coder": "💻", "analyst": "📊",
    "writer": "✍️", "critic": "🔎",
}
LOG_TYPE_STYLE = {
    "thought": ("thought-card", "💭"),
    "action":  ("action-card",  "⚡"),
    "observation": ("obs-card", "👁️"),
    "error":   ("error-card",   "❌"),
}


def run_pipeline(goal: str):
    import threading

    queue = get_queue()
    goal_task = queue.create_goal(goal)
    task_id = goal_task.task_id

    st.session_state["task_id"] = task_id
    st.session_state["running"] = True
    st.session_state["start_time"] = time.time()

    def _run():
        try:
            from src.agents.planner import PlannerAgent
            # Pass the existing task_id instead of creating a new one
            from src.core.task_queue import TaskStatus
            queue.update_goal(task_id, status=TaskStatus.IN_PROGRESS)

            # Run each agent directly using the existing task_id
            from src.agents.researcher import ResearcherAgent
            from src.agents.writer import WriterAgent
            from src.agents.critic import CriticAgent
            from src.core.llm_client import LLMClient, LLMParseError
            import json

            planner = PlannerAgent()

            # Create plan
            plan = planner._create_plan(goal, task_id)
            queue.update_goal(task_id, plan=json.dumps(plan))

            # Register subtasks under THIS task_id
            subtask_map = planner._register_subtasks(plan, task_id)

            subtask_results = {}
            specialist_subtasks = [s for s in plan["subtasks"] if s["agent"] != "writer"]

            for subtask_def in specialist_subtasks:
                agent_name = subtask_def["agent"]
                if agent_name not in planner._agents:
                    continue
                subtask_obj = subtask_map[agent_name]
                result = planner._agents[agent_name].run(subtask_obj)
                subtask_results[agent_name] = result

            writer_subtask = subtask_map.get("writer")
            if writer_subtask:
                report = planner._agents["writer"].run(writer_subtask)
                subtask_results["writer"] = report

            # Critic
            critic_subtask = planner._make_critic_subtask(goal, task_id, 0)
            # Save critic subtask to DB so update_subtask finds it
            queue._subtasks[critic_subtask.subtask_id] = critic_subtask
            queue._save_subtask(critic_subtask)
            critique = planner._agents["critic"].run(critic_subtask)

            try:
                result_data = json.loads(critique)
                approved = result_data.get("approved", False)
            except Exception:
                approved = True

            queue.update_goal(
                task_id,
                status=TaskStatus.COMPLETED,
                approved=approved,
            )

        except Exception as e:
            from src.core.task_queue import TaskStatus
            get_queue().update_goal(task_id, status=TaskStatus.FAILED)
            print(f"[dashboard] Pipeline failed: {e}")
        finally:
            st.session_state["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return task_id


def get_task_status(task_id: str):
    return get_queue().get_goal(task_id)


def render_activity_feed(task_id: str, limit: int = 80):
    queue = get_queue()
    logs = list(reversed(queue.get_agent_logs(task_id, limit=limit)))

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
        display_content = content[:300] + "..." if len(content) > 300 else content

        st.markdown(f"""
        <div class="agent-card {css_class}">
            <span style="color:{color}; font-weight:600;">{agent_emoji} {agent.upper()}</span>
            <span style="margin-left:8px; font-size:0.75rem; color:#6B7280; text-transform:uppercase;">
                {emoji} {log_type}
            </span>
            <span style="float:right; color:#9CA3AF; font-size:0.75rem;">{timestamp}</span>
            <br><span style="color:#374151;">{display_content}</span>
        </div>
        """, unsafe_allow_html=True)


def render_critic_scores(critique_json: str):
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
        st.markdown(f"""
        <div class="score-box">
            <div class="score-number">{overall:.2f}</div>
            <div style="color:#6B7280; font-size:0.85rem;">Overall Score</div>
            <br>{badge}
        </div>
        """, unsafe_allow_html=True)

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
    charts = get_charts_for_task(task_id)
    if not charts:
        st.info("No charts were generated for this task.")
        return
    for chart in charts:
        st.markdown(f"**{chart['title']}**")
        html_path = chart.get("html_path", "")
        if html_path and os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=520, scrolling=False)
        else:
            st.warning(f"Chart file not found: {html_path}")
        st.caption(chart.get("description", ""))


def render_history_sidebar():
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
            "completed": "✅", "failed": "❌",
            "in_progress": "⏳", "pending": "⏸️",
            "needs_revision": "🔄",
        }.get(task.status, "❓")
        label = f"{status_emoji} {task.goal[:38]}..."
        if st.sidebar.button(label, key=task.task_id, use_container_width=True):
            st.session_state["task_id"] = task.task_id
            st.session_state["running"] = False
            st.rerun()
        st.sidebar.caption(f"Score: {score_str} | {task.created_at[:10]}")


def main():
    for key, default in [("task_id", None), ("running", False), ("start_time", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    with st.sidebar:
        st.markdown("## 🤖 AgentForge")
        st.caption("Multi-Agent Collaborative Task System")
        st.divider()
        st.markdown("### Agent Team")
        for name, emoji in AGENT_EMOJIS.items():
            color = AGENT_COLORS[name]
            st.markdown(f'<span style="color:{color};">{emoji} **{name.capitalize()}**</span>', unsafe_allow_html=True)
        st.divider()
        st.markdown("### Recent Tasks")
        render_history_sidebar()
        st.divider()
        st.markdown("### Settings")
        st.caption(f"Model: `{settings.primary_model}`")
        st.caption(f"Pass threshold: `{settings.critic_pass_threshold}`")

    st.markdown('<div class="main-header">🤖 AgentForge</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Agent Collaborative Task System — Powered by Groq + Llama 3.3 70B</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown("### 🎯 Enter Your Goal")
        example_goals = [
            "Select an example...",
            "Research the top 5 countries by solar energy capacity and write a detailed summary",
            "Find the latest developments in large language models and summarise key trends",
            "Compare renewable energy adoption between India and China with data and analysis",
            "Research quantum computing breakthroughs in 2024 and write a technical summary",
        ]
        selected = st.selectbox("Quick examples:", example_goals)
        goal_text = st.text_area(
            "Or type your own goal:",
            value=selected if selected != example_goals[0] else "",
            height=100,
            placeholder="e.g. Research solar energy adoption globally and write a 400-word analyst note...",
        )
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            run_button = st.button("🚀 Run AgentForge", type="primary",
                                   disabled=st.session_state["running"],
                                   use_container_width=True)
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
            run_pipeline(goal_text.strip())
            st.rerun()
        elif run_button:
            st.warning("Please enter a goal first.")

    st.divider()
    task_id = st.session_state.get("task_id")

    if not task_id:
        st.markdown("### 👆 Enter a goal above to get started")
        with st.expander("What can AgentForge do?"):
            st.markdown("""
            1. **🧠 Planner** — Decomposes your goal into subtasks
            2. **🔍 Researcher** — Searches the web, reads articles, finds papers
            3. **💻 Coder** — Writes and executes Python code
            4. **📊 Analyst** — Runs statistical analysis, creates charts
            5. **✍️ Writer** — Assembles all outputs into a structured report
            6. **🔎 Critic** — Reviews quality and requests revisions if needed
            """)
        return

    task = get_task_status(task_id)
    if not task:
        st.error(f"Task {task_id} not found")
        return

    status_colors = {"completed": "🟢", "failed": "🔴", "in_progress": "🟡",
                     "pending": "⚪", "needs_revision": "🟠"}
    st.markdown(f"**Task:** `{task_id}` | **Status:** {status_colors.get(task.status,'❓')} `{task.status}`")
    st.caption(f"Goal: {task.goal}")

    tab1, tab2, tab3, tab4 = st.tabs(["⚡ Live Activity", "📄 Final Report", "🔎 Critic Review", "📊 Charts"])

    with tab1:
        st.markdown("### ⚡ Agent Activity Feed")
        if st.session_state["running"]:
            st.info("🔄 Agents are working... auto-refreshing every 3 seconds")
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
                            f'<div style="text-align:center;color:{color};font-size:1.5rem;">{emoji_agent}</div>'
                            f'<div style="text-align:center;font-size:0.75rem;color:#6B7280;">{agent_name}</div>',
                            unsafe_allow_html=True)
            except Exception:
                pass
        st.markdown("---")
        render_activity_feed(task_id, limit=80)
        if st.session_state["running"]:
            time.sleep(3)
            fresh_task = get_task_status(task_id)
            if fresh_task and fresh_task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                st.session_state["running"] = False
            st.rerun()

    with tab2:
        st.markdown("### 📄 Final Report")
        if task.final_report:
            st.markdown(task.final_report)
            st.download_button("⬇️ Download Report (Markdown)", data=task.final_report,
                               file_name=f"agentforge_report_{task_id}.md", mime="text/markdown")
        elif task.status == "in_progress":
            st.info("⏳ Report is being written...")
        elif task.status == "failed":
            st.error("❌ Pipeline failed. Check the Activity Feed for details.")
        else:
            st.info("Report will appear here once the Writer agent completes.")

    with tab3:
        st.markdown("### 🔎 Critic Quality Review")
        memory = get_memory()
        critique_chunks = memory.retrieve(query="quality scores approved",
                                          task_id=task_id, agent_name="critic", k=1)
        if critique_chunks:
            render_critic_scores(critique_chunks[0]["content"])
            if task.critic_feedback:
                st.markdown("---")
                st.markdown("**Revision Instructions:**")
                st.info(task.critic_feedback)
        elif task.critic_score:
            st.metric("Critic Score", f"{task.critic_score:.2f}")
            st.success("Approved" if task.approved else "Not approved")
        else:
            st.info("⏳ Critic review will appear after the Writer completes.")

    with tab4:
        st.markdown("### 📊 Generated Charts")
        render_charts(task_id)

    if st.session_state["running"]:
        fresh = get_task_status(task_id)
        if fresh and fresh.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.NEEDS_REVISION]:
            st.session_state["running"] = False
            st.rerun()


if __name__ == "__main__":
    main()
