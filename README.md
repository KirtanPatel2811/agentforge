# 🤖 AgentForge — Multi-Agent Collaborative Task System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-F55036?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Memory-orange?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)
![Tests](https://img.shields.io/badge/Tests-85%2F85_Passing-brightgreen?style=for-the-badge)

**A production-grade multi-agent AI system where 6 specialist agents collaborate to complete complex research, analysis, and coding tasks from a single natural language goal.**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [API](#-api-reference) • [Design](#-design-decisions) • [Tests](#-testing)

</div>

---

## 🎯 What It Does

Give AgentForge one natural language goal:

> *"Research the top 5 countries by solar energy capacity, analyse the data, write Python visualisation code, and produce a structured report with citations."*

AgentForge autonomously:

1. **🧠 Planner** — Parses the goal, builds an execution plan, assigns subtasks to specialist agents
2. **🔍 Researcher** — Searches the web (DuckDuckGo + Serper fallback), reads ArXiv papers, extracts full page content
3. **💻 Coder** — Writes Python code, executes it safely in a sandbox, debugs errors and retries
4. **📊 Analyst** — Runs statistical analysis with pandas/numpy, generates interactive Plotly charts
5. **✍️ Writer** — Pulls all outputs from shared vector memory, assembles a structured final report
6. **🔎 Critic** — Scores on 5 quality dimensions, requests targeted revisions if score < 0.7

**Live critic score on real run: 0.86 — Approved ✅**

---

## ✨ Features

| Feature | Description |
|---|---|
| **ReAct Loop from scratch** | Reason → Act → Observe cycle built without LangChain — every line explainable |
| **6 Specialist Agents** | Each with its own tools, system prompt, and memory chunk type |
| **Shared Vector Memory** | ChromaDB — agents retrieve each other's outputs by semantic similarity |
| **Agent Message Bus** | Loose coupling — agents never import each other, all communication via bus |
| **Critic Revision Loop** | Quality gate with weighted 5-dimension scoring, targeted revision instructions |
| **Safe Code Execution** | RestrictedPython compile-time validation + subprocess timeout isolation |
| **Dual LLM Provider** | Groq primary (Llama 3.3 70B) + Gemini fallback, auto-retry with backoff |
| **Live Dashboard** | Streamlit with real-time agent activity feed, auto-refresh every 3 seconds |
| **REST API** | FastAPI with background task execution, polling endpoints |
| **Full Test Suite** | 85 tests across all phases, offline + live test separation |

---

## 🏗️ Architecture

```
User Goal (natural language)
        │
        ▼
┌─────────────────────────────────────┐
│           PLANNER AGENT             │
│  Parses goal → Execution plan       │
│  Assigns subtasks → Tracks progress │
└──────────────┬──────────────────────┘
               │ Task assignments (SQLite)
       ┌───────┼───────────┐
       ▼       ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────────┐
│RESEARCHER│ │  CODER   │ │   ANALYST    │
│          │ │          │ │              │
│DuckDuckGo│ │Sandboxed │ │Pandas+NumPy  │
│ArXiv API │ │Python    │ │Plotly Charts │
│URL Reader│ │executor  │ │Stats Summary │
└────┬─────┘ └────┬─────┘ └──────┬───────┘
     └────────────┼───────────────┘
                  │ Results → ChromaDB (shared vector memory)
                  ▼
         ┌─────────────────┐
         │  WRITER AGENT   │
         │ Semantic memory │
         │ retrieval →     │
         │ Structured      │
         │ report          │
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │  CRITIC AGENT   │──── Score < 0.7 ──► Revision loop
         │ 5-dim scoring   │
         │ Targeted        │
         │ feedback        │
         └────────┬────────┘
                  │ Approved
                  ▼
         Final Report + Charts
         (Streamlit Dashboard)
```

### ReAct Loop (built from scratch)

```
┌─────────────────────────────────────────────┐
│              REACT LOOP                      │
│                                              │
│  THOUGHT: "I need to find solar data"        │
│      │                                       │
│      ▼                                       │
│  ACTION: web_search                          │
│  ACTION_INPUT: {"query": "solar 2024"}       │
│      │                                       │
│      ▼                                       │
│  [Tool executes → real web search]           │
│      │                                       │
│      ▼                                       │
│  OBSERVATION: [5 results with URLs]          │
│      │                                       │
│      ▼                                       │
│  THOUGHT: "Wikipedia looks comprehensive"    │
│      │                                       │
│      ▼                                       │
│  ACTION: read_url                            │
│  ACTION_INPUT: {"url": "wikipedia..."}       │
│      │                                       │
│      ▼                                       │
│  OBSERVATION: [full article text]            │
│      │                                       │
│      ▼                                       │
│  THOUGHT: "I have enough. Writing answer."   │
│      │                                       │
│      ▼                                       │
│  FINAL_ANSWER: [structured research output] │
└─────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10
- Conda (recommended)
- Free API keys (no credit card needed)

### 1. Get Free API Keys

| Service | URL | What For |
|---|---|---|
| **Groq** | [console.groq.com](https://console.groq.com) | Primary LLM — Llama 3.3 70B, 30 RPM free |
| **Google AI Studio** | [aistudio.google.com](https://aistudio.google.com) | Fallback LLM — Gemini 2.5 Flash |
| **Serper.dev** | [serper.dev](https://serper.dev) | Backup web search — 100 free/month |

### 2. Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/agentforge.git
cd agentforge

# Create conda environment
conda create -n agentforge python=3.10 -y
conda activate agentforge

# Install dependencies
pip install -r requirements.txt

# Configure API keys
copy .env.example .env
# Edit .env and paste your API keys
```

### 3. Run

```bash
# Option A — Command line (quickest)
python run_pipeline.py

# Option B — Streamlit Dashboard (recommended)
python -m streamlit run src/app/streamlit_app.py
# Opens at http://localhost:8501

# Option C — REST API
uvicorn src.api.main:app --reload --port 8000
# Swagger UI at http://localhost:8000/docs
```

### 4. Example Goals

```
"Research the top 5 countries by solar energy capacity and write a detailed summary with statistics"

"Find the latest developments in large language model agents and summarise key trends with citations"

"Compare renewable energy adoption in India vs China: find data, analyse trends, create charts"

"Research quantum computing breakthroughs in 2024 and write a technical summary for a CS audience"

"Find top ArXiv papers on transformer attention mechanisms and summarise their contributions"
```

---

## 📁 Project Structure

```
agentforge/
├── src/
│   ├── core/
│   │   ├── llm_client.py      # Groq + Gemini unified client, auto-fallback, JSON parsing
│   │   ├── base_agent.py      # ReAct loop — Reason→Act→Observe, tool dispatch, memory
│   │   ├── memory.py          # ChromaDB vector store, semantic retrieval, task namespacing
│   │   ├── message_bus.py     # Agent inbox system, message routing, full history log
│   │   └── task_queue.py      # SQLite task lifecycle, dependency ordering, agent logs
│   ├── agents/
│   │   ├── planner.py         # Orchestrator — plan generation, agent dispatch, revision loop
│   │   ├── researcher.py      # Web search + ArXiv + URL reader, structured findings
│   │   ├── coder.py           # Write + execute code, error recovery loop
│   │   ├── analyst.py         # Pandas stats + Plotly charts, data narrative
│   │   ├── writer.py          # Memory retrieval, report assembly (single-shot LLM)
│   │   └── critic.py          # 5-dimension weighted scoring, revision targeting
│   ├── tools/
│   │   ├── web_search.py      # DuckDuckGo primary, Serper fallback
│   │   ├── arxiv_search.py    # ArXiv paper search, abstract preview
│   │   ├── url_reader.py      # 3-stage content extraction, noise removal
│   │   ├── code_executor.py   # RestrictedPython + subprocess sandbox
│   │   └── chart_generator.py # Plotly bar/line/pie/scatter/heatmap, HTML + PNG output
│   ├── api/
│   │   └── main.py            # FastAPI — /run, /status, /result, /logs, /history
│   └── app/
│       └── streamlit_app.py   # Live dashboard, activity feed, critic scores, charts
├── tests/
│   ├── test_phase1_core.py    # 24 tests — config, bus, queue, memory, LLM parsing
│   ├── test_phase2_tools.py   # 21 tests — code executor, chart generator, URL reader
│   ├── test_phase3_agents.py  # 22 tests — agent instantiation, critic logic, planner
│   └── test_phase4_api.py     # 18 tests — all FastAPI endpoints
├── data/
│   ├── cache/                 # SQLite task database (agentforge.db)
│   ├── chroma/                # ChromaDB vector store (persists between runs)
│   └── outputs/               # Generated reports (.md) and charts (.html, .png)
├── notebooks/
│   └── 01_agent_exploration.ipynb
├── config.py                  # All settings, loaded from .env, singleton
├── run_pipeline.py            # CLI runner for quick testing
└── requirements.txt
```

---

## 🔧 Configuration

All settings in `.env` (copy from `.env.example`):

```env
# Required
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_key_here

# Optional (enhances web search reliability)
SERPER_API_KEY=your_serper_key_here
```

Key settings in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `primary_model` | `llama-3.3-70b-versatile` | Groq model |
| `fallback_model` | `gemini-2.5-flash` | Google AI Studio model |
| `max_react_iterations` | `8` | Max ReAct loop iterations per agent |
| `max_revision_rounds` | `2` | Max Critic revision cycles |
| `critic_pass_threshold` | `0.7` | Minimum score to approve report |
| `code_timeout_seconds` | `30` | Sandbox execution timeout |

---

## 🛠️ The 5 Tools

### web_search
DuckDuckGo primary search with automatic Serper.dev fallback. Returns structured `{title, url, snippet, source}` dicts. Also supports `news_search()` for recent events.

### arxiv_search
ArXiv API search returning `{title, authors, abstract, pdf_url, published, categories}`. Supports relevance and date sorting. Use `arxiv_get_paper(id)` for specific papers.

### url_reader
Three-stage content extraction: semantic HTML5 tags → class-name heuristics → paragraph fallback. Removes nav/footer/ads. Caps at 8000 chars. Handles PDF detection gracefully.

### code_executor
Two-layer safety: RestrictedPython compile-time validation + subprocess with timeout. Captures stdout + stderr. Returns `{success, stdout, stderr, error, execution_time}`. Available packages: pandas, numpy, scipy, plotly, math, statistics.

### chart_generator
Plotly charts (bar, line, scatter, pie, histogram, heatmap) saved as interactive HTML + static PNG. JSON registry at `data/outputs/chart_registry.json` lets Writer and dashboard discover all charts.

---

## 🔎 The Critic Agent

The Critic evaluates reports on 5 weighted dimensions:

| Dimension | Weight | What It Checks |
|---|---|---|
| Completeness | 25% | Did it address all parts of the goal? |
| Factual Grounding | 25% | Are claims backed by specific data and sources? |
| Code Correctness | 20% | Is code syntactically valid, explained, output shown? |
| Source Citation | 15% | Are URLs provided? Are sources named? |
| Readability | 15% | Clear structure, logical flow, professional tone? |

**Overall score = weighted average. Approved if ≥ 0.7.**

If rejected, the Critic sends targeted revision instructions to specific agents:
- Low `factual_grounding` → Researcher re-runs with feedback
- Low `code_correctness` → Coder re-runs with feedback
- Low `readability` → Writer re-runs with feedback

Maximum 2 revision rounds (configurable in `config.py`).

---

## 🌐 API Reference

Run the API: `uvicorn src.api.main:app --reload --port 8000`

Interactive docs: `http://localhost:8000/docs`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/run` | Submit a goal → returns `task_id` immediately |
| `GET` | `/status/{task_id}` | Check task status |
| `GET` | `/result/{task_id}` | Get full result (report + scores + charts) |
| `GET` | `/logs/{task_id}` | Get agent activity logs |
| `GET` | `/history` | List recent tasks |
| `DELETE` | `/task/{task_id}` | Clear task from memory |

### Example Usage

```python
import requests

# Submit a goal
resp = requests.post("http://localhost:8000/run", json={
    "goal": "Research top 5 solar energy countries and write a summary"
})
task_id = resp.json()["task_id"]

# Poll until complete
import time
while True:
    status = requests.get(f"http://localhost:8000/status/{task_id}").json()
    print(f"Status: {status['status']}")
    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(5)

# Get result
result = requests.get(f"http://localhost:8000/result/{task_id}").json()
print(f"Critic score: {result['critic_score']}")
print(f"Approved: {result['approved']}")
print(result["report"][:500])
```

---

## 🧪 Testing

```bash
# All offline tests (no API keys or internet needed)
python -m pytest tests/ -v -m "not live"
# Expected: 85/85 passing

# Phase-specific
python -m pytest tests/test_phase1_core.py -v    # Core infrastructure
python -m pytest tests/test_phase2_tools.py -v -m "not live"  # Tools
python -m pytest tests/test_phase3_agents.py -v -m "not live" # Agents
python -m pytest tests/test_phase4_api.py -v     # FastAPI endpoints

# Live tests (needs API keys + internet, ~2-3 minutes)
python -m pytest tests/ -v -m live
```

### Test Coverage

| Phase | Module | Tests | Type |
|---|---|---|---|
| Phase 1 | Config, MessageBus, TaskQueue, Memory, LLM parsing | 24 | Offline |
| Phase 2 | CodeExecutor, ChartGenerator, URLReader | 21 | Offline |
| Phase 3 | Agent instantiation, Critic logic, Planner logic | 22 | Offline + Live |
| Phase 4 | All FastAPI endpoints | 18 | Offline |

---

## 💡 Key Design Decisions

### Why build ReAct from scratch instead of LangChain?
LangChain abstracts away exactly what you need to understand for interviews and production debugging. Building the loop yourself means understanding how tool calling works, how to parse LLM output into structured actions, and how to handle errors gracefully. LangChain is a great tool — but building without it first means you can explain every line.

### Why ChromaDB for shared memory?
Key-value lookup requires knowing the exact key. Vector search lets the Writer ask "find all research about solar energy" and retrieve semantically relevant chunks from Researcher + Analyst outputs without knowing their keys in advance. This is how production RAG pipelines work.

### Why a Message Bus?
Without the bus, the Planner would directly import and call `researcher.run()`. This creates tight coupling — the Planner knows the Researcher's interface. The bus decouples senders from receivers. Any agent can be replaced or tested independently.

### Why subprocess for code execution instead of exec()?
`exec()` in the same process shares memory. A malicious or buggy script could access anything in the parent process's scope. `subprocess.run()` creates a truly separate process with its own memory space — if it crashes or times out, only the subprocess dies.

### Why a Critic agent?
Real agentic systems fail silently — the LLM produces plausible-sounding but wrong output. The Critic adds a quality gate with specific, measurable dimensions. If `factual_grounding` is low, the Researcher gets sent back — not the Writer. This targeted revision is what makes the system self-improving rather than just self-repeating.

---

## 🗺️ Roadmap

- [x] Phase 1 — Core infrastructure (LLM client, memory, bus, task queue, ReAct base)
- [x] Phase 2 — 5 specialist tools (web search, ArXiv, URL reader, code executor, charts)
- [x] Phase 3 — 6 specialist agents (Planner, Researcher, Coder, Analyst, Writer, Critic)
- [x] Phase 4 — Dashboard + API (Streamlit live feed, FastAPI endpoints)
- [ ] Phase 5 — Parallel agent execution with ThreadPoolExecutor
- [ ] Phase 6 — Docker containerisation
- [ ] Phase 7 — LangGraph integration (optional alternative orchestration)

---

## 🛠️ Built With

| Layer | Technology | Why |
|---|---|---|
| LLM Primary | [Groq](https://groq.com) — Llama 3.3 70B | Free, 300+ tok/s, no credit card |
| LLM Fallback | [Google AI Studio](https://aistudio.google.com) — Gemini 2.5 Flash | Free fallback, 1500 req/day |
| Agent Framework | Custom ReAct loop | Built from scratch for learning and explainability |
| Vector Memory | [ChromaDB](https://www.trychroma.com) | Local, persistent, semantic search |
| Web Search | [DuckDuckGo](https://pypi.org/project/duckduckgo-search/) + [Serper](https://serper.dev) | No API key primary, paid fallback |
| Paper Search | [ArXiv API](https://pypi.org/project/arxiv/) | Free, unlimited |
| Code Safety | [RestrictedPython](https://restrictedpython.readthedocs.io) | Compile-time validation |
| Data Analysis | Pandas, NumPy, SciPy | Industry standard |
| Visualisation | [Plotly](https://plotly.com) | Interactive HTML charts |
| Dashboard | [Streamlit](https://streamlit.io) | Rapid AI app development |
| API | [FastAPI](https://fastapi.tiangolo.com) | Modern async Python API |
| Storage | SQLite | Zero-dependency task persistence |
| Testing | pytest | 85 tests, offline/live separation |

---

## 👨‍💻 Author

**Kittu** — B.Tech CSE, CSPIT CHARUSAT

Portfolio projects:
- **MediScan** — Medical document OCR + NLP classification
- **MarketPulse** — FinBERT sentiment + LSTM price forecasting + FastAPI
- **AgentForge** — Multi-agent AI system (this project)

---

## 📄 License

MIT License — feel free to use, modify, and build on this project.

---

<div align="center">

**If this project helped you, please ⭐ the repo!**

Built with 🤖 Groq + Llama 3.3 70B | ChromaDB | FastAPI | Streamlit

</div>
