"""
setup_phase1.py — Run this once to create all Phase 1 files.
Place this file in your agentforge root folder and run:
    python setup_phase1.py
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def write(path, content):
    full = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {path}")


# ── __init__.py files ──────────────────────────────────────────────────────────
for p in [
    "src/__init__.py",
    "src/core/__init__.py",
    "src/agents/__init__.py",
    "src/tools/__init__.py",
    "src/api/__init__.py",
    "src/app/__init__.py",
    "tests/__init__.py",
]:
    write(p, "")

# ── .gitignore ─────────────────────────────────────────────────────────────────
write(
    ".gitignore",
    """\
# Environment
.env
*.env

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.pytest_cache/
.mypy_cache/

# Data (generated at runtime)
data/cache/
data/chroma/
data/outputs/

# Conda / venv
.conda/
venv/
env/

# IDE
.vscode/
.idea/

# Notebooks
notebooks/.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Logs
*.log
""",
)

# ── .env.example ───────────────────────────────────────────────────────────────
write(
    ".env.example",
    """\
# Copy this file to .env and fill in your keys.
# NEVER commit .env to git.

# Get free key at: console.groq.com (no credit card)
GROQ_API_KEY=your_groq_api_key_here

# Get free key at: aistudio.google.com (no credit card)
GOOGLE_API_KEY=your_google_api_key_here

# Get 100 free searches/month at: serper.dev (optional)
SERPER_API_KEY=your_serper_api_key_here
""",
)

# ── requirements.txt ───────────────────────────────────────────────────────────
write(
    "requirements.txt",
    """\
# LLM Clients
groq==0.9.0
google-generativeai==0.7.2
pydantic-settings==2.3.4

# Agent Framework & Memory
chromadb==0.5.3

# Tools
duckduckgo-search==6.2.6
arxiv==2.1.0
requests==2.32.3
beautifulsoup4==4.12.3
lxml==5.2.2
RestrictedPython==7.1

# Data & Visualisation
pandas==2.2.2
numpy==1.26.4
scipy==1.13.1
plotly==5.22.0

# API & Dashboard
fastapi==0.111.0
uvicorn[standard]==0.30.1
streamlit==1.36.0
pydantic==2.7.4

# Utilities
python-dotenv==1.0.1
httpx==0.27.0
tenacity==8.3.0
loguru==0.7.2
rich==13.7.1

# Testing
pytest==8.2.2
pytest-asyncio==0.23.7
""",
)

# ── config.py ──────────────────────────────────────────────────────────────────
write(
    "config.py",
    '''\
"""
config.py — AgentForge Central Configuration
All project-wide settings live here. Every module imports from config.py.
Design: pydantic BaseSettings auto-reads .env files and gives type validation.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUTS_DIR = DATA_DIR / "outputs"
CHROMA_DIR = DATA_DIR / "chroma"

for _dir in [CACHE_DIR, OUTPUTS_DIR, CHROMA_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    groq_api_key: str = Field(default="")
    google_api_key: str = Field(default="")
    serper_api_key: str = Field(default="")

    # LLM Settings
    primary_model: str = "llama-3.3-70b-versatile"
    fallback_model: str = "gemini-2.5-flash"
    max_tokens: int = 4096
    temperature: float = 0.1

    # Retry / Fallback
    max_retries: int = 3
    retry_delay: float = 2.0
    request_timeout: int = 60

    # Agent Settings
    max_react_iterations: int = 8
    max_revision_rounds: int = 2
    agent_memory_k: int = 5

    # Code Executor
    code_timeout_seconds: int = 30
    max_code_output_chars: int = 5000

    # Web Search
    max_search_results: int = 5
    max_arxiv_results: int = 5

    # ChromaDB
    chroma_collection: str = "agentforge_memory"
    chroma_path: str = str(CHROMA_DIR)

    # SQLite
    db_path: str = str(CACHE_DIR / "agentforge.db")

    # Critic Thresholds
    critic_pass_threshold: float = 0.7
    critic_dimensions: list[str] = [
        "completeness",
        "factual_grounding",
        "code_correctness",
        "source_citation",
        "readability",
    ]

    # Streamlit & FastAPI
    streamlit_port: int = 8501
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_debug: bool = True

    # Logging
    log_level: str = "DEBUG"
    log_file: str = str(CACHE_DIR / "agentforge.log")


settings = Settings()


class AgentName:
    PLANNER    = "planner"
    RESEARCHER = "researcher"
    CODER      = "coder"
    ANALYST    = "analyst"
    WRITER     = "writer"
    CRITIC     = "critic"
    ALL = [PLANNER, RESEARCHER, CODER, ANALYST, WRITER, CRITIC]


class TaskStatus:
    PENDING          = "pending"
    IN_PROGRESS      = "in_progress"
    COMPLETED        = "completed"
    FAILED           = "failed"
    NEEDS_REVISION   = "needs_revision"


if __name__ == "__main__":
    from rich import print as rprint
    rprint("[bold green]AgentForge Configuration[/bold green]")
    rprint(f"  Primary model : {settings.primary_model}")
    rprint(f"  Fallback model: {settings.fallback_model}")
    rprint(f"  Groq key set  : {\'✓\' if settings.groq_api_key else \'✗ MISSING\'}")
    rprint(f"  Google key set: {\'✓\' if settings.google_api_key else \'✗ MISSING\'}")
    rprint(f"  Data dir      : {DATA_DIR}")
''',
)

# ── src/core/llm_client.py ────────────────────────────────────────────────────
write(
    "src/core/llm_client.py",
    '''\
"""
src/core/llm_client.py — Unified LLM Client
All agent LLM calls go through this single interface.

Design decisions:
1. UNIFIED INTERFACE: Hides provider differences — agents just call llm.complete().
2. AUTOMATIC FALLBACK: Groq fails → transparently retry with Gemini.
3. RETRY WITH BACKOFF: Transient errors retried before triggering fallback.
4. JSON PARSING: parse_json() strips markdown fences and validates JSON.
"""

import json
import re
import time
from typing import Optional
from loguru import logger
from config import settings


class LLMError(Exception):
    pass

class LLMParseError(Exception):
    pass


_groq_client = None
_gemini_client = None


def _get_groq():
    global _groq_client
    if _groq_client is None:
        if not settings.groq_api_key:
            raise LLMError("GROQ_API_KEY not set in .env")
        from groq import Groq
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        if not settings.google_api_key:
            raise LLMError("GOOGLE_API_KEY not set in .env")
        import google.generativeai as genai
        genai.configure(api_key=settings.google_api_key)
        _gemini_client = genai.GenerativeModel(settings.fallback_model)
    return _gemini_client


def _groq_complete(messages, system_prompt=None, max_tokens=None, temperature=None):
    client = _get_groq()
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)
    response = client.chat.completions.create(
        model=settings.primary_model,
        messages=full_messages,
        max_tokens=max_tokens or settings.max_tokens,
        temperature=temperature if temperature is not None else settings.temperature,
        timeout=settings.request_timeout,
    )
    return response.choices[0].message.content


def _gemini_complete(messages, system_prompt=None, max_tokens=None, temperature=None):
    model = _get_gemini()
    history = []
    last_user_message = None
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        if msg == messages[-1] and role == "user":
            last_user_message = msg["content"]
        else:
            history.append({"role": role, "parts": [msg["content"]]})
    if last_user_message is None:
        last_user_message = messages[-1]["content"]
    if system_prompt and not history:
        last_user_message = f"{system_prompt}\\n\\n{last_user_message}"
    elif system_prompt:
        history[0]["parts"][0] = f"{system_prompt}\\n\\n{history[0][\'parts\'][0]}"
    chat = model.start_chat(history=history)
    response = chat.send_message(last_user_message)
    return response.text


class LLMClient:
    """
    Unified LLM interface for all agents.
    Usage:
        client = LLMClient(system_prompt="You are a research agent...")
        response = client.complete([{"role": "user", "content": "Find X"}])
        data = client.complete_json([{"role": "user", "content": "Return JSON..."}])
    """

    def __init__(self, agent_name: str = "agent", system_prompt: Optional[str] = None):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self._call_count = 0
        self._groq_failures = 0

    def complete(self, messages, max_tokens=None, temperature=None, force_fallback=False):
        self._call_count += 1
        kwargs = dict(messages=messages, system_prompt=self.system_prompt,
                      max_tokens=max_tokens, temperature=temperature)

        if not force_fallback and settings.groq_api_key:
            for attempt in range(settings.max_retries):
                try:
                    logger.debug(f"[{self.agent_name}] Groq call #{self._call_count} attempt {attempt+1}")
                    result = _groq_complete(**kwargs)
                    logger.debug(f"[{self.agent_name}] Groq OK — {len(result)} chars")
                    return result
                except Exception as e:
                    err_str = str(e).lower()
                    logger.warning(f"[{self.agent_name}] Groq attempt {attempt+1} failed: {e}")
                    if "rate_limit" in err_str or "429" in err_str:
                        wait = settings.retry_delay * (2 ** attempt)
                        logger.info(f"[{self.agent_name}] Rate limited, waiting {wait}s")
                        time.sleep(wait)
                    elif attempt < settings.max_retries - 1:
                        time.sleep(settings.retry_delay)
                    else:
                        logger.warning(f"[{self.agent_name}] Groq exhausted — falling back to Gemini")
                        self._groq_failures += 1

        if settings.google_api_key:
            for attempt in range(settings.max_retries):
                try:
                    logger.debug(f"[{self.agent_name}] Gemini fallback attempt {attempt+1}")
                    result = _gemini_complete(**kwargs)
                    logger.debug(f"[{self.agent_name}] Gemini OK — {len(result)} chars")
                    return result
                except Exception as e:
                    logger.warning(f"[{self.agent_name}] Gemini attempt {attempt+1} failed: {e}")
                    if attempt < settings.max_retries - 1:
                        time.sleep(settings.retry_delay)

        raise LLMError(
            f"[{self.agent_name}] All LLM providers failed. "
            "Check your API keys in .env and your network connection."
        )

    def complete_json(self, messages, max_tokens=None, temperature=None):
        raw = self.complete(messages, max_tokens=max_tokens, temperature=temperature)
        return parse_json(raw, agent_name=self.agent_name)

    def simple(self, prompt: str, **kwargs) -> str:
        return self.complete([{"role": "user", "content": prompt}], **kwargs)

    def simple_json(self, prompt: str, **kwargs) -> dict:
        return self.complete_json([{"role": "user", "content": prompt}], **kwargs)

    @property
    def stats(self):
        return {"agent": self.agent_name, "total_calls": self._call_count,
                "groq_failures": self._groq_failures}


def parse_json(text: str, agent_name: str = "agent") -> dict:
    """
    Extract and parse JSON from LLM output.
    Handles markdown fences, preambles, and raw JSON.
    """
    if not text or not text.strip():
        raise LLMParseError(f"[{agent_name}] LLM returned empty response")
    stripped = re.sub(r"```(?:json)?\\s*", "", text).replace("```", "").strip()
    json_match = re.search(r"(\\{[\\s\\S]*\\}|\\[[\\s\\S]*\\])", stripped)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as e:
        raise LLMParseError(
            f"[{agent_name}] Could not parse JSON.\\n"
            f"Error: {e}\\nRaw output: {text[:500]}"
        )


llm = LLMClient(agent_name="default")


if __name__ == "__main__":
    from rich import print as rprint
    rprint("[bold cyan]Testing LLM Client...[/bold cyan]")
    client = LLMClient(agent_name="test", system_prompt="You are a helpful assistant. Be brief.")
    response = client.simple("Say exactly: \'AgentForge LLM client works!\'")
    rprint(f"Response: {response}")
    json_response = client.simple_json(
        "Return a JSON object with keys \'status\' (value: \'ok\') and \'agent\' (value: \'test\'). "
        "Return ONLY the JSON object, no other text."
    )
    rprint(f"Parsed JSON: {json_response}")
    rprint(f"Stats: {client.stats}")
    rprint("[bold green]✓ LLM Client test passed![/bold green]")
''',
)

# ── src/core/memory.py ────────────────────────────────────────────────────────
write(
    "src/core/memory.py",
    '''\
"""
src/core/memory.py — Shared Vector Memory Store
All agents read/write to ChromaDB through this module.

Design decisions:
1. VECTOR MEMORY: Semantic search lets Writer find relevant chunks
   from any agent without knowing exact keys — like production RAG.
2. NAMESPACING: Every chunk tagged with agent_name, task_id, chunk_type.
3. LOCAL EMBEDDINGS: ChromaDB uses all-MiniLM-L6-v2 (~80MB, free, offline).
4. PERSISTENT: Survives restarts. Call clear_task() after each run.
"""

import uuid
from datetime import datetime
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger
from config import settings, CHROMA_DIR


class ChunkType:
    RESEARCH     = "research"
    CODE         = "code"
    ANALYSIS     = "analysis"
    REPORT_DRAFT = "report_draft"
    CRITIQUE     = "critique"
    PLAN         = "plan"
    TOOL_OUTPUT  = "tool_output"


class AgentMemory:
    """
    Shared vector memory backed by ChromaDB.
    All agents use the same instance via get_memory() singleton.
    """

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"AgentMemory initialised — {self._collection.count()} existing chunks"
        )

    def store(self, content: str, agent_name: str, task_id: str,
              chunk_type: str, metadata: Optional[dict] = None,
              chunk_id: Optional[str] = None) -> str:
        if not content or not content.strip():
            logger.warning(f"[{agent_name}] Empty content, skipping store")
            return ""
        chunk_id = chunk_id or f"{agent_name}_{task_id}_{uuid.uuid4().hex[:8]}"
        full_metadata = {
            "agent_name": agent_name, "task_id": task_id,
            "chunk_type": chunk_type,
            "timestamp": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }
        self._collection.upsert(ids=[chunk_id], documents=[content],
                                metadatas=[full_metadata])
        logger.debug(f"[memory] Stored chunk {chunk_id} ({chunk_type}, {len(content)} chars)")
        return chunk_id

    def store_many(self, chunks: list[dict]) -> list[str]:
        ids, documents, metadatas = [], [], []
        for chunk in chunks:
            content = chunk.get("content", "").strip()
            if not content:
                continue
            chunk_id = chunk.get("chunk_id") or (
                f"{chunk[\'agent_name\']}_{chunk[\'task_id\']}_{uuid.uuid4().hex[:8]}"
            )
            full_metadata = {
                "agent_name": chunk["agent_name"], "task_id": chunk["task_id"],
                "chunk_type": chunk["chunk_type"],
                "timestamp": datetime.utcnow().isoformat(),
                **(chunk.get("metadata") or {}),
            }
            ids.append(chunk_id)
            documents.append(content)
            metadatas.append(full_metadata)
        if ids:
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            logger.debug(f"[memory] Batch stored {len(ids)} chunks")
        return ids

    def retrieve(self, query: str, task_id: Optional[str] = None,
                 agent_name: Optional[str] = None, chunk_type: Optional[str] = None,
                 k: Optional[int] = None) -> list[dict]:
        k = k or settings.agent_memory_k
        conditions = []
        if task_id:
            conditions.append({"task_id": {"$eq": task_id}})
        if agent_name:
            conditions.append({"agent_name": {"$eq": agent_name}})
        if chunk_type:
            conditions.append({"chunk_type": {"$eq": chunk_type}})

        where = {}
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        total = self._collection.count()
        if total == 0:
            return []

        query_kwargs = {
            "query_texts": [query],
            "n_results": min(k, total),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)
        chunks = []
        if results["ids"] and results["ids"][0]:
            for i, cid in enumerate(results["ids"][0]):
                chunks.append({
                    "id": cid,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                })
        logger.debug(f"[memory] Retrieved {len(chunks)} chunks for query \'{query[:50]}\'")
        return chunks

    def get_task_summary(self, task_id: str) -> dict[str, list[dict]]:
        results = self._collection.get(
            where={"task_id": {"$eq": task_id}},
            include=["documents", "metadatas"],
        )
        grouped: dict[str, list[dict]] = {}
        if results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                agent = results["metadatas"][i].get("agent_name", "unknown")
                if agent not in grouped:
                    grouped[agent] = []
                grouped[agent].append({
                    "id": chunk_id,
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i],
                })
        return grouped

    def clear_task(self, task_id: str) -> int:
        results = self._collection.get(
            where={"task_id": {"$eq": task_id}}, include=[],
        )
        ids = results.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
            logger.info(f"[memory] Cleared {len(ids)} chunks for task {task_id}")
        return len(ids)

    def clear_all(self):
        self._client.delete_collection(settings.chroma_collection)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("[memory] All memory cleared!")

    @property
    def count(self) -> int:
        return self._collection.count()


_memory_instance: Optional[AgentMemory] = None

def get_memory() -> AgentMemory:
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = AgentMemory()
    return _memory_instance


if __name__ == "__main__":
    from rich import print as rprint
    rprint("[bold cyan]Testing Agent Memory...[/bold cyan]")
    mem = get_memory()
    test_task = "test_task_001"
    id1 = mem.store("Solar energy capacity reached 1.6 TW in 2023.",
                    "researcher", test_task, ChunkType.RESEARCH)
    id2 = mem.store("China leads with 430 GW installed capacity.",
                    "researcher", test_task, ChunkType.RESEARCH)
    id3 = mem.store("import plotly.express as px; fig = px.bar(...)",
                    "coder", test_task, ChunkType.CODE)
    rprint(f"Stored 3 chunks: {id1}, {id2}, {id3}")
    results = mem.retrieve("solar energy statistics", task_id=test_task, k=3)
    for r in results:
        rprint(f"  [{r[\'metadata\'][\'agent_name\']}] {r[\'content\'][:80]}")
    deleted = mem.clear_task(test_task)
    rprint(f"Cleaned up {deleted} test chunks")
    rprint("[bold green]✓ Memory test passed![/bold green]")
''',
)

# ── src/core/message_bus.py ───────────────────────────────────────────────────
write(
    "src/core/message_bus.py",
    '''\
"""
src/core/message_bus.py — Agent-to-Agent Message Bus
All agent communication goes through here. No agent imports another agent.

Design decisions:
1. DECOUPLING: Planner posts a Message. Bus delivers it. Agents swappable.
2. IN-MEMORY QUEUE: Simple dict of deques — one inbox per agent.
3. FULL HISTORY: All messages logged for Streamlit activity feed.
4. SYNC BY DEFAULT: Phase 4 can add async without changing the interface.
"""

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from loguru import logger


class MessageType(str, Enum):
    TASK             = "task"
    RESULT           = "result"
    REVISION_REQUEST = "revision"
    STATUS_UPDATE    = "status"
    MEMORY_NOTIFY    = "memory_notify"


class MessagePriority(int, Enum):
    LOW    = 0
    NORMAL = 1
    HIGH   = 2
    URGENT = 3


@dataclass
class Message:
    sender:       str
    recipient:    str
    message_type: MessageType
    task_id:      str
    content:      dict[str, Any] = field(default_factory=dict)
    priority:     MessagePriority = MessagePriority.NORMAL
    message_id:   str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp:    str = field(default_factory=lambda: datetime.utcnow().isoformat())
    reply_to:     Optional[str] = None

    def __repr__(self):
        return (f"Message(id={self.message_id}, {self.sender}→{self.recipient}, "
                f"type={self.message_type.value}, task={self.task_id})")


class MessageBus:
    """
    Central in-memory message bus.
    Each agent has its own inbox (deque). Bus routes messages to the right inbox.
    """

    def __init__(self):
        self._inboxes: dict[str, deque[Message]] = {}
        self._history: list[Message] = []
        logger.info("MessageBus initialised")

    def _ensure_inbox(self, agent_name: str):
        if agent_name not in self._inboxes:
            self._inboxes[agent_name] = deque()

    def send(self, message: Message) -> str:
        self._ensure_inbox(message.recipient)
        self._inboxes[message.recipient].append(message)
        self._history.append(message)
        logger.debug(f"[bus] {message.sender} → {message.recipient} [{message.message_type.value}]")
        return message.message_id

    def broadcast(self, sender, recipients, message_type, task_id, content, priority=MessagePriority.NORMAL):
        ids = []
        for recipient in recipients:
            msg = Message(sender=sender, recipient=recipient, message_type=message_type,
                          task_id=task_id, content=content, priority=priority)
            ids.append(self.send(msg))
        return ids

    def receive(self, agent_name: str, message_type: Optional[MessageType] = None,
                drain: bool = True) -> list[Message]:
        self._ensure_inbox(agent_name)
        inbox = self._inboxes[agent_name]
        if not inbox:
            return []
        if drain:
            if message_type is None:
                messages = list(inbox)
                inbox.clear()
            else:
                matching, remaining = [], []
                while inbox:
                    msg = inbox.popleft()
                    (matching if msg.message_type == message_type else remaining).append(msg)
                for msg in remaining:
                    inbox.appendleft(msg)
                messages = matching
        else:
            messages = [m for m in inbox if message_type is None or m.message_type == message_type]
        logger.debug(f"[bus] {agent_name} received {len(messages)} messages")
        return messages

    def receive_one(self, agent_name: str, message_type: Optional[MessageType] = None):
        messages = self.receive(agent_name, message_type=message_type, drain=True)
        return messages[0] if messages else None

    def pending_count(self, agent_name: str) -> int:
        self._ensure_inbox(agent_name)
        return len(self._inboxes[agent_name])

    def get_history(self, task_id=None, sender=None, recipient=None):
        h = self._history
        if task_id:   h = [m for m in h if m.task_id == task_id]
        if sender:    h = [m for m in h if m.sender == sender]
        if recipient: h = [m for m in h if m.recipient == recipient]
        return h

    def clear_task(self, task_id: str):
        for inbox in self._inboxes.values():
            to_remove = [m for m in inbox if m.task_id == task_id]
            for msg in to_remove:
                inbox.remove(msg)

    def reset(self):
        self._inboxes.clear()
        self._history.clear()
        logger.info("[bus] Message bus reset")


_bus_instance: Optional[MessageBus] = None

def get_bus() -> MessageBus:
    global _bus_instance
    if _bus_instance is None:
        _bus_instance = MessageBus()
    return _bus_instance


def make_task_message(sender, recipient, task_id, description,
                      context=None, expected_output="",
                      priority=MessagePriority.NORMAL) -> Message:
    return Message(
        sender=sender, recipient=recipient, message_type=MessageType.TASK,
        task_id=task_id,
        content={"description": description, "context": context or {},
                 "expected_output": expected_output},
        priority=priority,
    )


def make_result_message(sender, recipient, task_id, output,
                        chunk_ids=None, tool_calls=None,
                        success=True, error=None) -> Message:
    return Message(
        sender=sender, recipient=recipient, message_type=MessageType.RESULT,
        task_id=task_id,
        content={"output": output, "chunk_ids": chunk_ids or [],
                 "tool_calls": tool_calls or [], "success": success, "error": error},
    )


if __name__ == "__main__":
    from rich import print as rprint
    from rich.table import Table
    rprint("[bold cyan]Testing Message Bus...[/bold cyan]")
    bus = get_bus()
    task_id = "demo_001"
    bus.send(make_task_message("planner", "researcher", task_id, "Find solar data"))
    messages = bus.receive("researcher")
    rprint(f"Researcher received: {messages[0].content[\'description\']}")
    bus.send(make_result_message("researcher", "planner", task_id, "China: 430 GW"))
    history = bus.get_history(task_id=task_id)
    table = Table(title="Message History")
    table.add_column("From"); table.add_column("To"); table.add_column("Type")
    for msg in history:
        table.add_row(msg.sender, msg.recipient, msg.message_type.value)
    rprint(table)
    rprint("[bold green]✓ Message Bus test passed![/bold green]")
''',
)

# ── src/core/task_queue.py ────────────────────────────────────────────────────
write(
    "src/core/task_queue.py",
    '''\
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
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
from loguru import logger
from config import settings, TaskStatus, AgentName


@dataclass
class SubTask:
    task_id:         str
    subtask_id:      str
    agent_name:      str
    description:     str
    expected_output: str
    context:         dict
    status:          str = TaskStatus.PENDING
    priority:        int = 1
    dependencies:    list[str] = field(default_factory=list)
    result:          Optional[str] = None
    error:           Optional[str] = None
    chunk_ids:       list[str] = field(default_factory=list)
    created_at:      str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at:      Optional[str] = None
    completed_at:    Optional[str] = None
    revision_count:  int = 0

    def to_dict(self): return asdict(self)


@dataclass
class GoalTask:
    goal:             str
    task_id:          str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    status:           str = TaskStatus.PENDING
    plan:             Optional[str] = None
    subtask_ids:      list[str] = field(default_factory=list)
    final_report:     Optional[str] = None
    critic_score:     Optional[float] = None
    critic_feedback:  Optional[str] = None
    created_at:       str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at:     Optional[str] = None
    total_llm_calls:  int = 0
    approved:         bool = False

    def to_dict(self): return asdict(self)


class TaskQueue:
    """
    Manages task lifecycle and persists to SQLite.
    All agents and the Planner interact with this.
    """

    def __init__(self, db_path: Optional[str] = None):
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
        logger.info(f"[queue] Created goal {task.task_id}: \'{goal[:60]}\'")
        return task

    def update_goal(self, task_id: str, **kwargs) -> GoalTask:
        task = self._get_or_load_goal(task_id)
        for k, v in kwargs.items():
            if hasattr(task, k): setattr(task, k, v)
        if kwargs.get("status") in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            if not task.completed_at:
                task.completed_at = datetime.utcnow().isoformat()
        self._save_goal(task)
        return task

    def get_goal(self, task_id: str) -> Optional[GoalTask]:
        return self._get_or_load_goal(task_id)

    # ── SubTask management ───────────────────────────────────────────────────

    def add_subtask(self, task_id, agent_name, description,
                    expected_output="", context=None, priority=1,
                    dependencies=None, subtask_id=None) -> SubTask:
        subtask = SubTask(
            task_id=task_id,
            subtask_id=subtask_id or f"{agent_name}_{uuid.uuid4().hex[:8]}",
            agent_name=agent_name, description=description,
            expected_output=expected_output, context=context or {},
            priority=priority, dependencies=dependencies or [],
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
        for k, v in kwargs.items():
            if hasattr(subtask, k): setattr(subtask, k, v)
        if kwargs.get("status") == TaskStatus.IN_PROGRESS and not subtask.started_at:
            subtask.started_at = datetime.utcnow().isoformat()
        if kwargs.get("status") in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            if not subtask.completed_at:
                subtask.completed_at = datetime.utcnow().isoformat()
        self._save_subtask(subtask)
        return subtask

    def complete_subtask(self, subtask_id, result, chunk_ids=None) -> SubTask:
        return self.update_subtask(subtask_id, status=TaskStatus.COMPLETED,
                                   result=result, chunk_ids=chunk_ids or [],
                                   completed_at=datetime.utcnow().isoformat())

    def fail_subtask(self, subtask_id, error) -> SubTask:
        return self.update_subtask(subtask_id, status=TaskStatus.FAILED, error=error)

    def request_revision(self, subtask_id, feedback) -> SubTask:
        subtask = self._get_or_load_subtask(subtask_id)
        return self.update_subtask(subtask_id, status=TaskStatus.NEEDS_REVISION,
                                   revision_count=subtask.revision_count + 1)

    def get_subtask(self, subtask_id) -> Optional[SubTask]:
        return self._get_or_load_subtask(subtask_id)

    def get_subtasks_for_goal(self, task_id, status_filter=None) -> list[SubTask]:
        goal = self._get_or_load_goal(task_id)
        if not goal: return []
        subtasks = []
        for sid in goal.subtask_ids:
            st = self._get_or_load_subtask(sid)
            if st and (status_filter is None or st.status == status_filter):
                subtasks.append(st)
        return sorted(subtasks, key=lambda s: s.priority, reverse=True)

    def get_ready_subtasks(self, task_id) -> list[SubTask]:
        """SubTasks that are PENDING and have all dependencies completed."""
        all_subtasks = self.get_subtasks_for_goal(task_id)
        completed_ids = {s.subtask_id for s in all_subtasks if s.status == TaskStatus.COMPLETED}
        ready = [s for s in all_subtasks
                 if s.status == TaskStatus.PENDING
                 and all(dep in completed_ids for dep in s.dependencies)]
        return sorted(ready, key=lambda s: s.priority, reverse=True)

    def all_subtasks_complete(self, task_id) -> bool:
        return all(s.status == TaskStatus.COMPLETED
                   for s in self.get_subtasks_for_goal(task_id))

    def any_subtask_failed(self, task_id) -> bool:
        return any(s.status == TaskStatus.FAILED
                   for s in self.get_subtasks_for_goal(task_id))

    # ── Logging ──────────────────────────────────────────────────────────────

    def log_agent_action(self, task_id, agent_name, log_type, content, subtask_id=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO agent_logs VALUES (?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, task_id, subtask_id, agent_name, log_type,
                 content[:5000], datetime.utcnow().isoformat()))
            conn.commit()

    def get_agent_logs(self, task_id, agent_name=None, limit=100) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if agent_name:
                rows = conn.execute(
                    "SELECT * FROM agent_logs WHERE task_id=? AND agent_name=? "
                    "ORDER BY timestamp DESC LIMIT ?", (task_id, agent_name, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM agent_logs WHERE task_id=? "
                    "ORDER BY timestamp DESC LIMIT ?", (task_id, limit)
                ).fetchall()
        return [dict(r) for r in rows]

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_goal(self, task: GoalTask):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO goal_tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (task.task_id, task.goal, task.status, task.plan,
                 json.dumps(task.subtask_ids), task.final_report,
                 task.critic_score, task.critic_feedback,
                 task.created_at, task.completed_at,
                 task.total_llm_calls, int(task.approved)))
            conn.commit()

    def _save_subtask(self, subtask: SubTask):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO subtasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (subtask.subtask_id, subtask.task_id, subtask.agent_name,
                 subtask.description, subtask.expected_output,
                 json.dumps(subtask.context), subtask.status, subtask.priority,
                 json.dumps(subtask.dependencies), subtask.result, subtask.error,
                 json.dumps(subtask.chunk_ids), subtask.created_at,
                 subtask.started_at, subtask.completed_at, subtask.revision_count))
            conn.commit()

    def _get_or_load_goal(self, task_id) -> Optional[GoalTask]:
        if task_id in self._goals:
            return self._goals[task_id]
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM goal_tasks WHERE task_id=?",
                               (task_id,)).fetchone()
        if not row: return None
        task = GoalTask(
            goal=row["goal"], task_id=row["task_id"], status=row["status"],
            plan=row["plan"], subtask_ids=json.loads(row["subtask_ids"] or "[]"),
            final_report=row["final_report"], critic_score=row["critic_score"],
            critic_feedback=row["critic_feedback"], created_at=row["created_at"],
            completed_at=row["completed_at"], total_llm_calls=row["total_llm_calls"],
            approved=bool(row["approved"]),
        )
        self._goals[task_id] = task
        return task

    def _get_or_load_subtask(self, subtask_id) -> Optional[SubTask]:
        if subtask_id in self._subtasks:
            return self._subtasks[subtask_id]
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM subtasks WHERE subtask_id=?",
                               (subtask_id,)).fetchone()
        if not row: return None
        subtask = SubTask(
            subtask_id=row["subtask_id"], task_id=row["task_id"],
            agent_name=row["agent_name"], description=row["description"],
            expected_output=row["expected_output"] or "",
            context=json.loads(row["context"] or "{}"),
            status=row["status"], priority=row["priority"],
            dependencies=json.loads(row["dependencies"] or "[]"),
            result=row["result"], error=row["error"],
            chunk_ids=json.loads(row["chunk_ids"] or "[]"),
            created_at=row["created_at"], started_at=row["started_at"],
            completed_at=row["completed_at"], revision_count=row["revision_count"],
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


_queue_instance: Optional[TaskQueue] = None

def get_queue() -> TaskQueue:
    global _queue_instance
    if _queue_instance is None:
        _queue_instance = TaskQueue()
    return _queue_instance


if __name__ == "__main__":
    import os, tempfile
    from rich import print as rprint
    test_db = tempfile.mktemp(suffix=".db")
    rprint("[bold cyan]Testing Task Queue...[/bold cyan]")
    queue = TaskQueue(db_path=test_db)
    goal = queue.create_goal("Analyse solar energy globally")
    st_r = queue.add_subtask(goal.task_id, AgentName.RESEARCHER, "Find data", priority=2)
    st_c = queue.add_subtask(goal.task_id, AgentName.CODER, "Write chart code",
                              priority=1, dependencies=[st_r.subtask_id])
    ready = queue.get_ready_subtasks(goal.task_id)
    rprint(f"Ready before research: {[s.agent_name for s in ready]}")
    assert len(ready) == 1 and ready[0].agent_name == AgentName.RESEARCHER
    queue.complete_subtask(st_r.subtask_id, result="China: 430 GW, USA: 140 GW")
    ready = queue.get_ready_subtasks(goal.task_id)
    rprint(f"Ready after research: {[s.agent_name for s in ready]}")
    assert len(ready) == 1 and ready[0].agent_name == AgentName.CODER
    os.remove(test_db)
    rprint("[bold green]✓ Task Queue test passed![/bold green]")
''',
)

# ── src/core/base_agent.py ────────────────────────────────────────────────────
write(
    "src/core/base_agent.py",
    '''\
"""
src/core/base_agent.py — Base ReAct Agent
All specialist agents inherit from this class.

What is the ReAct loop?
  THOUGHT: "I need to find solar energy statistics"
  ACTION:  call web_search("solar energy capacity 2024")
  OBSERVATION: [results]
  THOUGHT: "I need India data too"
  ACTION:  call web_search("India solar capacity")
  OBSERVATION: [more results]
  THOUGHT: "I have enough"
  FINAL_ANSWER: [full summary]

Specialist agents override:
  - agent_name, system_prompt, tools — their identity and capabilities
  - run() — entry point (receives SubTask, returns result string)
"""

import json
import re
from abc import ABC
from typing import Any, Callable, Optional
from loguru import logger

from config import settings, TaskStatus
from src.core.llm_client import LLMClient
from src.core.memory import get_memory, ChunkType
from src.core.message_bus import get_bus, make_result_message, MessageType
from src.core.task_queue import get_queue, SubTask


class Tool:
    """A named, callable tool the agent can invoke."""
    def __init__(self, name: str, description: str, func: Callable, example: str = ""):
        self.name = name
        self.description = description
        self.func = func
        self.example = example

    def call(self, **kwargs) -> str:
        try:
            result = self.func(**kwargs)
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Tool error in \'{self.name}\': {e}"

    def to_prompt_str(self) -> str:
        base = f"- {self.name}: {self.description}"
        if self.example:
            base += f"\\n  Example: {self.example}"
        return base


class BaseAgent(ABC):
    """
    Abstract base class for all AgentForge agents.
    Provides the full ReAct loop, memory access, and message bus integration.

    Subclasses define:
      agent_name   (str)          : unique identifier
      system_prompt (str)         : agent personality
      tools        (dict[str, Tool]): available tools
    """

    agent_name: str = "base_agent"
    system_prompt: str = "You are a helpful AI assistant."
    tools: dict[str, Tool] = {}

    def __init__(self):
        self.memory = get_memory()
        self.bus    = get_bus()
        self.queue  = get_queue()
        self.llm    = LLMClient(
            agent_name=self.agent_name,
            system_prompt=self._build_system_prompt(),
        )
        self._conversation_history: list[dict] = []
        logger.info(f"[{self.agent_name}] Agent initialised")

    def _build_system_prompt(self) -> str:
        tools_desc = "\\n".join(t.to_prompt_str() for t in self.tools.values()) if self.tools \
                     else "No tools — reason from existing knowledge."
        react_instructions = f"""
You operate in a THOUGHT → ACTION → OBSERVATION loop.

Format A — to use a tool:
THOUGHT: [your reasoning]
ACTION: tool_name
ACTION_INPUT: {{"arg1": "value1"}}

Format B — when you have the final answer:
THOUGHT: [your reasoning]
FINAL_ANSWER: [complete response]

Available tools:
{tools_desc}

Rules:
- Always start with THOUGHT
- ACTION_INPUT must be valid JSON
- On tool error, try a different approach
- Stop when you have enough info
- FINAL_ANSWER must fully address the task
- Max {settings.max_react_iterations} iterations
"""
        return f"{self.system_prompt}\\n\\n{react_instructions}"

    def react(self, task_description: str, task_id: str, subtask_id: Optional[str] = None) -> str:
        """Run the ReAct loop. Returns the final answer string."""
        self._conversation_history = [{"role": "user", "content": f"Task: {task_description}"}]
        iterations = 0
        final_answer = None

        while iterations < settings.max_react_iterations:
            iterations += 1
            logger.debug(f"[{self.agent_name}] ReAct iteration {iterations}")

            response = self.llm.complete(self._conversation_history)
            self._conversation_history.append({"role": "assistant", "content": response})

            thought = self._extract_thought(response)
            if thought and task_id:
                self.queue.log_agent_action(task_id=task_id, agent_name=self.agent_name,
                                            log_type="thought", content=thought,
                                            subtask_id=subtask_id)

            if "FINAL_ANSWER:" in response:
                final_answer = self._extract_final_answer(response)
                logger.info(f"[{self.agent_name}] FINAL_ANSWER after {iterations} iterations")
                break

            action_name, action_input = self._parse_action(response)
            if not action_name:
                logger.warning(f"[{self.agent_name}] No valid action on iteration {iterations}")
                final_answer = response.strip()
                break

            observation = self._execute_tool(action_name, action_input, task_id, subtask_id)
            self._conversation_history.append({"role": "user", "content": f"OBSERVATION: {observation}"})

        if final_answer is None:
            logger.warning(f"[{self.agent_name}] Max iterations reached, using last response")
            last = self._conversation_history[-1]["content"]
            final_answer = self._extract_final_answer(last) or last

        return final_answer

    def _parse_action(self, text: str):
        action_match = re.search(r"ACTION:\\s*(\\w+)", text)
        if not action_match:
            return None, {}
        action_name = action_match.group(1).strip()
        input_match = re.search(r"ACTION_INPUT:\\s*(\\{.*?\\})", text, re.DOTALL)
        if not input_match:
            return action_name, {}
        try:
            return action_name, json.loads(input_match.group(1))
        except json.JSONDecodeError:
            return action_name, {}

    def _execute_tool(self, action_name, action_input, task_id, subtask_id=None) -> str:
        if action_name not in self.tools:
            return (f"Error: Unknown tool \'{action_name}\'. "
                    f"Available: {list(self.tools.keys())}")
        tool = self.tools[action_name]
        logger.debug(f"[{self.agent_name}] Calling {action_name} with {action_input}")
        if task_id:
            self.queue.log_agent_action(task_id=task_id, agent_name=self.agent_name,
                log_type="action",
                content=f"Tool: {action_name}, Input: {json.dumps(action_input)}",
                subtask_id=subtask_id)
        observation = tool.call(**action_input)
        if task_id:
            self.queue.log_agent_action(task_id=task_id, agent_name=self.agent_name,
                log_type="observation",
                content=f"Tool: {action_name}, Result: {observation[:500]}",
                subtask_id=subtask_id)
        return observation

    def _extract_thought(self, text) -> Optional[str]:
        m = re.search(r"THOUGHT:\\s*(.+?)(?=\\nACTION:|FINAL_ANSWER:|$)", text, re.DOTALL)
        return m.group(1).strip() if m else None

    def _extract_final_answer(self, text) -> Optional[str]:
        m = re.search(r"FINAL_ANSWER:\\s*(.+)", text, re.DOTALL)
        return m.group(1).strip() if m else None

    def store_result(self, content, task_id, chunk_type=ChunkType.TOOL_OUTPUT, metadata=None) -> str:
        return self.memory.store(content=content, agent_name=self.agent_name,
                                 task_id=task_id, chunk_type=chunk_type, metadata=metadata)

    def retrieve_context(self, query, task_id, k=3) -> str:
        chunks = self.memory.retrieve(query=query, task_id=task_id, k=k)
        if not chunks:
            return "No relevant context found in memory."
        return "\\n\\n".join(
            f"[Context {i+1} from {c[\'metadata\'].get(\'agent_name\',\'unknown\')}]:\\n{c[\'content\']}"
            for i, c in enumerate(chunks)
        )

    def run(self, subtask: SubTask) -> str:
        """Main entry point. Runs ReAct, stores result, sends via bus."""
        task_id, subtask_id = subtask.task_id, subtask.subtask_id
        logger.info(f"[{self.agent_name}] Starting {subtask_id}: \'{subtask.description[:60]}\'")
        self.queue.update_subtask(subtask_id, status=TaskStatus.IN_PROGRESS)

        try:
            context_str = ""
            if subtask.context:
                context_str = f"\\n\\nContext from previous agents:\\n{json.dumps(subtask.context, indent=2)}"
            task_description = (f"{subtask.description}\\n\\n"
                                f"Expected output: {subtask.expected_output}{context_str}")

            result = self.react(task_description, task_id, subtask_id)
            result = self._post_process(result, subtask)

            chunk_id = self.store_result(content=result, task_id=task_id,
                chunk_type=self._chunk_type(),
                metadata={"subtask_id": subtask_id, "description": subtask.description})

            self.queue.complete_subtask(subtask_id, result=result, chunk_ids=[chunk_id])
            self.bus.send(make_result_message(
                sender=self.agent_name, recipient="planner",
                task_id=task_id, output=result, chunk_ids=[chunk_id], success=True))

            logger.info(f"[{self.agent_name}] {subtask_id} completed")
            return result

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"[{self.agent_name}] {subtask_id} failed: {error_msg}")
            self.queue.fail_subtask(subtask_id, error=error_msg)
            self.queue.log_agent_action(task_id=task_id, agent_name=self.agent_name,
                                        log_type="error", content=error_msg,
                                        subtask_id=subtask_id)
            self.bus.send(make_result_message(
                sender=self.agent_name, recipient="planner",
                task_id=task_id, output="", success=False, error=error_msg))
            return ""

    def _post_process(self, result: str, subtask: SubTask) -> str:
        return result

    def _chunk_type(self) -> str:
        return ChunkType.TOOL_OUTPUT


if __name__ == "__main__":
    from rich import print as rprint
    rprint("[bold cyan]Testing BaseAgent...[/bold cyan]")

    class EchoAgent(BaseAgent):
        agent_name = "echo_agent"
        system_prompt = "You are a simple echo agent for testing."
        tools = {}

    agent = EchoAgent()
    rprint(f"Agent name   : {agent.agent_name}")
    rprint(f"System prompt: {len(agent.llm.system_prompt)} chars built")
    rprint("[bold green]✓ BaseAgent instantiation passed![/bold green]")
    rprint("[dim](Full ReAct test needs API keys — run src/core/llm_client.py)[/dim]")
''',
)

# ── tests/test_phase1_core.py ─────────────────────────────────────────────────
write(
    "tests/test_phase1_core.py",
    '''\
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
        self.queue.complete_subtask(st.subtask_id, result="print(\'hello\')")
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
        assert parse_json(\'{"a": 1}\')["a"] == 1

    def test_markdown_fences(self):
        from src.core.llm_client import parse_json
        assert parse_json(\'```json\\n{"k": "v"}\\n```\')["k"] == "v"

    def test_json_with_preamble(self):
        from src.core.llm_client import parse_json
        assert parse_json(\'Here you go:\\n{"x": 42}\')["x"] == 42

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
''',
)

print("\n✅ All Phase 1 files created successfully!")
print("\nNext steps:")
print("  1. pip install -r requirements.txt")
print("  2. copy .env.example .env   (then edit with your API keys)")
print("  3. pytest tests/test_phase1_core.py -v")
print("  4. python config.py         (check keys are loaded)")
print("  5. python src/core/llm_client.py   (live API test)")
