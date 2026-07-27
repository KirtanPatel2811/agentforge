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
    rprint(f"  Groq key set  : {'✓' if settings.groq_api_key else '✗ MISSING'}")
    rprint(f"  Google key set: {'✓' if settings.google_api_key else '✗ MISSING'}")
    rprint(f"  Data dir      : {DATA_DIR}")
