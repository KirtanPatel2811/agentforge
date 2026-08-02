import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.base_agent import BaseAgent, Tool
from src.core.memory import ChunkType
from src.tools.web_search import web_search, news_search
from src.tools.arxiv_search import arxiv_search
from src.tools.url_reader import read_url
from loguru import logger


class ResearcherAgent(BaseAgent):
    """
    Researcher agent — finds information via web, news, ArXiv, and URL reading.

    Interview talking point:
        The Researcher combines three information sources: general web search
        for broad coverage, news search for recency, and ArXiv for academic
        depth. It uses url_reader to go beyond search snippets. All findings
        are stored in ChromaDB with source metadata so the Writer can cite them.
    """

    agent_name = "researcher"

    system_prompt = (
        "You are an expert Research Agent in a multi-agent AI system.\n\n"
        "Your job is to find accurate, relevant, up-to-date information on any topic.\n\n"
        "Tools available:\n"
        "- web_search: general web results\n"
        "- news_search: recent news articles\n"
        "- arxiv_search: academic papers\n"
        "- read_url: fetch and read full page content\n\n"
        "Your FINAL_ANSWER must be a structured research summary with:\n"
        "1. KEY FINDINGS: bullet points with specific numbers/statistics\n"
        "2. SOURCES: URLs you actually read\n"
        "3. ACADEMIC CONTEXT: relevant papers found\n"
        "4. GAPS: what you could not find\n\n"
        "Quality standards:\n"
        "- Prefer specific numbers over vague claims\n"
        "- Always read at least 2 URLs before writing your final answer\n"
        "- Note publication dates of statistics\n"
        "- If sources conflict, report both"
    )

    def __init__(self):
        self.tools = {
            "web_search": Tool(
                name="web_search",
                description="Search the web. Returns titles, URLs, snippets.",
                func=web_search,
                example='ACTION_INPUT: {"query": "solar energy capacity 2024 by country"}',
            ),
            "news_search": Tool(
                name="news_search",
                description="Search recent news. Use for current events.",
                func=news_search,
                example='ACTION_INPUT: {"query": "solar energy record 2024"}',
            ),
            "arxiv_search": Tool(
                name="arxiv_search",
                description="Search academic papers on ArXiv.",
                func=arxiv_search,
                example='ACTION_INPUT: {"query": "renewable energy deep learning", "max_results": 3}',
            ),
            "read_url": Tool(
                name="read_url",
                description="Fetch and read a full web page. Use after web_search for complete content.",
                func=read_url,
                example='ACTION_INPUT: {"url": "https://example.com/article"}',
            ),
        }
        super().__init__()

    def _chunk_type(self):
        return ChunkType.RESEARCH

    def _post_process(self, result, subtask):
        if "KEY FINDINGS" not in result and "SOURCES" not in result:
            return "RESEARCH FINDINGS\n" + "=" * 40 + "\n" + result
        return result
