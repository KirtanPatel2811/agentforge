"""
src/tools/web_search.py — DuckDuckGo Web Search Tool
──────────────────────────────────────────────────────
Gives the Researcher agent the ability to search the web.

Design decisions:
1. WHY DUCKDUCKGO? No API key needed, no rate limits for reasonable use.
   Production would use Serper or Bing for more reliability.
2. STRUCTURED OUTPUT: Returns {title, url, snippet} dicts — not raw HTML.
3. FALLBACK TO SERPER: If DDG fails, we try Serper.dev (100 free/month).
4. NEWS SEARCH: Separate ddgs.news() for recent events queries.
"""

import json
from typing import Optional
from loguru import logger

import sys, os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from config import settings

try:
    from duckduckgo_search import DDGS

    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("duckduckgo-search not installed.")

try:
    import requests as _requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def web_search(query: str, max_results: Optional[int] = None) -> str:
    """
    Search the web using DuckDuckGo. Falls back to Serper if DDG fails.

    Args:
        query: The search query string.
        max_results: Number of results to return (default from config).

    Returns:
        JSON string: list of {title, url, snippet, source} dicts.
        On failure: JSON error object.

    Interview talking point:
        "I implemented dual-provider search with automatic fallback.
        DuckDuckGo needs no API key, but I added Serper as a fallback
        since DDG can be rate-limited in automated contexts. Both providers
        return the same structured format so the agent doesn't know or care
        which one answered."
    """
    max_results = max_results or settings.max_search_results
    logger.debug(f"[web_search] Query: '{query}' (max={max_results})")

    # Try DuckDuckGo first
    if DDGS_AVAILABLE:
        try:
            results = _ddg_search(query, max_results)
            if results:
                logger.info(f"[web_search] DDG returned {len(results)} results")
                return json.dumps(results, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[web_search] DuckDuckGo failed: {e}. Trying Serper.")

    # Serper fallback
    if settings.serper_api_key and REQUESTS_AVAILABLE:
        try:
            results = _serper_search(query, max_results)
            if results:
                logger.info(f"[web_search] Serper returned {len(results)} results")
                return json.dumps(results, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[web_search] Serper also failed: {e}")

    return json.dumps(
        {
            "error": "Web search unavailable",
            "query": query,
            "suggestion": "Check internet connection or add SERPER_API_KEY to .env",
        }
    )


def _ddg_search(query: str, max_results: int) -> list[dict]:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                    "source": "duckduckgo",
                }
            )
    return results


def _serper_search(query: str, max_results: int) -> list[dict]:
    response = _requests.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": settings.serper_api_key,
            "Content-Type": "application/json",
        },
        json={"q": query, "num": max_results},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    results = []
    for item in data.get("organic", [])[:max_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": "serper",
            }
        )
    return results


def news_search(query: str, max_results: Optional[int] = None) -> str:
    """
    Search recent news via DuckDuckGo News.
    Use when the query is about current events, not general knowledge.

    Returns JSON string: list of {title, url, snippet, date, source} dicts.
    """
    max_results = max_results or settings.max_search_results
    logger.debug(f"[news_search] Query: '{query}'")

    if not DDGS_AVAILABLE:
        return json.dumps({"error": "duckduckgo-search not installed"})

    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=max_results):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("body", ""),
                        "date": r.get("date", ""),
                        "source": r.get("source", ""),
                    }
                )
        logger.info(f"[news_search] Returned {len(results)} news results")
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[news_search] Failed: {e}")
        return json.dumps({"error": str(e), "query": query})


if __name__ == "__main__":
    from rich import print as rprint

    rprint("[bold cyan]Testing Web Search Tool...[/bold cyan]")

    rprint("\n[yellow]Test 1: Web search[/yellow]")
    result = web_search("solar energy global capacity 2024 top countries")
    data = json.loads(result)
    if isinstance(data, list):
        rprint(f"Got {len(data)} results:")
        for r in data[:3]:
            rprint(f"  [{r['source']}] {r['title'][:70]}")
    else:
        rprint(f"[red]Error: {data}[/red]")

    rprint("\n[yellow]Test 2: News search[/yellow]")
    result2 = news_search("AI large language models 2024")
    data2 = json.loads(result2)
    if isinstance(data2, list):
        rprint(f"Got {len(data2)} news results:")
        for r in data2[:2]:
            rprint(f"  {r['title'][:70]} ({r.get('date', '')})")
    else:
        rprint(f"[red]Error: {data2}[/red]")

    rprint("[bold green]✓ Web search test complete![/bold green]")
