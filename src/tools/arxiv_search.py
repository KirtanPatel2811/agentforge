"""
src/tools/arxiv_search.py — ArXiv Academic Paper Search Tool
──────────────────────────────────────────────────────────────
Gives the Researcher agent access to academic papers on ArXiv.

Design decisions:
1. WHY ARXIV? The primary preprint server for CS/AI/ML research.
   Almost every important AI paper is here. No API key needed.
2. STRUCTURED OUTPUT: Returns title, authors, abstract preview, PDF URL.
   The agent reads abstracts to decide relevance before fetching full PDFs.
3. ABSTRACT TRUNCATION: Abstracts truncated to 400 chars for initial scan.
   Agent can use url_reader on the arxiv_url for the full abstract.
4. SORT OPTIONS: "relevance" (default) or "submittedDate" for newest first.

Interview talking point:
    "The Researcher uses two tools in combination: arxiv_search to find
    relevant papers by query, then url_reader to extract full content from
    the most promising ones. This mirrors how a human researcher would scan
    abstracts before committing to read a full paper."
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
    import arxiv

    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    logger.warning("arxiv not installed. Run: pip install arxiv")


def arxiv_search(
    query: str,
    max_results: Optional[int] = None,
    sort_by: str = "relevance",
) -> str:
    """
    Search ArXiv for academic papers matching the query.

    Args:
        query: Search query. Plain text works. Can also use operators:
               "ti:transformer" (title search), "au:vaswani" (author),
               "cat:cs.AI" (AI category).
        max_results: Number of papers to return.
        sort_by: "relevance" (default) or "submittedDate" (newest first).

    Returns:
        JSON string: list of {title, authors, abstract, pdf_url,
                              arxiv_url, published, categories, arxiv_id}
    """
    if not ARXIV_AVAILABLE:
        return json.dumps(
            {"error": "arxiv package not installed. Run: pip install arxiv"}
        )

    max_results = max_results or settings.max_arxiv_results
    logger.debug(f"[arxiv_search] Query: '{query}' (max={max_results}, sort={sort_by})")

    try:
        sort_criterion = (
            arxiv.SortCriterion.SubmittedDate
            if sort_by == "submittedDate"
            else arxiv.SortCriterion.Relevance
        )

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion,
        )

        papers = []
        for result in client.results(search):
            abstract = result.summary.replace("\n", " ").strip()
            abstract_preview = (
                abstract[:400] + "..." if len(abstract) > 400 else abstract
            )

            papers.append(
                {
                    "title": result.title,
                    "authors": [a.name for a in result.authors[:5]],
                    "abstract": abstract_preview,
                    "pdf_url": result.pdf_url,
                    "arxiv_url": result.entry_id,
                    "published": (
                        result.published.strftime("%Y-%m-%d")
                        if result.published
                        else ""
                    ),
                    "categories": result.categories,
                    "arxiv_id": result.entry_id.split("/")[-1],
                }
            )

        logger.info(f"[arxiv_search] Found {len(papers)} papers for '{query}'")
        return json.dumps(papers, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"[arxiv_search] Failed: {e}")
        return json.dumps({"error": str(e), "query": query})


def arxiv_get_paper(arxiv_id: str) -> str:
    """
    Get a specific ArXiv paper by its ID (e.g. "2307.09288" for LLaMA 2,
    or "1706.03762" for Attention is All You Need).

    Returns full paper details with complete abstract.
    Use when you know the exact paper you want.
    """
    if not ARXIV_AVAILABLE:
        return json.dumps({"error": "arxiv package not installed"})

    logger.debug(f"[arxiv_get_paper] Fetching: {arxiv_id}")

    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(client.results(search))

        if not results:
            return json.dumps({"error": f"Paper not found: {arxiv_id}"})

        r = results[0]
        paper = {
            "title": r.title,
            "authors": [a.name for a in r.authors],
            "abstract": r.summary.replace("\n", " ").strip(),
            "pdf_url": r.pdf_url,
            "arxiv_url": r.entry_id,
            "published": r.published.strftime("%Y-%m-%d") if r.published else "",
            "updated": r.updated.strftime("%Y-%m-%d") if r.updated else "",
            "categories": r.categories,
            "comment": r.comment or "",
        }
        return json.dumps(paper, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"[arxiv_get_paper] Failed: {e}")
        return json.dumps({"error": str(e), "arxiv_id": arxiv_id})


if __name__ == "__main__":
    from rich import print as rprint

    rprint("[bold cyan]Testing ArXiv Search Tool...[/bold cyan]")

    rprint("\n[yellow]Test 1: Search by query[/yellow]")
    result = arxiv_search("large language model agents ReAct reasoning", max_results=3)
    papers = json.loads(result)
    if isinstance(papers, list):
        rprint(f"Found {len(papers)} papers:")
        for p in papers:
            rprint(f"  [{p['published']}] {p['title'][:70]}")
            rprint(f"   Authors: {', '.join(p['authors'][:2])}")
    else:
        rprint(f"[red]{papers}[/red]")

    rprint("\n[yellow]Test 2: Get specific paper (Attention is All You Need)[/yellow]")
    result2 = arxiv_get_paper("1706.03762")
    paper = json.loads(result2)
    if "title" in paper:
        rprint(f"Title: {paper['title']}")
        rprint(f"Authors: {', '.join(paper['authors'][:3])}")
        rprint(f"Published: {paper['published']}")
    else:
        rprint(f"[red]{paper}[/red]")

    rprint("[bold green]✓ ArXiv search test complete![/bold green]")
