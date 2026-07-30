"""
src/tools/url_reader.py — URL Content Reader Tool
───────────────────────────────────────────────────
Fetches a URL and extracts clean readable text from the HTML.

Design decisions:
1. WHY THIS TOOL? Web search gives URLs + 2-sentence snippets.
   The Researcher needs to actually READ page content to extract
   real statistics, quotes, and facts. This tool does that.
2. THREE-STAGE CONTENT EXTRACTION:
   a) Semantic HTML5 tags: <article>, <main> — highest signal
   b) Class-name heuristics: divs with "content"/"article" in class
   c) Fallback: all <p> tags concatenated
   This handles the vast majority of modern web pages cleanly.
3. NOISE REMOVAL: Strips <nav>, <header>, <footer>, <script>, ads, etc.
   These waste context window space and confuse the agent.
4. LENGTH CAP: 8000 chars — enough for several paragraphs of real content.
5. PDF DETECTION: Returns a helpful note instead of garbled binary.

Interview talking point:
    "The content extraction strategy tries semantic HTML elements first,
    then falls back through class-name heuristics to raw paragraph tags.
    This extracts clean content from ~90% of modern pages without needing
    a paid scraping service."
"""

import re
import sys
import os
import json
from typing import Optional
from loguru import logger

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not installed")

try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("beautifulsoup4 not installed")


NOISE_TAGS = [
    "nav",
    "header",
    "footer",
    "aside",
    "script",
    "style",
    "noscript",
    "iframe",
    "form",
    "button",
    "input",
    "select",
]

CONTENT_TAGS = [
    ("article", {}),
    ("main", {}),
    ("div", {"class": re.compile(r"article|content|post|entry|story|body|text", re.I)}),
    (
        "section",
        {"class": re.compile(r"article|content|post|entry|story|body|text", re.I)},
    ),
]

MAX_CONTENT_CHARS = 8000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def read_url(url: str, max_chars: Optional[int] = None) -> str:
    """
    Fetch a URL and return clean extracted text.

    Args:
        url: The URL to fetch.
        max_chars: Max characters to return (default 8000).

    Returns:
        String starting with "URL: ...\nTitle: ...\n\n" followed by content.
        On error: string starting with "Error: ..."
    """
    if not REQUESTS_AVAILABLE or not BS4_AVAILABLE:
        return "Error: requests and beautifulsoup4 required."

    max_chars = max_chars or MAX_CONTENT_CHARS
    logger.debug(f"[url_reader] Fetching: {url}")

    try:
        response = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")

        # PDFs can't be extracted — return a helpful note
        if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
            return (
                f"[PDF document at {url}]\n"
                "This is a PDF file. To read it, try the ArXiv abstract page instead "
                "(remove /pdf/ from the ArXiv URL, or use the arxiv_url field)."
            )

        soup = BeautifulSoup(response.text, "lxml")

        # Remove noise tags
        for tag_name in NOISE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Remove by noise class names
        for tag in soup.find_all(
            class_=re.compile(
                r"nav|menu|sidebar|widget|ad|banner|cookie|popup|footer|header|related|comment",
                re.I,
            )
        ):
            tag.decompose()

        # Extract title
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        content = _extract_content(soup)
        if not content:
            return f"Error: Could not extract text content from {url}"

        # Clean up whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = re.sub(r" {2,}", " ", content)
        content = content.strip()

        # Truncate
        if len(content) > max_chars:
            content = (
                content[:max_chars]
                + f"\n\n[Truncated at {max_chars} chars. Full page: {url}]"
            )

        result = f"URL: {url}\nTitle: {title}\n\n{content}"
        logger.info(f"[url_reader] Extracted {len(content)} chars from {url}")
        return result

    except requests.exceptions.Timeout:
        return f"Error: Request timed out after 10s for {url}"
    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to {url}"
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} from {url}"
    except Exception as e:
        logger.error(f"[url_reader] Unexpected error for {url}: {e}")
        return f"Error reading {url}: {type(e).__name__}: {e}"


def _extract_content(soup) -> str:
    """
    Try multiple strategies to extract main content.
    Returns the best text found.
    """
    # Strategy 1: Semantic HTML5 content tags
    for tag_name, attrs in CONTENT_TAGS:
        found = soup.find(tag_name, attrs)
        if found:
            text = found.get_text(separator="\n", strip=True)
            if len(text) > 200:
                return text

    # Strategy 2: Largest div/section (likely the main content area)
    best_block = None
    best_len = 0
    for tag in soup.find_all(["div", "section"]):
        text = tag.get_text(separator=" ", strip=True)
        if len(text) > best_len:
            best_len = len(text)
            best_block = tag

    if best_block and best_len > 200:
        return best_block.get_text(separator="\n", strip=True)

    # Strategy 3: All paragraph tags
    paragraphs = soup.find_all("p")
    if paragraphs:
        text = "\n\n".join(
            p.get_text(strip=True)
            for p in paragraphs
            if len(p.get_text(strip=True)) > 30
        )
        if len(text) > 100:
            return text

    # Strategy 4: Body fallback
    body = soup.find("body")
    if body:
        return body.get_text(separator="\n", strip=True)

    return soup.get_text(separator="\n", strip=True)


def extract_links(url: str, max_links: int = 10) -> str:
    """
    Extract hyperlinks from a page.
    Useful for finding source links, references, or related pages.
    Returns JSON string: list of {text, url} dicts.
    """
    if not REQUESTS_AVAILABLE or not BS4_AVAILABLE:
        return json.dumps({"error": "requests and beautifulsoup4 required"})

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "lxml")

        links = []
        seen = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            if not href or href.startswith("#") or href.startswith("javascript"):
                continue
            if href.startswith("/"):
                from urllib.parse import urlparse

                parsed = urlparse(url)
                href = f"{parsed.scheme}://{parsed.netloc}{href}"
            if href not in seen and text:
                seen.add(href)
                links.append({"text": text[:100], "url": href})
            if len(links) >= max_links:
                break

        return json.dumps(links, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    from rich import print as rprint

    rprint("[bold cyan]Testing URL Reader Tool...[/bold cyan]")

    rprint("\n[yellow]Test 1: Read Wikipedia page[/yellow]")
    result = read_url("https://en.wikipedia.org/wiki/Solar_energy", max_chars=800)
    rprint(f"Extracted {len(result)} chars:")
    rprint(result[:600] + "...")

    rprint("\n[yellow]Test 2: Read ArXiv abstract[/yellow]")
    result2 = read_url("https://arxiv.org/abs/1706.03762", max_chars=600)
    rprint(result2[:500])

    rprint("[bold green]✓ URL Reader test complete![/bold green]")
