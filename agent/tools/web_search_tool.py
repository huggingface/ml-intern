"""
Web search tool backed by the Exa API.

Exposes general-purpose web search so the agent can ground answers in
current web content — useful when a topic falls outside the HF ecosystem,
when training recipes require recent blog posts or announcements, or when
arxiv alone does not surface the best reference.

Disabled unless ``EXA_API_KEY`` is set in the environment; the tool
spec factory returns ``None`` so the router simply won't register it.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

ENV_VAR = "EXA_API_KEY"
INTEGRATION_NAME = "ml-intern"

DEFAULT_NUM_RESULTS = 5
MAX_NUM_RESULTS = 25
DEFAULT_SUMMARY_CHARACTERS = 1200

SEARCH_TYPES = ["auto", "neural", "fast"]
CATEGORIES = [
    "company",
    "research paper",
    "news",
    "personal site",
    "financial report",
    "linkedin profile",
    "pdf",
    "github",
    "tweet",
]


@dataclass
class WebSearchResult:
    """Normalized search result built from the Exa response."""

    title: str
    url: str
    published_date: str | None = None
    author: str | None = None
    score: float | None = None
    summary: str | None = None
    highlights: list[str] = field(default_factory=list)
    text: str | None = None

    def snippet(self, max_characters: int = 500) -> str:
        """Return the best-available snippet, preferring summary > highlights > text."""
        if self.summary:
            return _truncate(self.summary, max_characters)
        if self.highlights:
            joined = " … ".join(h.strip() for h in self.highlights if h and h.strip())
            if joined:
                return _truncate(joined, max_characters)
        if self.text:
            return _truncate(self.text, max_characters)
        return ""


def _truncate(text: str, max_characters: int) -> str:
    text = text.strip()
    if len(text) <= max_characters:
        return text
    return text[: max_characters - 1].rstrip() + "…"


def _coerce_result(raw: Any) -> WebSearchResult:
    """Map an Exa SDK result object (or plain dict) to WebSearchResult."""
    def _get(key: str, default: Any = None) -> Any:
        if isinstance(raw, dict):
            return raw.get(key, default)
        return getattr(raw, key, default)

    highlights = _get("highlights") or []
    if not isinstance(highlights, list):
        highlights = [str(highlights)]

    return WebSearchResult(
        title=(_get("title") or "").strip() or "(untitled)",
        url=(_get("url") or "").strip(),
        published_date=_get("published_date") or _get("publishedDate"),
        author=_get("author"),
        score=_get("score"),
        summary=_get("summary"),
        highlights=[str(h) for h in highlights if h],
        text=_get("text"),
    )


def _format_results(
    query: str,
    results: list[WebSearchResult],
    search_type: str,
    auto_selected: str | None,
) -> str:
    if not results:
        return f"No web results found for query: {query!r}."

    header = f"Web search results for: {query!r}"
    if auto_selected and auto_selected != search_type:
        header += f" (type={search_type} → resolved={auto_selected})"
    else:
        header += f" (type={search_type})"
    lines: list[str] = [header, ""]

    for i, r in enumerate(results, 1):
        lines.append(f"{i}. **{r.title}**")
        lines.append(f"   URL: {r.url}")
        meta: list[str] = []
        if r.published_date:
            meta.append(f"published: {r.published_date}")
        if r.author:
            meta.append(f"author: {r.author}")
        if r.score is not None:
            meta.append(f"score: {r.score:.2f}")
        if meta:
            lines.append(f"   {' | '.join(meta)}")

        snippet = r.snippet()
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _build_contents_kwargs(
    text: bool, summary: bool, highlights: bool
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if text:
        kwargs["text"] = True
    if highlights:
        kwargs["highlights"] = True
    if summary:
        kwargs["summary"] = True
    return kwargs


def _run_search(
    api_key: str,
    query: str,
    num_results: int,
    search_type: str,
    category: str | None,
    include_domains: list[str] | None,
    exclude_domains: list[str] | None,
    start_published_date: str | None,
    end_published_date: str | None,
    include_text: bool,
    include_summary: bool,
    include_highlights: bool,
) -> tuple[list[WebSearchResult], str | None]:
    """Synchronous Exa call, run inside asyncio.to_thread."""
    from exa_py import Exa  # imported lazily so the module loads without the dep

    client = Exa(api_key)
    # Integration tracking — lets the Exa team attribute usage to this repo.
    client.headers["x-exa-integration"] = INTEGRATION_NAME

    params: dict[str, Any] = {
        "query": query,
        "num_results": num_results,
        "type": search_type,
    }
    if category:
        params["category"] = category
    if include_domains:
        params["include_domains"] = include_domains
    if exclude_domains:
        params["exclude_domains"] = exclude_domains
    if start_published_date:
        params["start_published_date"] = start_published_date
    if end_published_date:
        params["end_published_date"] = end_published_date

    content_kwargs = _build_contents_kwargs(
        text=include_text,
        summary=include_summary,
        highlights=include_highlights,
    )

    if content_kwargs:
        response = client.search_and_contents(**params, **content_kwargs)
    else:
        response = client.search(**params)

    raw_results = getattr(response, "results", None) or []
    auto_type = getattr(response, "resolved_search_type", None) or getattr(
        response, "search_type", None
    )
    return [_coerce_result(r) for r in raw_results], auto_type


async def web_search_handler(
    arguments: dict[str, Any], session=None
) -> tuple[str, bool]:
    """Agent handler: run an Exa web search and format the results."""
    api_key = os.environ.get(ENV_VAR)
    if not api_key:
        return (
            f"Error: {ENV_VAR} is not set — web_search is unavailable.",
            False,
        )

    query = (arguments.get("query") or "").strip()
    if not query:
        return "Error: 'query' is required.", False

    try:
        num_results = int(arguments.get("num_results", DEFAULT_NUM_RESULTS))
    except (TypeError, ValueError):
        return "Error: num_results must be an integer.", False
    num_results = max(1, min(num_results, MAX_NUM_RESULTS))

    search_type = (arguments.get("type") or "auto").strip() or "auto"
    if search_type not in SEARCH_TYPES:
        return (
            f"Error: type must be one of {SEARCH_TYPES}, got {search_type!r}.",
            False,
        )

    category = (arguments.get("category") or "").strip() or None
    if category and category not in CATEGORIES:
        return (
            f"Error: category must be one of {CATEGORIES}, got {category!r}.",
            False,
        )

    include_domains = arguments.get("include_domains") or None
    exclude_domains = arguments.get("exclude_domains") or None
    if include_domains is not None and not isinstance(include_domains, list):
        return "Error: include_domains must be a list of strings.", False
    if exclude_domains is not None and not isinstance(exclude_domains, list):
        return "Error: exclude_domains must be a list of strings.", False

    start_published_date = (arguments.get("start_published_date") or "").strip() or None
    end_published_date = (arguments.get("end_published_date") or "").strip() or None

    include_text = bool(arguments.get("include_text", False))
    include_summary = bool(arguments.get("include_summary", True))
    include_highlights = bool(arguments.get("include_highlights", True))

    try:
        results, auto_type = await asyncio.to_thread(
            _run_search,
            api_key,
            query,
            num_results,
            search_type,
            category,
            include_domains,
            exclude_domains,
            start_published_date,
            end_published_date,
            include_text,
            include_summary,
            include_highlights,
        )
    except ImportError:
        return (
            "Error: exa-py is not installed. Run `uv sync` or "
            "`pip install exa-py` to enable web_search.",
            False,
        )
    except Exception as e:  # noqa: BLE001 — Exa SDK raises several error types
        logger.exception("Exa web_search failed")
        return f"Web search error: {e}", False

    return _format_results(query, results, search_type, auto_type), True


WEB_SEARCH_TOOL_SPEC = {
    "name": "web_search",
    "description": (
        "Search the open web with Exa for current information outside the HF ecosystem. "
        "Use when HF docs / papers / GitHub search aren't enough — e.g. recent blog posts, "
        "announcements, product pages, non-arxiv references, or cross-domain context.\n\n"
        "Tips:\n"
        "  • Set category='research paper' to bias toward academic sources.\n"
        "  • Use include_domains / exclude_domains to scope to known-good sources.\n"
        "  • Results include a summary by default; set include_text=true for full page text.\n"
        "  • Narrow recency with start_published_date / end_published_date (ISO 8601).\n\n"
        "Requires the EXA_API_KEY environment variable."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query (natural-language phrase works best for neural search).",
            },
            "num_results": {
                "type": "integer",
                "description": f"Number of results to return (default {DEFAULT_NUM_RESULTS}, max {MAX_NUM_RESULTS}).",
                "minimum": 1,
                "maximum": MAX_NUM_RESULTS,
            },
            "type": {
                "type": "string",
                "enum": SEARCH_TYPES,
                "description": "Search mode: 'auto' (default) balances neural + fast, 'neural' for semantic, 'fast' for low-latency.",
            },
            "category": {
                "type": "string",
                "enum": CATEGORIES,
                "description": "Optional content category filter (e.g. 'research paper', 'news', 'github', 'pdf').",
            },
            "include_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Only return results from these domains (e.g. ['huggingface.co', 'arxiv.org']).",
            },
            "exclude_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exclude results from these domains.",
            },
            "start_published_date": {
                "type": "string",
                "description": "Only return results published on/after this ISO 8601 date (YYYY-MM-DD).",
            },
            "end_published_date": {
                "type": "string",
                "description": "Only return results published on/before this ISO 8601 date (YYYY-MM-DD).",
            },
            "include_text": {
                "type": "boolean",
                "description": "Include the full page text for each result (verbose; default false).",
            },
            "include_summary": {
                "type": "boolean",
                "description": "Include an LLM-generated summary per result (default true).",
            },
            "include_highlights": {
                "type": "boolean",
                "description": "Include LLM-selected highlight snippets per result (default true).",
            },
        },
        "required": ["query"],
    },
}


def web_search_enabled() -> bool:
    """Whether the tool should be registered — depends on the env var only."""
    return bool(os.environ.get(ENV_VAR))
