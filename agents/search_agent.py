"""
Search agent for web-based evidence retrieval.
All HTTP calls are synchronous (httpx.Client inside a context manager).
Parallelism is handled by QueryRouter via ThreadPoolExecutor.
"""
import html
import re
import time
from typing import List, Dict, Any, Optional

import httpx
from loguru import logger

from agents.base_agent import BaseAgent, AgentConfig
from evidence.models import Evidence, Source, Claim
from config import settings


class SearchAgentConfig(AgentConfig):
    """Configuration for search agent."""
    max_results: int = 5
    language: str = "en"
    safe_search: bool = True
    provider: str = "duckduckgo"  # duckduckgo | wikipedia | crossref


class SearchAgent(BaseAgent):
    """Agent that retrieves evidence using web search (synchronous)."""

    def __init__(self, config: Optional[SearchAgentConfig] = None):
        super().__init__(config or SearchAgentConfig())
        self.config: SearchAgentConfig = self.config

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _make_client(self) -> httpx.Client:
        """Create a fresh synchronous HTTP client (use with `with` statement)."""
        return httpx.Client(timeout=max(self.config.timeout, 45.0))

    def _retry_get(self, client: httpx.Client, url: str, **kwargs) -> httpx.Response:
        """GET with up to 3 retries on timeout / 5xx / 429, with exponential backoff."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                resp = client.get(url, **kwargs)
                if resp.status_code == 429:
                    wait = 2 ** attempt  # 2s, 4s, 8s
                    if attempt < 3:
                        logger.warning(f"Rate limited ({url[:50]}…), waiting {wait}s")
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                if resp.status_code in (500, 502, 503, 504):
                    if attempt < 3:
                        time.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                resp.raise_for_status()
                return resp
            except httpx.TimeoutException as e:
                last_exc = e
                if attempt < 3:
                    time.sleep(2 ** attempt)
            except httpx.HTTPStatusError:
                raise
        raise last_exc or httpx.TimeoutException("All retries exhausted")

    # ------------------------------------------------------------------ #
    # Provider implementations — NO cross-provider fallbacks              #
    # ------------------------------------------------------------------ #

    def _duckduckgo_search(self, query: str) -> List[Dict[str, Any]]:
        """DuckDuckGo HTML search — free, no API key required."""
        with self._make_client() as client:
            resp = self._retry_get(
                client,
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "evidence-mad/1.0"},
            )
        matches = re.findall(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            resp.text,
            re.I | re.S,
        )
        snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</a>',
            resp.text,
            re.I | re.S,
        )
        results = []
        for i, (link, title) in enumerate(matches[: self.config.max_results]):
            results.append({
                "title": re.sub(r"<[^>]+>", "", html.unescape(title)).strip(),
                "snippet": re.sub(
                    r"<[^>]+>",
                    "",
                    html.unescape(snippets[i] if i < len(snippets) else title),
                ).strip(),
                "link": html.unescape(link),
            })
        return results

    def _wikipedia_search(self, query: str) -> List[Dict[str, Any]]:
        """Wikipedia free public API — NO fallback on failure."""
        with self._make_client() as client:
            resp = self._retry_get(
                client,
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "utf8": 1,
                    "srlimit": self.config.max_results,
                },
                headers={"User-Agent": "EvidenceMAD/1.0 research contact@example.com"},
            )
        items = resp.json().get("query", {}).get("search", [])
        return [
            {
                "title": item.get("title", ""),
                "snippet": re.sub(r"<[^>]+>", "", html.unescape(item.get("snippet", ""))).strip(),
                "link": f"https://en.wikipedia.org/?curid={item.get('pageid', '')}",
            }
            for item in items
        ]

    def _crossref_search(self, query: str) -> List[Dict[str, Any]]:
        """Crossref scholarly metadata API — NO fallback on failure."""
        with self._make_client() as client:
            resp = self._retry_get(
                client,
                "https://api.crossref.org/works",
                params={
                    "query.bibliographic": query,
                    "rows": self.config.max_results,
                    "select": "title,URL,container-title,published",
                },
                headers={"User-Agent": "EvidenceMAD/1.0 (mailto:research@example.com)"},
            )
        items = resp.json().get("message", {}).get("items", [])
        results = []
        for item in items:
            title = (item.get("title") or ["Untitled"])[0]
            venue = (item.get("container-title") or [""])[0]
            published = item.get("published", {}).get("date-parts", [[""]])[0]
            year = published[0] if published else ""
            results.append({
                "title": title,
                "snippet": (
                    f"Scholarly work indexed by Crossref"
                    f"{f' · {venue}' if venue else ''}"
                    f"{f' · {year}' if year else ''}"
                ),
                "link": item.get("URL", ""),
            })
        return results

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Dispatch to the configured provider. Raises on failure (no fallback)."""
        if self.config.provider == "wikipedia":
            return self._wikipedia_search(query)
        if self.config.provider == "crossref":
            return self._crossref_search(query)
        # Default: DuckDuckGo
        return self._duckduckgo_search(query)

    def retrieve(self, query: str) -> Evidence:
        """
        Retrieve evidence for a query using web search.

        Args:
            query: User query (passed verbatim — no suffix)

        Returns:
            Evidence object
        """
        self.logger.info(f"Retrieving evidence for: {query!r}")

        search_results = self.search(query)

        sources = []
        retrieved_texts = []
        for result in search_results:
            source = Source(
                url=result.get("link", ""),
                title=result.get("title", ""),
                snippet=result.get("snippet", ""),
                metadata={"search_query": query},
            )
            sources.append(source)
            retrieved_texts.append(f"{source.title}: {source.snippet}")

        combined_text = "\n\n".join(retrieved_texts)
        claims = self.extract_claims(combined_text, sources)
        answer = self.generate_answer(query, claims, sources)

        return Evidence(
            agent_id=self.agent_id,
            query=query,
            answer=answer,
            claims=claims,
            sources=sources,
            retrieved_text=combined_text,
            metadata={
                "search_results_count": len(search_results),
                "provider": self.config.provider,
            },
        )


# ------------------------------------------------------------------ #
# Concrete agent classes (Task 17 defines routing logic)             #
# ------------------------------------------------------------------ #

class WebAgent(SearchAgent):
    """Primary web agent using DuckDuckGo."""

    def __init__(self):
        config = SearchAgentConfig(
            agent_id="web_agent",
            name="WebAgent",
            description="Primary web search agent (DuckDuckGo)",
            max_results=5,
            provider="duckduckgo",
        )
        super().__init__(config)


class ReferenceAgent(SearchAgent):
    """Reference agent using Wikipedia."""

    def __init__(self):
        config = SearchAgentConfig(
            agent_id="reference_agent",
            name="ReferenceAgent",
            description="Reference search agent (Wikipedia)",
            max_results=5,
            provider="wikipedia",
        )
        super().__init__(config)


class ScholarlyAgent(SearchAgent):
    """Scholarly agent using Crossref."""

    def __init__(self):
        config = SearchAgentConfig(
            agent_id="scholarly_agent",
            name="ScholarlyAgent",
            description="Scholarly search agent (Crossref)",
            max_results=5,
            provider="crossref",
        )
        super().__init__(config)


# Keep legacy names as aliases so existing imports don't break
SearchAgentA = WebAgent
SearchAgentB = ReferenceAgent
SearchAgentC = ScholarlyAgent
