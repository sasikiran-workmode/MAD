"""
Evidence normalizer (Task 8).

Converts raw search results → AgentEvidence with one NormalizedClaim per agent.

Two stages:
1. Relevance filter (cheap, no LLM): content-word overlap between query
   and each result's title + snippet. Below threshold → drop.
2. LLM normalization (one call per agent, role="normalize"):
   turns surviving snippets into one NormalizedClaim with
   text + date/version/scope extracted.
   If no snippets remain or LLM says {"answer": null} → NO_RELEVANT_RESULTS.
"""
import json
import re
from typing import List, Dict, Any, Optional
from loguru import logger

from evidence.models import (
    AgentEvidence, EvidenceStatus, NormalizedClaim, Source,
)
from llm_client import LLMClient, LLMError
from config import settings


# ------------------------------------------------------------------ #
# Relevance filter                                                     #
# ------------------------------------------------------------------ #

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "i", "you", "he",
    "she", "it", "we", "they", "what", "which", "who", "whom", "this",
    "that", "these", "those", "of", "in", "on", "at", "by", "for", "with",
    "about", "as", "into", "through", "during", "to", "from", "up",
    "down", "and", "or", "but", "not", "no", "so",
}


def _content_words(text: str) -> set:
    # Include all alphanumeric tokens (numbers included regardless of length)
    tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def _is_relevant(query: str, result: Dict[str, Any], min_overlap: int = 1) -> bool:
    """Return True if the result has at least min_overlap content-word matches with query.

    Uses min_overlap=1 for very short queries (≤2 content words) so terse queries
    like 'What is 2+2?' or 'Who is Einstein?' are not incorrectly filtered out.
    """
    q_words = _content_words(query)
    if not q_words:
        # Query has no content words (e.g. pure stopwords) — keep all results
        return True
    r_words = _content_words(
        result.get("title", "") + " " + result.get("snippet", "")
    )
    effective_min = 1 if len(q_words) <= 2 else min_overlap
    return len(q_words & r_words) >= effective_min


# ------------------------------------------------------------------ #
# LLM normalization prompt                                             #
# ------------------------------------------------------------------ #

_NORMALIZE_PROMPT = """You are an evidence extraction assistant.

QUERY: {query}

SEARCH SNIPPETS:
{snippets}

Task:
1. Read the snippets and answer the query in ONE concise sentence.
   - You MAY reason from the snippet content to form the answer (e.g. if snippets list
     versions, you can infer which is the 2nd latest).
   - Only return {{"answer": null}} if the snippets contain absolutely NO information
     relevant to the query topic — not just because the answer requires a small inference.
2. Extract any date/year, version number, and scope (platform / population / edition)
   from the most relevant snippet.

Return ONLY a JSON object with these keys:
{{
  "answer": "<one sentence answer, or null ONLY if snippets have zero relevant info>",
  "source_index": <0-based index of the most relevant snippet, or null>,
  "date": "<e.g. 2024, or null>",
  "version": "<e.g. Python 3.13, or null>",
  "scope": "<e.g. adults, enterprise, or null>"
}}"""


# ------------------------------------------------------------------ #
# Normalizer                                                           #
# ------------------------------------------------------------------ #

class EvidenceNormalizer:
    """
    Converts raw search results for one agent into AgentEvidence.
    Uses an LLM call (role="normalize") for claim extraction.
    Falls back to a regex-extracted claim if no LLM is available.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client

    def normalize(
        self,
        agent_id: str,
        query: str,
        raw_results: List[Dict[str, Any]],
    ) -> AgentEvidence:
        """
        Normalize raw search results for one agent.

        Args:
            agent_id: identifier of the retrieval agent
            query: original user query (verbatim)
            raw_results: list of {title, snippet, link} dicts from the provider

        Returns:
            AgentEvidence
        """
        raw_count = len(raw_results)

        # Stage 1: relevance filter
        relevant = [r for r in raw_results if _is_relevant(query, r)]
        filtered_out = raw_count - len(relevant)

        if not relevant:
            return AgentEvidence(
                agent_id=agent_id,
                query=query,
                status=EvidenceStatus.NO_RELEVANT_RESULTS,
                failure_reason="All search results filtered out as irrelevant",
                claims=[],
                sources=[],
                raw_results_count=raw_count,
                filtered_out_count=filtered_out,
            )

        # Build Source objects from relevant results
        sources = [
            Source(
                url=r.get("link", ""),
                title=r.get("title", ""),
                snippet=r.get("snippet", ""),
                metadata={"search_query": query},
            )
            for r in relevant[: settings.max_search_results]
        ]

        # Stage 2: LLM normalization
        claim = self._llm_normalize(agent_id, query, relevant[: settings.nli_max_claims * 3], sources)

        if claim is None:
            return AgentEvidence(
                agent_id=agent_id,
                query=query,
                status=EvidenceStatus.NO_RELEVANT_RESULTS,
                failure_reason="LLM determined snippets do not answer the query",
                claims=[],
                sources=sources,
                raw_results_count=raw_count,
                filtered_out_count=filtered_out,
            )

        return AgentEvidence(
            agent_id=agent_id,
            query=query,
            status=EvidenceStatus.OK,
            claims=[claim],
            sources=sources,
            raw_results_count=raw_count,
            filtered_out_count=filtered_out,
        )

    def _llm_normalize(
        self,
        agent_id: str,
        query: str,
        relevant: List[Dict[str, Any]],
        sources: List[Source],
    ) -> Optional[NormalizedClaim]:
        """Call LLM to extract one NormalizedClaim. Returns None if no answer found."""
        if not self.llm_client:
            # No LLM — use the first snippet as the claim text (offline tests only)
            return NormalizedClaim(
                text=relevant[0].get("snippet", ""),
                source=sources[0],
            )

        snippets_text = "\n".join(
            f"[{i}] {r.get('title', '')} — {r.get('snippet', '')}"
            for i, r in enumerate(relevant)
        )
        prompt = _NORMALIZE_PROMPT.format(query=query, snippets=snippets_text)

        try:
            result = self.llm_client.chat_json(prompt, role="normalize")
        except LLMError as e:
            logger.warning(f"[{agent_id}] LLM normalization failed: {e}; using first snippet")
            return NormalizedClaim(
                text=relevant[0].get("snippet", ""),
                source=sources[0],
            )

        answer = result.get("answer")
        if not answer:
            return None

        # Pick the source from source_index if valid
        src_idx = result.get("source_index")
        source = (
            sources[src_idx] if isinstance(src_idx, int) and src_idx < len(sources)
            else sources[0]
        )

        return NormalizedClaim(
            text=str(answer),
            source=source,
            date=result.get("date") or None,
            version=result.get("version") or None,
            scope=result.get("scope") or None,
        )
