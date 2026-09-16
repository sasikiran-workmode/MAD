"""
Query router for determining which agents to use for a given query.
Uses ThreadPoolExecutor for parallel synchronous retrieval (no asyncio).
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel
from loguru import logger

from agents.base_agent import BaseAgent, registry
from agents.search_agent import WebAgent, ReferenceAgent, ScholarlyAgent
from config import settings


# Scholarly query keywords — only these queries get the Crossref agent
SCHOLARLY_KEYWORDS = {
    "study", "trial", "meta-analysis", "meta analysis", "systematic review",
    "randomized", "randomised", "clinical", "cohort", "evidence for",
    "research on", "doi", "journal", "peer reviewed", "peer-reviewed",
    "pubmed", "arxiv", "preprint",
}


class RoutingDecision(BaseModel):
    """Result of routing a query."""
    agents: List[str] = []  # agent IDs selected
    reasoning: str = ""


def _is_scholarly(query: str) -> bool:
    """Return True if the query warrants the Crossref scholarly agent."""
    q = query.lower()
    return any(kw in q for kw in SCHOLARLY_KEYWORDS)


class QueryRouter:
    """
    Routes queries to appropriate agents and executes them in parallel.

    Three fixed agent profiles:
    - WEB       (DuckDuckGo)   — always included
    - REFERENCE (Wikipedia)    — always included
    - SCHOLARLY (Crossref)     — only for research-flavoured queries
    """

    def __init__(self):
        self._initialized = False

    def initialize(self) -> None:
        """Register the three agents (idempotent)."""
        if self._initialized:
            return
        registry.clear()
        registry.register(WebAgent())
        registry.register(ReferenceAgent())
        registry.register(ScholarlyAgent())
        self._initialized = True
        logger.info(f"Router initialized with agents: {registry.list_ids()}")

    def route(self, query: str) -> RoutingDecision:
        """Determine which agents to use for a query."""
        self.initialize()

        if _is_scholarly(query):
            agents = registry.list_ids()   # all three
            reasoning = "Scholarly query — routing to WEB + REFERENCE + SCHOLARLY agents"
        else:
            # Exclude scholarly for general queries
            agents = [aid for aid in registry.list_ids() if aid != "scholarly_agent"]
            reasoning = "General query — routing to WEB + REFERENCE agents"

        return RoutingDecision(agents=agents, reasoning=reasoning)

    def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute the routing decision synchronously with a ThreadPoolExecutor.

        Args:
            query: User query (passed verbatim to every agent)

        Returns:
            Dictionary with decision and per-agent results
        """
        decision = self.route(query)
        agents: List[BaseAgent] = [
            registry.get(aid) for aid in decision.agents if registry.get(aid)
        ]

        if not agents:
            return {
                "decision": decision.model_dump(),
                "evidence": [],
                "error": "No agents available",
            }

        results = []
        with ThreadPoolExecutor(max_workers=len(agents)) as pool:
            futures = {pool.submit(agent.retrieve, query): agent for agent in agents}
            for fut in as_completed(futures):
                agent = futures[fut]
                try:
                    evidence = fut.result()
                    results.append({
                        "agent_id": agent.agent_id,
                        "evidence": evidence.model_dump(),
                    })
                except Exception as exc:
                    logger.error(f"Agent {agent.agent_id} failed: {exc}")
                    results.append({
                        "agent_id": agent.agent_id,
                        "error": repr(exc),
                    })

        return {
            "decision": decision.model_dump(),
            "evidence": results,
        }


# Global router instance
router = QueryRouter()