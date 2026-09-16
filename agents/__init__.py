"""
Agents package for the Multi-Agent Debate framework.
"""
from .base_agent import BaseAgent, AgentConfig, AgentRegistry, registry
from .search_agent import (
    SearchAgent, SearchAgentConfig,
    WebAgent, ReferenceAgent, ScholarlyAgent,
    # Legacy aliases kept for backwards-compat
    SearchAgentA, SearchAgentB, SearchAgentC,
)
from .router import QueryRouter, RoutingDecision, router

__all__ = [
    "BaseAgent",
    "AgentConfig",
    "AgentRegistry",
    "registry",
    "SearchAgent",
    "SearchAgentConfig",
    "WebAgent",
    "ReferenceAgent",
    "ScholarlyAgent",
    "SearchAgentA",
    "SearchAgentB",
    "SearchAgentC",
    "QueryRouter",
    "RoutingDecision",
    "router",
]