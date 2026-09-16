"""
Debate package for the Multi-Agent Debate framework.
"""
from .proposer import Proposer
from .critic import Critic
from .reviser import Reviser
from .manager import DebateManager, get_debate_manager
from .judge import Judge, get_judge

__all__ = [
    "Proposer",
    "Critic",
    "Reviser",
    "DebateManager",
    "get_debate_manager",
    "Judge",
    "get_judge",
]