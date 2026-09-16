"""
Evidence package for the Multi-Agent Debate framework.
"""
from .models import (
    ConflictType,
    NLILabel,
    EvidenceStatus,
    EvidenceState,
    ComponentStatus,
    Source,
    Claim,
    NormalizedClaim,
    Evidence,
    AgentEvidence,
    EvidenceComparison,
    EvidenceDecision,
    PairwiseRelation,
    Conflict,
    ClassificationResult,
    DebateArgument,
    DebateRound,
    DebateOutcome,
    JudgeResult,
    FinalAnswer,
    ExecutionTrace,
)
from .comparator import EvidenceComparator, find_most_contradictory_pair
from .disagreement_detector import NLIDetector, MockNLIDetector, NLIError, get_nli_detector
from .aggregator import EvidenceAggregator
from .normalizer import EvidenceNormalizer

__all__ = [
    "ConflictType",
    "NLILabel",
    "EvidenceStatus",
    "EvidenceState",
    "ComponentStatus",
    "Source",
    "Claim",
    "NormalizedClaim",
    "Evidence",
    "AgentEvidence",
    "EvidenceComparison",
    "EvidenceDecision",
    "PairwiseRelation",
    "Conflict",
    "ClassificationResult",
    "DebateArgument",
    "DebateRound",
    "DebateOutcome",
    "JudgeResult",
    "FinalAnswer",
    "ExecutionTrace",
    "EvidenceComparator",
    "find_most_contradictory_pair",
    "NLIDetector",
    "MockNLIDetector",
    "NLIError",
    "get_nli_detector",
    "EvidenceAggregator",
    "EvidenceNormalizer",
]