"""
Core data models for the Multi-Agent Debate framework.
All models use Pydantic for validation and serialization.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


# ------------------------------------------------------------------ #
# Enums                                                                #
# ------------------------------------------------------------------ #

class ConflictType(str, Enum):
    """Types of conflicts that can be detected between evidence."""
    FACTUAL = "factual"
    TEMPORAL = "temporal"
    VERSION = "version"
    CONTEXTUAL = "contextual"
    SOURCE = "source"
    UNKNOWN = "unknown"


class NLILabel(str, Enum):
    """NLI relationship labels."""
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


class EvidenceStatus(str, Enum):
    """Status of a single agent's evidence retrieval."""
    OK = "ok"
    NO_RELEVANT_RESULTS = "no_relevant_results"
    RETRIEVAL_FAILED = "retrieval_failed"


class EvidenceState(str, Enum):
    """Multi-agent evidence decision state."""
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    UNCERTAIN = "uncertain"
    INSUFFICIENT = "insufficient"


class ComponentStatus(str, Enum):
    """Status of an LLM-backed component call."""
    LLM_SUCCESS = "llm_success"
    FALLBACK = "fallback"
    FAILED = "failed"


# ------------------------------------------------------------------ #
# Source / Claim / Evidence primitives                                 #
# ------------------------------------------------------------------ #

class Source(BaseModel):
    """Source information for evidence."""
    url: str
    title: str
    snippet: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retrieved_at: datetime = Field(default_factory=datetime.now)
    credibility_score: Optional[float] = None


class Claim(BaseModel):
    """A single claim extracted from evidence (legacy regex path)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str
    source: Source
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NormalizedClaim(BaseModel):
    """
    A structured claim produced by the LLM normalizer (Task 8).
    Carries provenance and temporal/version/scope metadata.
    """
    text: str                          # one sentence that answers the query
    source: Source                     # provenance preserved
    date: Optional[str] = None         # enables temporal reasoning
    version: Optional[str] = None      # enables version reasoning
    scope: Optional[str] = None        # platform / population / edition


class Evidence(BaseModel):
    """Evidence retrieved by an agent for a query (legacy model kept for compatibility)."""
    agent_id: str
    query: str
    answer: str
    claims: List[Claim] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)
    retrieved_text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retrieved_at: datetime = Field(default_factory=datetime.now)


class AgentEvidence(BaseModel):
    """
    Structured evidence from one agent after relevance filtering + normalization (Task 8).
    This replaces the raw Evidence model in the aggregation pipeline.
    """
    agent_id: str
    query: str
    status: EvidenceStatus
    failure_reason: Optional[str] = None
    claims: List[NormalizedClaim] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)
    raw_results_count: int = 0
    filtered_out_count: int = 0
    retrieved_at: datetime = Field(default_factory=datetime.now)

    @property
    def is_usable(self) -> bool:
        """True only when retrieval succeeded and at least one claim exists."""
        return self.status is EvidenceStatus.OK and bool(self.claims)


# ------------------------------------------------------------------ #
# Evidence comparison                                                  #
# ------------------------------------------------------------------ #

class PairwiseRelation(BaseModel):
    """NLI result between one pair of agent claims."""
    agent_a_id: str
    agent_b_id: str
    claim_a_text: str
    claim_b_text: str
    label: NLILabel
    contradiction_score: float = 0.0
    entailment_score: float = 0.0
    neutral_score: float = 0.0


class EvidenceDecision(BaseModel):
    """
    Output of the N-agent evidence aggregator (Task 9).
    The decision state drives all downstream branching in app.py.
    """
    state: EvidenceState
    pairwise: List[PairwiseRelation] = Field(default_factory=list)
    usable_agents: List[str] = Field(default_factory=list)
    failed_agents: List[Dict[str, str]] = Field(default_factory=list)  # id → reason
    top_conflict: Optional[Dict[str, Any]] = None   # highest-contradiction pair
    rationale: str = ""


# Legacy comparison model kept for backward compat
class EvidenceComparison(BaseModel):
    """Result of comparing evidence from two agents (legacy — use EvidenceDecision instead)."""
    agent_a_id: str
    agent_b_id: str
    claim_pairs: List[Dict[str, Any]] = Field(default_factory=list)
    nli_results: List[Dict[str, Any]] = Field(default_factory=list)
    has_disagreement: bool = False
    contradiction_score: float = 0.0
    entailment_score: float = 0.0
    neutral_score: float = 0.0
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    compared_at: datetime = Field(default_factory=datetime.now)


# ------------------------------------------------------------------ #
# Conflict                                                             #
# ------------------------------------------------------------------ #

class Conflict(BaseModel):
    """A detected conflict between evidence."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: ConflictType
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    claim_a: NormalizedClaim           # guaranteed when constructed on DISAGREEMENT path
    claim_b: NormalizedClaim
    metadata: Dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime = Field(default_factory=datetime.now)


class ClassificationResult(BaseModel):
    """Result from ConflictClassifier with status tracking (Task 11)."""
    type: ConflictType
    confidence: float
    explanation: str
    status: ComponentStatus = ComponentStatus.LLM_SUCCESS
    error: Optional[str] = None


# ------------------------------------------------------------------ #
# Debate                                                               #
# ------------------------------------------------------------------ #

class DebateArgument(BaseModel):
    """An argument made during debate."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    round_number: int
    role: str  # "proposer" | "critic" | "reviser"
    agent_id: str
    content: str
    evidence_refs: List[str] = Field(default_factory=list)
    conflict_type: Optional[ConflictType] = None
    strategy_used: Optional[str] = None
    strategy_text_preview: Optional[str] = None   # first 200 chars of strategy text sent
    created_at: datetime = Field(default_factory=datetime.now)


class DebateRound(BaseModel):
    """A single round of debate."""
    round_number: int
    proposer_argument: Optional[DebateArgument] = None
    critic_arguments: List[DebateArgument] = Field(default_factory=list)
    reviser_argument: Optional[DebateArgument] = None
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class DebateOutcome(BaseModel):
    """Complete outcome of a debate run (Task 14)."""
    rounds: List[DebateRound] = Field(default_factory=list)
    status: Literal["completed", "partial", "failed"] = "failed"
    stop_reason: Literal["converged", "no_objection", "max_rounds", "stage_failure"] = "stage_failure"
    failed_stages: List[str] = Field(default_factory=list)
    rounds_used: int = 0


class JudgeResult(BaseModel):
    """Result from the judge evaluating debate outcomes."""
    winner: str  # candidate identifier
    reason: str
    evidence_score: float = Field(ge=0.0, le=1.0)
    reasoning_score: float = Field(ge=0.0, le=1.0)
    source_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    scores: Dict[str, float] = Field(default_factory=dict)
    evaluated_at: datetime = Field(default_factory=datetime.now)


# ------------------------------------------------------------------ #
# Final answer + Execution trace                                       #
# ------------------------------------------------------------------ #

class FinalAnswer(BaseModel):
    """Final answer generated by the system."""
    query: str
    answer: str
    reasoning: str
    evidence_state: EvidenceState = EvidenceState.INSUFFICIENT
    sources: List[Source] = Field(default_factory=list)
    debate_triggered: bool = False
    conflict: Optional[Conflict] = None
    debate_outcome: Optional[DebateOutcome] = None
    judge_result: Optional[JudgeResult] = None
    independent_agents_agreeing: int = 0   # distinct agents (not source count)
    source_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.now)


class ExecutionTrace(BaseModel):
    """Complete execution trace for logging and evaluation."""
    query: str
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    router_result: Dict[str, Any] = Field(default_factory=dict)
    # Populated after retrieval
    agent_evidence: List[AgentEvidence] = Field(default_factory=list)
    # Legacy field kept so old code that writes evidence_list still works
    agents: List[Evidence] = Field(default_factory=list)
    # Aggregator decision
    evidence_decision: Optional[EvidenceDecision] = None
    # Legacy
    evidence_comparison: Optional[EvidenceComparison] = None
    disagreement_detected: bool = False
    contradiction_score: float = 0.0
    debate_triggered: bool = False
    conflict: Optional[Conflict] = None
    debate_rounds: List[DebateRound] = Field(default_factory=list)
    debate_outcome: Optional[DebateOutcome] = None
    judge_result: Optional[JudgeResult] = None
    final_answer: Optional[FinalAnswer] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None