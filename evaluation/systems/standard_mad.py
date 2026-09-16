"""
Standard MAD Baseline System (Task 25).
Real retrieval and normalization.
No gate — always debates using UNKNOWN (generic) strategy.
Same debate code, same round cap as ProposedMADSystem.
Does NOT inject mock conflicts.
"""
import time
from typing import Optional, List
from loguru import logger

from evidence.models import (
    ExecutionTrace, AgentEvidence, EvidenceStatus,
    Conflict, ConflictType, NormalizedClaim, Source, EvidenceState,
)
from evidence.aggregator import EvidenceAggregator
from evidence.normalizer import EvidenceNormalizer
from evidence.disagreement_detector import MockNLIDetector, get_nli_detector
from debate.manager import get_debate_manager
from debate.judge import get_judge
from output.generator import get_answer_generator
from llm_client import LLMClient, get_llm_client
from run_mode import RunMode
from config import settings
from app import MADSystem, save_trace


class StandardMADSystem:
    """
    Standard MAD: real retrieval + real normalization + always debate (no gate).
    Uses UNKNOWN conflict type (generic strategy, Task 16).
    Does NOT classify the conflict type.
    """

    SYSTEM_NAME = "standard_mad"

    def __init__(self, run_mode: RunMode = RunMode.LIVE):
        self.run_mode = run_mode
        # Reuse MADSystem infrastructure but override the debate trigger
        self._base = MADSystem(run_mode=run_mode)

    def run(self, query: str) -> ExecutionTrace:
        """Always debate regardless of evidence state (no gate)."""
        # Run the full retrieval + normalization + aggregation pipeline
        trace = self._base.solve(query)

        # Override: if the base system did NOT trigger a debate but we have usable agents,
        # force-trigger debate using UNKNOWN (generic) strategy
        if not trace.debate_triggered and trace.evidence_decision:
            decision = trace.evidence_decision
            usable = [e for e in trace.agent_evidence if e.is_usable]

            if len(usable) >= 2:
                # Build a synthetic conflict from the top two usable agents
                claim_a = usable[0].claims[0]
                claim_b = usable[1].claims[0]
                synthetic_conflict = Conflict(
                    type=ConflictType.UNKNOWN,
                    confidence=0.5,
                    explanation="Standard MAD always debates (no gate) — generic strategy",
                    claim_a=claim_a,
                    claim_b=claim_b,
                    metadata={
                        "agent_a_id": usable[0].agent_id,
                        "agent_b_id": usable[1].agent_id,
                        "forced": True,
                    },
                )

                debate_manager = get_debate_manager(
                    self._base.llm_client,
                    run_mode=self.run_mode.value,
                )
                debate_outcome = debate_manager.run_debate(
                    query=query,
                    evidence_list=usable,
                    conflict=synthetic_conflict,
                )

                top_conflict = {
                    "agent_a_id": usable[0].agent_id,
                    "agent_b_id": usable[1].agent_id,
                    "claim_a": claim_a,
                    "claim_b": claim_b,
                }
                judge = get_judge(self._base.llm_client, run_mode=self.run_mode.value)
                judge_result = judge.evaluate(
                    query=query,
                    evidence_list=usable,
                    debate_outcome=debate_outcome,
                    conflict_type=ConflictType.UNKNOWN,
                    top_conflict=top_conflict,
                )

                trace.debate_triggered = True
                trace.conflict = synthetic_conflict
                trace.debate_outcome = debate_outcome
                trace.debate_rounds = debate_outcome.rounds
                trace.judge_result = judge_result
                trace.metrics["debate_triggered"] = True

        trace.metrics["system"] = self.SYSTEM_NAME
        return trace
