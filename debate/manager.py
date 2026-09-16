"""
Debate manager for orchestrating the Propose → Critique → Revise cycle.
Task 14: Per-stage validation; DebateOutcome with status/stop_reason/failed_stages.
Task 30: Early stopping (converged / no_objection / max_rounds).
"""
import re
from typing import List, Optional
from datetime import datetime
from loguru import logger

from evidence.models import (
    AgentEvidence, ConflictType, Conflict, DebateArgument,
    DebateRound, DebateOutcome,
)
from debate.proposer import Proposer, is_valid_argument
from debate.critic import Critic
from debate.reviser import Reviser
from llm_client import LLMClient, LLMError
from config import settings


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity between two strings (word-set level)."""
    sa = set(re.findall(r"\b\w+\b", a.lower()))
    sb = set(re.findall(r"\b\w+\b", b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _has_no_objection(critiques: List[DebateArgument]) -> bool:
    """Return True if the critic signalled no substantive objection."""
    for c in critiques:
        if c.strategy_used and "no_objection" in c.strategy_used:
            return True
        # Also check content directly
        if re.search(r"SUBSTANTIVE_OBJECTION:\s*false", c.content, re.I):
            return True
    return False


class DebateManager:
    """
    Manages the debate process.
    Runs up to max_rounds of Propose → Critique → Revise with early stopping.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_rounds: Optional[int] = None,
        run_mode: str = "live",
    ):
        self.llm_client = llm_client
        self.max_rounds = max_rounds or settings.max_debate_rounds
        self.run_mode = run_mode
        self.proposer = Proposer(llm_client, run_mode=run_mode)
        self.critic = Critic(llm_client, run_mode=run_mode)
        self.reviser = Reviser(llm_client, run_mode=run_mode)

    def run_debate(
        self,
        query: str,
        evidence_list: List[AgentEvidence],
        conflict: Conflict,
    ) -> DebateOutcome:
        """
        Run the full debate process.

        Args:
            query: Original user query
            evidence_list: All usable AgentEvidence objects
            conflict: The detected Conflict (guarantees top_conflict fields)

        Returns:
            DebateOutcome with status, stop_reason, failed_stages, and rounds
        """
        top_conflict = {
            "agent_a_id": conflict.claim_a.source.metadata.get("agent_id", "agent_a"),
            "agent_b_id": conflict.claim_b.source.metadata.get("agent_id", "agent_b"),
            "claim_a": conflict.claim_a,
            "claim_b": conflict.claim_b,
        }

        logger.info(
            f"Starting debate | conflict_type={conflict.type.value} | "
            f"max_rounds={self.max_rounds}"
        )

        rounds: List[DebateRound] = []
        failed_stages: List[str] = []
        prev_revision_content: Optional[str] = None

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"Debate Round {round_num}/{self.max_rounds}")
            debate_round = DebateRound(
                round_number=round_num,
                started_at=datetime.now(),
            )

            # PROPOSE
            try:
                proposal = self.proposer.propose(
                    query=query,
                    evidence_list=evidence_list,
                    conflict_type=conflict.type,
                    conflict_explanation=conflict.explanation,
                    top_conflict=top_conflict,
                    round_number=round_num,
                )
                if not is_valid_argument(proposal):
                    raise LLMError("Proposer returned invalid/empty content")
                debate_round.proposer_argument = proposal
            except LLMError as e:
                logger.error(f"Round {round_num} stage failure [proposer]: {e}")
                failed_stages.append("proposer")
                rounds.append(debate_round)
                return DebateOutcome(
                    rounds=rounds,
                    status="failed" if round_num == 1 else "partial",
                    stop_reason="stage_failure",
                    failed_stages=failed_stages,
                    rounds_used=round_num - 1,
                )

            # CRITIQUE
            try:
                critiques = self.critic.critique(
                    query=query,
                    proposal=proposal,
                    evidence_list=evidence_list,
                    conflict_type=conflict.type,
                    top_conflict=top_conflict,
                    round_number=round_num,
                )
                debate_round.critic_arguments = critiques
            except LLMError as e:
                logger.error(f"Round {round_num} stage failure [critic]: {e}")
                failed_stages.append("critic")
                rounds.append(debate_round)
                return DebateOutcome(
                    rounds=rounds,
                    status="failed" if round_num == 1 else "partial",
                    stop_reason="stage_failure",
                    failed_stages=failed_stages,
                    rounds_used=round_num - 1,
                )

            # Early stop: no substantive objection (Task 30)
            if _has_no_objection(critiques):
                debate_round.completed_at = datetime.now()
                rounds.append(debate_round)
                logger.info(f"Early stop at round {round_num}: no_objection")
                return DebateOutcome(
                    rounds=rounds,
                    status="completed",
                    stop_reason="no_objection",
                    failed_stages=[],
                    rounds_used=round_num,
                )

            # REVISE
            try:
                revision = self.reviser.revise(
                    query=query,
                    proposal=proposal,
                    critiques=critiques,
                    evidence_list=evidence_list,
                    conflict_type=conflict.type,
                    top_conflict=top_conflict,
                    round_number=round_num,
                )
                if not is_valid_argument(revision):
                    raise LLMError("Reviser returned invalid/empty content")
                debate_round.reviser_argument = revision
            except LLMError as e:
                logger.error(f"Round {round_num} stage failure [reviser]: {e}")
                failed_stages.append("reviser")
                rounds.append(debate_round)
                return DebateOutcome(
                    rounds=rounds,
                    status="failed" if round_num == 1 else "partial",
                    stop_reason="stage_failure",
                    failed_stages=failed_stages,
                    rounds_used=round_num - 1,
                )

            debate_round.completed_at = datetime.now()
            rounds.append(debate_round)
            logger.info(f"Round {round_num} completed")

            # Early stop: convergence (Task 30)
            if prev_revision_content is not None:
                sim = _jaccard(prev_revision_content, revision.content)
                if sim >= 0.95:
                    logger.info(f"Early stop at round {round_num}: converged (Jaccard={sim:.3f})")
                    return DebateOutcome(
                        rounds=rounds,
                        status="completed",
                        stop_reason="converged",
                        failed_stages=[],
                        rounds_used=round_num,
                    )
            prev_revision_content = revision.content

        logger.info(f"Debate completed: {len(rounds)} rounds (max_rounds)")
        return DebateOutcome(
            rounds=rounds,
            status="completed",
            stop_reason="max_rounds",
            failed_stages=[],
            rounds_used=len(rounds),
        )


def get_debate_manager(
    llm_client=None,
    max_rounds: Optional[int] = None,
    run_mode: str = "live",
) -> DebateManager:
    """Get or create a debate manager instance."""
    return DebateManager(llm_client=llm_client, max_rounds=max_rounds, run_mode=run_mode)