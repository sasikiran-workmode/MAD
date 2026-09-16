"""
N-agent evidence aggregator (Task 9).

Decision rule (written once here, not in app.py):
  usable = [e for e in agent_evidence if e.is_usable]

  if len(usable) <= 1                               → INSUFFICIENT
  for each unordered pair: relation = nli(claim_i, claim_j)
  if any pair contradiction >= DISAGREEMENT_THRESHOLD → DISAGREEMENT
  elif all pairs entailment >= ENTAILMENT_THRESHOLD  → AGREEMENT
  else                                               → UNCERTAIN

The asymmetry is deliberate: disagreement needs ONE pair, agreement needs ALL.
Two agents entailing + one neutral → UNCERTAIN, never AGREEMENT.
"""
from itertools import combinations
from typing import List, Optional
from loguru import logger

from evidence.models import (
    AgentEvidence, EvidenceState, EvidenceDecision,
    PairwiseRelation, NLILabel,
)
from evidence.disagreement_detector import NLIError
from config import settings


class EvidenceAggregator:
    """Runs NLI over all agent-pairs and returns an EvidenceDecision."""

    def __init__(self, nli_detector=None):
        self.nli_detector = nli_detector
        self.disagreement_threshold = settings.disagreement_threshold
        self.entailment_threshold = settings.entailment_threshold

    def decide(self, agent_evidence: List[AgentEvidence]) -> EvidenceDecision:
        """
        Aggregate evidence from N agents into a single EvidenceDecision.

        Args:
            agent_evidence: list of AgentEvidence (one per agent, any status)

        Returns:
            EvidenceDecision with state, pairwise NLI results, and top conflict
        """
        usable = [e for e in agent_evidence if e.is_usable]
        failed = [
            {"agent_id": e.agent_id, "reason": e.failure_reason or e.status.value}
            for e in agent_evidence
            if not e.is_usable
        ]

        if len(usable) <= 1:
            return EvidenceDecision(
                state=EvidenceState.INSUFFICIENT,
                pairwise=[],
                usable_agents=[e.agent_id for e in usable],
                failed_agents=failed,
                rationale=(
                    f"Only {len(usable)} usable agent(s) — insufficient to compare."
                ),
            )

        pairwise: List[PairwiseRelation] = []
        top_conflict: Optional[dict] = None
        top_contra_score = 0.0

        for ev_a, ev_b in combinations(usable, 2):
            claim_a = ev_a.claims[0]
            claim_b = ev_b.claims[0]

            try:
                if self.nli_detector:
                    result = self.nli_detector.predict(claim_a.text, claim_b.text)
                    all_scores = result.get("all_scores", {})
                    contra = float(all_scores.get(NLILabel.CONTRADICTION, 0.0))
                    ent = float(all_scores.get(NLILabel.ENTAILMENT, 0.0))
                    neu = float(all_scores.get(NLILabel.NEUTRAL, 0.0))
                else:
                    contra = ent = neu = 0.0

                relation = PairwiseRelation(
                    agent_a_id=ev_a.agent_id,
                    agent_b_id=ev_b.agent_id,
                    claim_a_text=claim_a.text,
                    claim_b_text=claim_b.text,
                    label=result["label"] if self.nli_detector else NLILabel.NEUTRAL,
                    contradiction_score=contra,
                    entailment_score=ent,
                    neutral_score=neu,
                )
                pairwise.append(relation)

                if contra > top_contra_score:
                    top_contra_score = contra
                    top_conflict = {
                        "agent_a_id": ev_a.agent_id,
                        "agent_b_id": ev_b.agent_id,
                        "claim_a": claim_a,
                        "claim_b": claim_b,
                        "contradiction_score": contra,
                    }

            except NLIError as e:
                # NLI crashed for this pair — mark as uncertain (never agreement)
                logger.warning(f"NLI failed for pair {ev_a.agent_id}/{ev_b.agent_id}: {e}")
                relation = PairwiseRelation(
                    agent_a_id=ev_a.agent_id,
                    agent_b_id=ev_b.agent_id,
                    claim_a_text=claim_a.text,
                    claim_b_text=claim_b.text,
                    label=NLILabel.NEUTRAL,
                    contradiction_score=0.0,
                    entailment_score=0.0,
                    neutral_score=0.0,
                )
                pairwise.append(relation)

        # Apply decision rule
        any_contradiction = any(
            r.contradiction_score >= self.disagreement_threshold for r in pairwise
        )
        all_entailment = all(
            r.entailment_score >= self.entailment_threshold for r in pairwise
        )

        if any_contradiction:
            state = EvidenceState.DISAGREEMENT
            rationale = (
                f"At least one pair exceeded contradiction threshold "
                f"({self.disagreement_threshold}): score={top_contra_score:.3f}"
            )
        elif all_entailment:
            state = EvidenceState.AGREEMENT
            rationale = (
                f"All {len(pairwise)} pair(s) exceeded entailment threshold "
                f"({self.entailment_threshold})"
            )
        else:
            state = EvidenceState.UNCERTAIN
            rationale = (
                "Evidence neither clearly agreed nor conflicted — "
                "no pair crossed the contradiction threshold and "
                "not all pairs crossed the entailment threshold."
            )

        logger.info(
            f"EvidenceDecision: {state.value} | "
            f"{len(usable)} usable agents | "
            f"{len(pairwise)} pairs compared | "
            f"top_contra={top_contra_score:.3f}"
        )

        return EvidenceDecision(
            state=state,
            pairwise=pairwise,
            usable_agents=[e.agent_id for e in usable],
            failed_agents=failed,
            top_conflict=top_conflict,
            rationale=rationale,
        )
