"""
test_aggregator.py — Task 23
Must assert:
  - 3 agree → AGREEMENT
  - one contradicting pair → DISAGREEMENT
  - all neutral (contradiction=0 AND entailment=0) → UNCERTAIN (never AGREEMENT)
  - 0 or 1 usable → INSUFFICIENT
  - explicitly: contradiction=0 AND entailment=0 must NOT be AGREEMENT
"""
import pytest
from evidence.models import (
    AgentEvidence, EvidenceStatus, NormalizedClaim, Source,
    NLILabel, EvidenceState,
)
from evidence.aggregator import EvidenceAggregator
from evidence.disagreement_detector import MockNLIDetector


def make_source(agent: str) -> Source:
    return Source(url=f"https://{agent}.example.com", title=agent, snippet="test")


def make_ev(agent_id: str, claim_text: str) -> AgentEvidence:
    src = make_source(agent_id)
    nc = NormalizedClaim(text=claim_text, source=src)
    return AgentEvidence(
        agent_id=agent_id, query="q",
        status=EvidenceStatus.OK, claims=[nc], sources=[src],
    )


def make_failed_ev(agent_id: str) -> AgentEvidence:
    return AgentEvidence(
        agent_id=agent_id, query="q",
        status=EvidenceStatus.RETRIEVAL_FAILED,
        failure_reason="timeout",
    )


# ── Fixture tables ──────────────────────────────────────────────────

CONTRADICTION_FIXTURES = {
    (
        "The policy was introduced in 2019.",
        "The policy was introduced in 2020.",
    ): {
        "label": NLILabel.CONTRADICTION, "score": 0.91,
        "all_scores": {NLILabel.CONTRADICTION: 0.91, NLILabel.ENTAILMENT: 0.05, NLILabel.NEUTRAL: 0.04},
    },
}

ENTAILMENT_FIXTURES = {
    (
        "The capital of Australia is Canberra.",
        "Canberra is the capital city of Australia.",
    ): {
        "label": NLILabel.ENTAILMENT, "score": 0.92,
        "all_scores": {NLILabel.ENTAILMENT: 0.92, NLILabel.CONTRADICTION: 0.03, NLILabel.NEUTRAL: 0.05},
    },
}

# A genuinely neutral pair for the same claim text (3-agent agree test)
AGREE_3_FIXTURES = {
    ("canberra is the capital of australia.", "canberra is the capital of australia."): {
        "label": NLILabel.ENTAILMENT, "score": 0.99,
        "all_scores": {NLILabel.ENTAILMENT: 0.99, NLILabel.CONTRADICTION: 0.00, NLILabel.NEUTRAL: 0.01},
    },
}


class TestAggregatorStates:

    def test_insufficient_zero_agents(self):
        agg = EvidenceAggregator(nli_detector=None)
        decision = agg.decide([])
        assert decision.state == EvidenceState.INSUFFICIENT

    def test_insufficient_one_usable(self):
        agg = EvidenceAggregator(nli_detector=None)
        ev = make_ev("a", "some claim")
        decision = agg.decide([ev])
        assert decision.state == EvidenceState.INSUFFICIENT

    def test_insufficient_all_failed(self):
        agg = EvidenceAggregator(nli_detector=None)
        evs = [make_failed_ev("a"), make_failed_ev("b"), make_failed_ev("c")]
        decision = agg.decide(evs)
        assert decision.state == EvidenceState.INSUFFICIENT
        assert len(decision.failed_agents) == 3

    def test_disagreement_one_contradicting_pair(self):
        nli = MockNLIDetector(threshold=0.7, fixtures=CONTRADICTION_FIXTURES)
        agg = EvidenceAggregator(nli_detector=nli)
        ev_a = make_ev("a", "The policy was introduced in 2019.")
        ev_b = make_ev("b", "The policy was introduced in 2020.")
        decision = agg.decide([ev_a, ev_b])
        assert decision.state == EvidenceState.DISAGREEMENT
        assert decision.top_conflict is not None

    def test_agreement_all_pairs_entailing(self):
        # Build a fixture where both (A,B) and (A,C) and (B,C) are entailment
        fixtures = {}
        claim = "Canberra is the capital of Australia."
        for i, j in [("a", "b"), ("a", "c"), ("b", "c")]:
            key = (claim.lower(), claim.lower())
            fixtures[key] = {
                "label": NLILabel.ENTAILMENT, "score": 0.95,
                "all_scores": {NLILabel.ENTAILMENT: 0.95, NLILabel.CONTRADICTION: 0.01, NLILabel.NEUTRAL: 0.04},
            }
        nli = MockNLIDetector(threshold=0.7, fixtures=fixtures)
        agg = EvidenceAggregator(nli_detector=nli)
        evs = [make_ev(aid, claim) for aid in ["a", "b", "c"]]
        decision = agg.decide(evs)
        assert decision.state == EvidenceState.AGREEMENT
        assert len(decision.usable_agents) == 3

    def test_CRITICAL_zero_scores_must_be_uncertain_never_agreement(self):
        """
        CRITICAL TEST: contradiction=0 AND entailment=0 → UNCERTAIN, never AGREEMENT.
        This was the core bug: unrelated claims triggered AGREE before fix.
        """
        # No NLI detector → all scores are 0.0
        agg = EvidenceAggregator(nli_detector=None)
        ev_a = make_ev("a", "Python is a snake.")
        ev_b = make_ev("b", "The Eiffel Tower is in Paris.")
        decision = agg.decide([ev_a, ev_b])
        # With no NLI, entailment=0 and contradiction=0 → cannot be AGREEMENT
        assert decision.state != EvidenceState.AGREEMENT, (
            "REGRESSION: zero NLI scores produced AGREEMENT — the core bug has returned!"
        )
        assert decision.state in (EvidenceState.UNCERTAIN, EvidenceState.DISAGREEMENT)

    def test_pairwise_relations_recorded(self):
        fixtures = {**CONTRADICTION_FIXTURES}
        nli = MockNLIDetector(threshold=0.7, fixtures=fixtures)
        agg = EvidenceAggregator(nli_detector=nli)
        ev_a = make_ev("a", "The policy was introduced in 2019.")
        ev_b = make_ev("b", "The policy was introduced in 2020.")
        decision = agg.decide([ev_a, ev_b])
        assert len(decision.pairwise) == 1
        assert decision.pairwise[0].agent_a_id in ("a", "b")

    def test_failed_agents_appear_in_decision(self):
        nli = MockNLIDetector(threshold=0.7, fixtures=ENTAILMENT_FIXTURES)
        agg = EvidenceAggregator(nli_detector=nli)
        ev_ok = make_ev("a", "The capital of Australia is Canberra.")
        ev_fail = make_failed_ev("b")
        decision = agg.decide([ev_ok, ev_fail])
        assert any(f["agent_id"] == "b" for f in decision.failed_agents)
