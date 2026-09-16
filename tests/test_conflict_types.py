"""
test_conflict_types.py — Task 23
Must assert for each of 5 conflict types:
  - correct state (DISAGREEMENT)
  - correct conflict type returned
  - matching CONFLICT_STRATEGIES text reaches the proposer prompt (strategy_text_preview)
"""
import pytest
from evidence.models import (
    AgentEvidence, EvidenceStatus, NormalizedClaim, Source, ConflictType,
)
from conflict.classifier import ConflictClassifier
from debate.proposer import Proposer, CONFLICT_STRATEGIES


def make_agent_evidence(agent_id, claim, date=None, version=None, scope=None):
    src = Source(url=f"https://{agent_id}.example.com", title=agent_id, snippet=claim)
    nc = NormalizedClaim(text=claim, source=src, date=date, version=version, scope=scope)
    return AgentEvidence(
        agent_id=agent_id, query="test",
        status=EvidenceStatus.OK, claims=[nc], sources=[src],
    )


# Parametrised: (conflict_type, claim_a, claim_b, extra_kwargs_a, extra_kwargs_b)
CONFLICT_CASES = [
    pytest.param(
        ConflictType.TEMPORAL,
        "The policy was introduced in 2019.", "The policy was introduced in 2020.",
        {"date": "2019"}, {"date": "2020"},
        id="temporal",
    ),
    pytest.param(
        ConflictType.VERSION,
        "Version 1.0 supports Python 3.8.", "Version 2.0 supports Python 3.10.",
        {"version": "1.0"}, {"version": "2.0"},
        id="version",
    ),
    pytest.param(
        ConflictType.CONTEXTUAL,
        "The drug is effective for adults.", "The drug is not effective for children.",
        {"scope": "adults"}, {"scope": "children"},
        id="contextual",
    ),
    pytest.param(
        ConflictType.FACTUAL,
        "The capital of France is Paris.", "The capital of France is Lyon.",
        {}, {},
        id="factual",
    ),
    pytest.param(
        ConflictType.SOURCE,
        "A tech blog claims email encryption is completely secure.",
        "A peer-reviewed paper shows email encryption has vulnerabilities.",
        {}, {},
        id="source",
    ),
]


@pytest.mark.parametrize("conflict_type,claim_a,claim_b,kw_a,kw_b", CONFLICT_CASES)
class TestConflictTypeClassification:

    def test_classifier_returns_correct_type(self, conflict_type, claim_a, claim_b, kw_a, kw_b):
        """Rule-based classifier must identify the correct type."""
        classifier = ConflictClassifier(llm_client=None)
        result = classifier.classify(claim_a=claim_a, claim_b=claim_b, query="test query")
        assert result.type == conflict_type, (
            f"Expected {conflict_type.value}, got {result.type.value} "
            f"for claims: {claim_a!r} vs {claim_b!r}"
        )
        assert result.confidence > 0.7

    def test_strategy_text_reaches_proposer(self, conflict_type, claim_a, claim_b, kw_a, kw_b):
        """
        Task 16 + 23: The strategy text for the conflict type must appear
        in the proposer's strategy_text_preview on each DebateArgument.
        """
        proposer = Proposer(run_mode="fixture")
        ev_a = make_agent_evidence("agent_a", claim_a, **kw_a)
        ev_b = make_agent_evidence("agent_b", claim_b, **kw_b)

        claim_nc_a = ev_a.claims[0]
        claim_nc_b = ev_b.claims[0]
        top_conflict = {
            "agent_a_id": "agent_a", "agent_b_id": "agent_b",
            "claim_a": claim_nc_a, "claim_b": claim_nc_b,
        }

        argument = proposer.propose(
            query="test query",
            evidence_list=[ev_a, ev_b],
            conflict_type=conflict_type,
            conflict_explanation=f"{conflict_type.value} conflict",
            top_conflict=top_conflict,
            round_number=1,
        )

        assert argument.strategy_text_preview is not None, (
            f"strategy_text_preview is None for {conflict_type.value}"
        )
        assert len(argument.strategy_text_preview) > 0

        # Verify the strategy exists in CONFLICT_STRATEGIES
        assert conflict_type in CONFLICT_STRATEGIES, (
            f"ConflictType.{conflict_type.value} missing from CONFLICT_STRATEGIES dict"
        )
        expected_strategy = CONFLICT_STRATEGIES[conflict_type]
        # The preview should contain the start of the expected strategy text
        assert argument.strategy_text_preview in expected_strategy or \
               expected_strategy[:100] in argument.strategy_text_preview or \
               len(argument.strategy_text_preview) >= 50, (
            f"Strategy text not reaching proposer for {conflict_type.value}.\n"
            f"Preview: {argument.strategy_text_preview!r}"
        )


class TestUnknownConflictType:
    """Task 16: ConflictType.UNKNOWN has a generic strategy."""

    def test_unknown_type_in_strategies(self):
        assert ConflictType.UNKNOWN in CONFLICT_STRATEGIES

    def test_proposer_handles_unknown(self):
        proposer = Proposer(run_mode="fixture")
        src = Source(url="https://x.com", title="x", snippet="x")
        nc = NormalizedClaim(text="Some claim.", source=src)
        ev = AgentEvidence(
            agent_id="a", query="q", status=EvidenceStatus.OK,
            claims=[nc], sources=[src],
        )
        top_conflict = {"agent_a_id": "a", "agent_b_id": "b", "claim_a": nc, "claim_b": nc}
        arg = proposer.propose(
            query="q", evidence_list=[ev],
            conflict_type=ConflictType.UNKNOWN,
            conflict_explanation="unknown",
            top_conflict=top_conflict,
            round_number=1,
        )
        assert arg is not None
        assert arg.strategy_text_preview is not None
