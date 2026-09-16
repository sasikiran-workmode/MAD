"""
Tests for evidence models, NLI detector, aggregator, and comparator.
Task 29: Tests use MockNLIDetector with exact-match fixture table.
         No word-overlap heuristic pairs allowed.
"""
import pytest
from evidence.models import (
    Source, Claim, NormalizedClaim, Evidence, AgentEvidence, EvidenceStatus,
    EvidenceComparison, Conflict, ConflictType, NLILabel, EvidenceState,
)
from evidence.comparator import EvidenceComparator
from evidence.aggregator import EvidenceAggregator
from evidence.disagreement_detector import MockNLIDetector, NLIError, get_nli_detector


# ------------------------------------------------------------------ #
# Shared fixtures                                                      #
# ------------------------------------------------------------------ #

@pytest.fixture
def sample_source():
    return Source(url="https://example.com/article", title="Test Article", snippet="Snippet")


@pytest.fixture
def mock_nli():
    """MockNLIDetector with all test pairs registered."""
    extra = {
        (
            "the policy was introduced in 2019.",
            "the policy was introduced in 2020.",
        ): {
            "label": NLILabel.CONTRADICTION,
            "score": 0.91,
            "all_scores": {
                NLILabel.CONTRADICTION: 0.91,
                NLILabel.ENTAILMENT: 0.05,
                NLILabel.NEUTRAL: 0.04,
            },
        },
        (
            "the capital of australia is canberra.",
            "canberra is the capital city of australia.",
        ): {
            "label": NLILabel.ENTAILMENT,
            "score": 0.92,
            "all_scores": {
                NLILabel.ENTAILMENT: 0.92,
                NLILabel.CONTRADICTION: 0.03,
                NLILabel.NEUTRAL: 0.05,
            },
        },
        (
            "python is a programming language.",
            "the weather today is sunny.",
        ): {
            "label": NLILabel.NEUTRAL,
            "score": 0.88,
            "all_scores": {
                NLILabel.NEUTRAL: 0.88,
                NLILabel.ENTAILMENT: 0.06,
                NLILabel.CONTRADICTION: 0.06,
            },
        },
    }
    return MockNLIDetector(threshold=0.7, fixtures=extra)


# ------------------------------------------------------------------ #
# Model tests                                                          #
# ------------------------------------------------------------------ #

class TestEvidenceModels:
    """Test evidence data models."""

    def test_source_creation(self, sample_source):
        assert sample_source.url == "https://example.com/article"
        assert sample_source.title == "Test Article"

    def test_claim_creation(self, sample_source):
        claim = Claim(text="This is a test claim", source=sample_source)
        assert claim.text == "This is a test claim"
        assert claim.source.url == sample_source.url
        assert claim.id is not None

    def test_normalized_claim_creation(self, sample_source):
        nc = NormalizedClaim(
            text="The policy was introduced in 2019.",
            source=sample_source,
            date="2019",
        )
        assert nc.text == "The policy was introduced in 2019."
        assert nc.date == "2019"

    def test_evidence_creation(self, sample_source):
        claim = Claim(text="Test claim", source=sample_source)
        ev = Evidence(
            agent_id="agent_1",
            query="What is testing?",
            answer="Testing is important",
            claims=[claim],
            sources=[sample_source],
        )
        assert ev.agent_id == "agent_1"
        assert len(ev.claims) == 1

    def test_agent_evidence_usable(self, sample_source):
        nc = NormalizedClaim(text="The capital of Australia is Canberra.", source=sample_source)
        ae = AgentEvidence(
            agent_id="agent_a",
            query="capital of australia",
            status=EvidenceStatus.OK,
            claims=[nc],
            sources=[sample_source],
        )
        assert ae.is_usable is True

    def test_agent_evidence_not_usable_no_claims(self, sample_source):
        ae = AgentEvidence(
            agent_id="agent_a",
            query="capital",
            status=EvidenceStatus.OK,
            claims=[],
            sources=[],
        )
        assert ae.is_usable is False

    def test_agent_evidence_not_usable_failed(self, sample_source):
        ae = AgentEvidence(
            agent_id="agent_b",
            query="q",
            status=EvidenceStatus.RETRIEVAL_FAILED,
            failure_reason="timeout",
        )
        assert ae.is_usable is False

    def test_evidence_comparison(self):
        comparison = EvidenceComparison(
            agent_a_id="agent_a",
            agent_b_id="agent_b",
            has_disagreement=True,
            contradiction_score=0.85,
        )
        assert comparison.has_disagreement is True
        assert comparison.contradiction_score == 0.85

    def test_conflict_types(self):
        assert ConflictType.FACTUAL.value == "factual"
        assert ConflictType.TEMPORAL.value == "temporal"
        assert ConflictType.VERSION.value == "version"
        assert ConflictType.CONTEXTUAL.value == "contextual"
        assert ConflictType.SOURCE.value == "source"

    def test_nli_labels(self):
        assert NLILabel.ENTAILMENT.value == "entailment"
        assert NLILabel.CONTRADICTION.value == "contradiction"
        assert NLILabel.NEUTRAL.value == "neutral"


# ------------------------------------------------------------------ #
# MockNLIDetector tests (exact-match fixture)                         #
# ------------------------------------------------------------------ #

class TestMockNLIDetector:
    """Test the Mock NLI detector using the built-in exact-match fixture table."""

    def test_contradiction_detection(self, mock_nli):
        """Test detection of contradictions from fixture table."""
        result = mock_nli.predict(
            "The policy was introduced in 2019.",
            "The policy was introduced in 2020.",
        )
        assert result["label"] == NLILabel.CONTRADICTION
        assert result["score"] > 0.8

    def test_entailment_detection(self, mock_nli):
        """Test detection of entailment from fixture table."""
        result = mock_nli.predict(
            "The capital of Australia is Canberra.",
            "Canberra is the capital city of Australia.",
        )
        assert result["label"] == NLILabel.ENTAILMENT
        assert result["score"] > 0.8

    def test_neutral_detection(self, mock_nli):
        """Test detection of neutral relationships from fixture table."""
        result = mock_nli.predict(
            "Python is a programming language.",
            "The weather today is sunny.",
        )
        assert result["label"] == NLILabel.NEUTRAL

    def test_unknown_pair_raises(self, mock_nli):
        """Unknown pairs must raise KeyError — NOT return a fake NEUTRAL."""
        with pytest.raises(KeyError):
            mock_nli.predict("some random claim", "another random claim")

    def test_disagreement_detection(self, mock_nli):
        """Test the detect_disagreement helper."""
        result = mock_nli.detect_disagreement(
            "The policy was introduced in 2019.",
            "The policy was introduced in 2020.",
        )
        assert result["disagreement"] is True
        assert result["contradiction_score"] >= 0.7
        assert "threshold" in result


# ------------------------------------------------------------------ #
# EvidenceAggregator tests                                             #
# ------------------------------------------------------------------ #

class TestEvidenceAggregator:
    """Test N-agent aggregation."""

    def _make_agent_evidence(self, agent_id, claim_text, source):
        nc = NormalizedClaim(text=claim_text, source=source)
        return AgentEvidence(
            agent_id=agent_id,
            query="test",
            status=EvidenceStatus.OK,
            claims=[nc],
            sources=[source],
        )

    def test_insufficient_one_agent(self, sample_source):
        agg = EvidenceAggregator(nli_detector=None)
        ev = self._make_agent_evidence("a", "some claim", sample_source)
        decision = agg.decide([ev])
        assert decision.state == EvidenceState.INSUFFICIENT

    def test_insufficient_zero_agents(self, sample_source):
        agg = EvidenceAggregator(nli_detector=None)
        decision = agg.decide([])
        assert decision.state == EvidenceState.INSUFFICIENT

    def test_disagreement(self, mock_nli, sample_source):
        agg = EvidenceAggregator(nli_detector=mock_nli)
        ev_a = self._make_agent_evidence(
            "agent_a", "The policy was introduced in 2019.", sample_source
        )
        ev_b = self._make_agent_evidence(
            "agent_b", "The policy was introduced in 2020.", sample_source
        )
        decision = agg.decide([ev_a, ev_b])
        assert decision.state == EvidenceState.DISAGREEMENT
        assert decision.top_conflict is not None

    def test_agreement(self, mock_nli, sample_source):
        agg = EvidenceAggregator(nli_detector=mock_nli)
        ev_a = self._make_agent_evidence(
            "agent_a", "The capital of Australia is Canberra.", sample_source
        )
        ev_b = self._make_agent_evidence(
            "agent_b", "Canberra is the capital city of Australia.", sample_source
        )
        decision = agg.decide([ev_a, ev_b])
        assert decision.state == EvidenceState.AGREEMENT

    def test_usable_agents_listed(self, mock_nli, sample_source):
        agg = EvidenceAggregator(nli_detector=mock_nli)
        ev_a = self._make_agent_evidence(
            "agent_a", "The capital of Australia is Canberra.", sample_source
        )
        ev_b = self._make_agent_evidence(
            "agent_b", "Canberra is the capital city of Australia.", sample_source
        )
        decision = agg.decide([ev_a, ev_b])
        assert "agent_a" in decision.usable_agents
        assert "agent_b" in decision.usable_agents


# ------------------------------------------------------------------ #
# Legacy EvidenceComparator tests (kept for backward compat)          #
# ------------------------------------------------------------------ #

class TestEvidenceComparator:
    """Test the legacy EvidenceComparator (two-agent only)."""

    def test_comparator_disagreement(self, mock_nli):
        comparator = EvidenceComparator(mock_nli)
        source_a = Source(url="https://example.com/a", title="A", snippet="A")
        source_b = Source(url="https://example.com/b", title="B", snippet="B")

        ev_a = Evidence(
            agent_id="agent_a", query="q", answer="A",
            claims=[Claim(text="The policy was introduced in 2019.", source=source_a)],
            sources=[source_a],
        )
        ev_b = Evidence(
            agent_id="agent_b", query="q", answer="B",
            claims=[Claim(text="The policy was introduced in 2020.", source=source_b)],
            sources=[source_b],
        )

        comparison = comparator.compare(ev_a, ev_b)
        assert comparison.has_disagreement is True
        assert comparison.contradiction_score >= 0.7


# ------------------------------------------------------------------ #
# Real NLI detector (slow, integration tests)                         #
# ------------------------------------------------------------------ #

class TestRealNLIDetector:
    """Test real NLI detector (requires model loading)."""

    @pytest.mark.slow
    def test_detector_loading(self):
        detector = get_nli_detector()
        assert detector is not None

    @pytest.mark.slow
    def test_real_prediction(self):
        detector = get_nli_detector()
        result = detector.predict(
            "The sky is blue.",
            "The sky is green.",
        )
        assert "label" in result
        assert "score" in result
        assert result["label"] in [NLILabel.CONTRADICTION, NLILabel.NEUTRAL, NLILabel.ENTAILMENT]

    @pytest.mark.slow
    def test_correct_input_format(self):
        """Task 7 verification: contradiction pair should score > 0.5."""
        detector = get_nli_detector()
        result = detector.predict(
            "The policy was introduced in 2019.",
            "The policy was introduced in 2020.",
        )
        contradiction_score = result["all_scores"].get(NLILabel.CONTRADICTION, 0.0)
        assert contradiction_score > 0.5, (
            f"Expected contradiction score > 0.5, got {contradiction_score}. "
            "Check NLI input format (text/text_pair, not manual </s> concatenation)."
        )