"""
Tests for conflict classification.
Task 29: Uses ClassificationResult objects; removes DemoLLMClient; tests ComponentStatus.
"""
import pytest
from conflict.classifier import ConflictClassifier, get_conflict_classifier
from evidence.models import ConflictType, ComponentStatus


class TestConflictClassifier:
    """Test conflict type classification (rule-based path)."""

    def test_temporal_classification(self):
        """Test temporal conflict classification."""
        classifier = ConflictClassifier()
        result = classifier.classify(
            claim_a="The policy was introduced in 2019",
            claim_b="The policy was introduced in 2020",
            query="When was the policy introduced?",
        )
        assert result.type == ConflictType.TEMPORAL
        assert result.confidence > 0.8

    def test_version_classification(self):
        """Test version conflict classification."""
        classifier = ConflictClassifier()
        result = classifier.classify(
            claim_a="Version 1.0 supports Python 3.8",
            claim_b="Version 2.0 supports Python 3.10",
            query="Which Python versions are supported?",
        )
        assert result.type == ConflictType.VERSION
        assert result.confidence > 0.8

    def test_contextual_classification(self):
        """Test contextual conflict classification."""
        classifier = ConflictClassifier()
        result = classifier.classify(
            claim_a="The drug is effective for adults",
            claim_b="The drug is not effective for children",
            query="Is the drug effective?",
        )
        assert result.type == ConflictType.CONTEXTUAL
        assert result.confidence > 0.7

    def test_factual_classification(self):
        """Test factual conflict classification — no year, no version, no scope."""
        classifier = ConflictClassifier()
        result = classifier.classify(
            claim_a="The capital of France is Paris",
            claim_b="The capital of France is Lyon",
            query="What is the capital of France?",
        )
        assert result.type == ConflictType.FACTUAL
        assert result.confidence > 0.7

    def test_classification_result_structure(self):
        """Test that ClassificationResult has correct fields."""
        classifier = ConflictClassifier()
        result = classifier.classify(
            claim_a="Test claim A",
            claim_b="Test claim B",
            query="Test query",
        )
        assert hasattr(result, "type")
        assert hasattr(result, "confidence")
        assert hasattr(result, "explanation")
        assert hasattr(result, "status")
        assert isinstance(result.type, ConflictType)
        assert 0 <= result.confidence <= 1

    def test_no_llm_gives_fallback_status(self):
        """No LLM client → status should be FALLBACK."""
        classifier = ConflictClassifier(llm_client=None)
        result = classifier.classify("A", "B", "q")
        assert result.status == ComponentStatus.FALLBACK


class TestConflictClassifierEdgeCases:
    """Test edge cases for conflict classification."""

    def test_empty_claims(self):
        """Test classification with empty claims returns valid ConflictType."""
        classifier = ConflictClassifier()
        result = classifier.classify("", "", "Test query")
        assert result.type in list(ConflictType)

    def test_long_claims(self):
        """Test classification with long claims."""
        classifier = ConflictClassifier()
        long_claim_a = "This is a very long claim. " * 100
        long_claim_b = "This is another very long claim. " * 100
        result = classifier.classify(long_claim_a, long_claim_b, "Test query")
        assert result.type in list(ConflictType)

    def test_non_english_text_with_years(self):
        """Test classification with non-English text containing years (temporal)."""
        classifier = ConflictClassifier()
        result = classifier.classify(
            "2019年に導入された政策",
            "2020年に導入された政策",
            "政策はいつ導入されたか？",
        )
        # Should still return a valid type
        assert result.type in list(ConflictType)

    def test_pattern_order_version_beats_temporal(self):
        """Version check should fire before temporal when both match."""
        classifier = ConflictClassifier()
        result = classifier.classify(
            claim_a="Python 3.10 was released in 2021",
            claim_b="Python 3.11 was released in 2022",
            query="When were Python releases?",
        )
        # Has explicit version numbers — should be VERSION, not TEMPORAL
        assert result.type == ConflictType.VERSION


class TestConflictClassifierWithLLM:
    """Tests with a real LLM client (requires API key, marked slow)."""

    @pytest.mark.slow
    def test_llm_classification_returns_llm_success_status(self):
        """LLM classification should return ComponentStatus.LLM_SUCCESS."""
        from llm_client import LLMClient
        llm_client = LLMClient()
        classifier = ConflictClassifier(llm_client)
        result = classifier.classify(
            claim_a="The event happened in 2022",
            claim_b="The event happened in 2024",
            query="When did the event happen?",
        )
        assert result.status == ComponentStatus.LLM_SUCCESS
        assert result.type in list(ConflictType)