"""
Conflict type classifier for classifying evidence disagreements.
Task 11 fixes:
1. Returns ClassificationResult with ComponentStatus (LLM_SUCCESS / FALLBACK / FAILED)
2. LLM exception → FAILED; optional rule fallback → FALLBACK
3. Tightened pattern classifier — no modal verbs or generic discourse markers
4. Entry guard: only call when decision.state == DISAGREEMENT
5. Uses top_conflict from EvidenceDecision for both input and Conflict construction
"""
import re
from typing import Optional, Dict, Any
from loguru import logger

from evidence.models import ConflictType, ClassificationResult, ComponentStatus, NormalizedClaim
from llm_client import LLMClient, LLMError
from config import settings


CONFLICT_TYPE_PROMPT = """You are a conflict type classifier. Given two conflicting claims, classify the type of conflict.

## Conflict Types

1. FACTUAL — Two sources make mutually incompatible factual claims about the same thing at the same time and in the same context.
   Example: "Water boils at 100°C" vs "Water boils at 90°C"

2. TEMPORAL — The claims differ because they refer to different points in time.
   Example: "The policy was introduced in 2019" vs "The policy was introduced in 2020"

3. VERSION — The claims refer to different software/product/document/policy versions.
   Example: "Python 3.13 supports X" vs "Python 3.14 supports Y"

4. CONTEXTUAL — The claims differ because their conditions or contexts differ (population, platform, edition).
   Example: "The drug is effective for adults" vs "The drug is not effective for children"

5. SOURCE — The disagreement is primarily caused by differences in source authority/reliability/provenance.
   Example: A blog says X vs a peer-reviewed paper says Y

## Input

Claim A: {claim_a}
Claim B: {claim_b}

Query: {query}

## Response Format

Return ONLY a JSON object:
{{
    "type": "FACTUAL|TEMPORAL|VERSION|CONTEXTUAL|SOURCE",
    "confidence": 0.0-1.0,
    "explanation": "Brief explanation of why this conflict type was chosen"
}}
"""


class ConflictClassifier:
    """Classifies conflicts between evidence claims using LLM."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client

    def classify(
        self,
        claim_a: str,
        claim_b: str,
        query: str = "",
    ) -> ClassificationResult:
        """
        Classify the type of conflict between two claims.

        Args:
            claim_a: First conflicting claim text
            claim_b: Second conflicting claim text
            query: Original user query

        Returns:
            ClassificationResult with type, confidence, explanation, and status
        """
        if not self.llm_client:
            result = self._classify_by_patterns(claim_a, claim_b, query)
            result.status = ComponentStatus.FALLBACK
            result.error = "No LLM client available"
            return result

        try:
            result = self._classify_by_llm(claim_a, claim_b, query)
            result.status = ComponentStatus.LLM_SUCCESS
            return result

        except LLMError as e:
            logger.error(f"LLM conflict classification failed: {e}")
            if settings.allow_rule_fallback:
                result = self._classify_by_patterns(claim_a, claim_b, query)
                result.status = ComponentStatus.FALLBACK
                result.error = str(e)
                return result
            return ClassificationResult(
                type=ConflictType.UNKNOWN,
                confidence=0.0,
                explanation="LLM failed and rule fallback is disabled",
                status=ComponentStatus.FAILED,
                error=str(e),
            )

    def _classify_by_llm(self, claim_a: str, claim_b: str, query: str) -> ClassificationResult:
        """Use LLM to classify conflict type. Raises LLMError on failure."""
        prompt = CONFLICT_TYPE_PROMPT.format(
            claim_a=claim_a, claim_b=claim_b, query=query,
        )
        result = self.llm_client.chat_json(prompt, role="classifier")

        conflict_type_str = result.get("type", "UNKNOWN").upper()
        valid_values = {t.value.upper(): t for t in ConflictType}
        conflict_type = valid_values.get(conflict_type_str, ConflictType.UNKNOWN)

        return ClassificationResult(
            type=conflict_type,
            confidence=float(result.get("confidence", 0.8)),
            explanation=result.get("explanation", "LLM classification"),
        )

    def _classify_by_patterns(
        self, claim_a: str, claim_b: str, query: str
    ) -> ClassificationResult:
        """
        Pattern-based fallback.
        Check order: version → temporal → scope-based contextual → source → factual.
        Does NOT fire on generic modal verbs or discourse connectives.
        """
        a = claim_a.lower()
        b = claim_b.lower()

        # VERSION — explicit version numbers or release keywords in BOTH claims
        version_pat = re.compile(r"v(?:ersion)?\s*\d+[\.\d]*|python\s*\d+[\.\d]*|\bv\d+\b|\brelease\s*\d+")
        if version_pat.search(a) and version_pat.search(b):
            return ClassificationResult(
                type=ConflictType.VERSION,
                confidence=0.85,
                explanation="Both claims contain version identifiers",
            )

        # TEMPORAL — different years appear in the two claims
        year_a = set(re.findall(r"\b(?:19|20)\d{2}\b", claim_a))
        year_b = set(re.findall(r"\b(?:19|20)\d{2}\b", claim_b))
        if year_a and year_b and year_a != year_b:
            return ClassificationResult(
                type=ConflictType.TEMPORAL,
                confidence=0.88,
                explanation=f"Claims refer to different years: {year_a} vs {year_b}",
            )

        # CONTEXTUAL — explicit population or platform qualifiers
        context_groups = [
            {"adult", "child", "children", "elderly", "pediatric", "infant"},
            {"enterprise", "consumer", "individual", "business", "personal"},
            {"windows", "linux", "macos", "android", "ios"},
            {"europe", "usa", "uk", "us", "canada", "australia"},
        ]
        for group in context_groups:
            hits_a = group & set(re.findall(r"\b\w+\b", a))
            hits_b = group & set(re.findall(r"\b\w+\b", b))
            if hits_a or hits_b:
                return ClassificationResult(
                    type=ConflictType.CONTEXTUAL,
                    confidence=0.82,
                    explanation="Claims contain different qualifying conditions or scope",
                )

        # SOURCE — explicit source-quality words
        source_pairs = [
            ("blog", "peer-reviewed"), ("blog", "study"), ("blog", "paper"),
            ("unofficial", "official"), ("anecdotal", "statistical"),
            ("reported", "confirmed"),
        ]
        for w1, w2 in source_pairs:
            if (w1 in a or w2 in a) and (w1 in b or w2 in b):
                return ClassificationResult(
                    type=ConflictType.SOURCE,
                    confidence=0.80,
                    explanation="Claims differ in source credibility or provenance",
                )

        # Default
        return ClassificationResult(
            type=ConflictType.FACTUAL,
            confidence=0.75,
            explanation="Direct factual contradiction detected",
        )


def get_conflict_classifier(llm_client=None) -> ConflictClassifier:
    """Get or create a conflict classifier instance."""
    return ConflictClassifier(llm_client=llm_client)