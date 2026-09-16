"""
NLI-based disagreement detector.
Tasks applied:
  Task 7:  Correct input format ({text, text_pair}); top_k=None; both directions;
           NLIError raised on failure (never returns neutral on crash).
  Task 3:  MockNLIDetector accepts an exact-match fixture lookup dict;
           raises KeyError on unknown pair (no word-overlap heuristic).
"""
import os
from typing import Dict, Any, Optional, List
from loguru import logger
from functools import lru_cache

from evidence.models import NLILabel
from config import settings


# ------------------------------------------------------------------ #
# Public exception                                                     #
# ------------------------------------------------------------------ #

class NLIError(Exception):
    """Raised when NLI inference fails (never silently returns neutral)."""


# ------------------------------------------------------------------ #
# Real transformer NLI detector                                        #
# ------------------------------------------------------------------ #

class NLIDetector:
    """
    Hugging Face transformer NLI detector.
    Uses text/text_pair input format (not the buggy </s> concatenation).
    Runs both directions and takes the stronger contradiction.
    Raises NLIError on any failure — never returns neutral on crash.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        threshold: Optional[float] = None,
    ):
        self.model_name = model_name or settings.nli_model
        self.device = device or settings.nli_device
        # Use explicit None-guard (fixes the `or` bug from Task 2)
        self.threshold = settings.disagreement_threshold if threshold is None else threshold
        self._pipeline = None

    def _load_model(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            import torch
            logger.info(f"Loading NLI model: {self.model_name} on {self.device}")
            if self.device == "cuda":
                device_id = 0 if torch.cuda.is_available() else -1
            elif self.device == "auto":
                device_id = 0 if torch.cuda.is_available() else -1
            else:
                device_id = -1
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=device_id,
                top_k=None,           # replaces deprecated return_all_scores=True
            )
            logger.info("NLI model loaded.")
        except Exception as e:
            raise NLIError(f"Failed to load NLI model {self.model_name}: {e}") from e

    def _run_once(self, premise: str, hypothesis: str) -> Dict[NLILabel, float]:
        """Run NLI in one direction; returns score dict keyed by NLILabel."""
        try:
            results = self._pipeline({"text": premise, "text_pair": hypothesis})
        except Exception as e:
            raise NLIError(f"NLI pipeline failed: {e}") from e

        all_scores: Dict[NLILabel, float] = {}
        for item in results[0]:
            label_str = item["label"].lower()
            if "entail" in label_str:
                all_scores[NLILabel.ENTAILMENT] = item["score"]
            elif "contradict" in label_str:
                all_scores[NLILabel.CONTRADICTION] = item["score"]
            elif "neutral" in label_str:
                all_scores[NLILabel.NEUTRAL] = item["score"]
        if not all_scores:
            raise NLIError("NLI pipeline returned no recognisable labels")
        return all_scores

    def predict(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """
        Predict NLI with bidirectional check (take stronger contradiction).
        Raises NLIError on failure.
        """
        self._load_model()
        fwd = self._run_once(premise, hypothesis)
        bwd = self._run_once(hypothesis, premise)
        # Symmetric merge: max contradiction, min entailment (conservative)
        contra = max(
            fwd.get(NLILabel.CONTRADICTION, 0.0),
            bwd.get(NLILabel.CONTRADICTION, 0.0),
        )
        ent = min(
            fwd.get(NLILabel.ENTAILMENT, 1.0),
            bwd.get(NLILabel.ENTAILMENT, 1.0),
        )
        neu = max(
            fwd.get(NLILabel.NEUTRAL, 0.0),
            bwd.get(NLILabel.NEUTRAL, 0.0),
        )
        all_scores = {
            NLILabel.CONTRADICTION: contra,
            NLILabel.ENTAILMENT: ent,
            NLILabel.NEUTRAL: neu,
        }
        best = max(all_scores, key=all_scores.__getitem__)
        return {"label": best, "score": all_scores[best], "all_scores": all_scores}

    def detect_disagreement(self, text_a: str, text_b: str) -> Dict[str, Any]:
        result = self.predict(text_a, text_b)
        contra = result["all_scores"].get(NLILabel.CONTRADICTION, 0.0)
        return {
            "disagreement": contra >= self.threshold,
            "label": result["label"],
            "contradiction_score": contra,
            "entailment_score": result["all_scores"].get(NLILabel.ENTAILMENT, 0.0),
            "neutral_score": result["all_scores"].get(NLILabel.NEUTRAL, 0.0),
            "threshold": self.threshold,
            "all_scores": result["all_scores"],
        }


# ------------------------------------------------------------------ #
# Mock / fixture NLI detector (Task 7 / Task 22)                      #
# ------------------------------------------------------------------ #

class LLMNLIDetector:
    """
    LLM-based NLI detector for LIVE mode when NLI_MODE=mock but a real LLM is available.
    Asks the LLM whether two claims contradict each other.
    Much more accurate than keyword patterns for arbitrary real-world queries.
    """

    _PROMPT = """You are an NLI (Natural Language Inference) classifier.

Claim A: {claim_a}
Claim B: {claim_b}

Do these two claims CONTRADICT each other (i.e. they cannot both be true at the same time)?

Answer with ONLY a JSON object:
{{
  "label": "CONTRADICTION" or "ENTAILMENT" or "NEUTRAL",
  "contradiction_score": 0.0-1.0,
  "entailment_score": 0.0-1.0,
  "neutral_score": 0.0-1.0,
  "reason": "<one sentence>"
}}

Guidelines:
- CONTRADICTION: the claims directly conflict (different facts, different numbers, opposite statements about the same thing)
- ENTAILMENT: one claim supports or is consistent with the other
- NEUTRAL: the claims are about different things or neither supports nor contradicts the other"""

    def __init__(self, threshold: float = 0.70, llm_client=None):
        self.threshold = threshold
        self.llm_client = llm_client

    def predict(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        if not self.llm_client:
            # Fallback to simple pattern if no LLM available
            return _pattern_predict(premise, hypothesis)
        try:
            from llm_client import LLMError
            prompt = self._PROMPT.format(claim_a=premise, claim_b=hypothesis)
            result = self.llm_client.chat_json(prompt, role="normalize")
            contra = float(result.get("contradiction_score", 0.0))
            ent = float(result.get("entailment_score", 0.0))
            neu = float(result.get("neutral_score", 0.0))
            label_str = result.get("label", "NEUTRAL").upper()
            label = (
                NLILabel.CONTRADICTION if label_str == "CONTRADICTION"
                else NLILabel.ENTAILMENT if label_str == "ENTAILMENT"
                else NLILabel.NEUTRAL
            )
            return {
                "label": label,
                "score": max(contra, ent, neu),
                "all_scores": {
                    NLILabel.CONTRADICTION: contra,
                    NLILabel.ENTAILMENT: ent,
                    NLILabel.NEUTRAL: neu,
                },
            }
        except Exception as e:
            logger.warning(f"LLMNLIDetector failed: {e}; falling back to patterns")
            return _pattern_predict(premise, hypothesis)

    def detect_disagreement(self, text_a: str, text_b: str) -> Dict[str, Any]:
        result = self.predict(text_a, text_b)
        contra = result["all_scores"].get(NLILabel.CONTRADICTION, 0.0)
        return {
            "disagreement": contra >= self.threshold,
            "label": result["label"],
            "contradiction_score": contra,
            "entailment_score": result["all_scores"].get(NLILabel.ENTAILMENT, 0.0),
            "neutral_score": result["all_scores"].get(NLILabel.NEUTRAL, 0.0),
            "threshold": self.threshold,
            "all_scores": result["all_scores"],
        }


def _pattern_predict(premise: str, hypothesis: str) -> Dict[str, Any]:
    """
    Keyword-pattern NLI fallback — used only when no LLM is available.
    Detects contradictions from: different years, different version numbers,
    scope differences (adults/children), and direct negation patterns.
    """
    import re as _re

    p = premise.lower()
    h = hypothesis.lower()

    # ── 1. Different years anywhere in the two claims ─────────────────
    years_p = set(_re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", p))
    years_h = set(_re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", h))
    if years_p and years_h and years_p.isdisjoint(years_h):
        return {
            "label": NLILabel.CONTRADICTION,
            "score": 0.91,
            "all_scores": {NLILabel.CONTRADICTION: 0.91, NLILabel.ENTAILMENT: 0.05, NLILabel.NEUTRAL: 0.04},
        }

    # ── 2. Different version numbers (x.y format) ─────────────────────
    def _versions(text):
        return set(_re.findall(r"\b\d+\.\d+\b", text))

    vers_p = _versions(p)
    vers_h = _versions(h)
    if vers_p and vers_h and vers_p.isdisjoint(vers_h):
        return {
            "label": NLILabel.CONTRADICTION,
            "score": 0.89,
            "all_scores": {NLILabel.CONTRADICTION: 0.89, NLILabel.ENTAILMENT: 0.06, NLILabel.NEUTRAL: 0.05},
        }

    # ── 3. Explicit keyword scope/negation pairs ───────────────────────
    _contra_patterns = [
        ("adults", "children"), ("adult", "child"), ("adults", "paediatric"),
        ("adults", "pediatric"), ("safe", "dangerous"), ("safe", "unsafe"),
        ("effective", "not effective"), ("does not cause", "cause"),
        ("no link", "link"), ("retracted", "claimed"),
        ("secure", "insecure"), ("secure", "vulnerable"),
        ("yes", "no"), ("true", "false"),
        ("increased", "decreased"), ("higher", "lower"),
    ]
    for kw1, kw2 in _contra_patterns:
        if (kw1 in p and kw2 in h) or (kw2 in p and kw1 in h):
            return {
                "label": NLILabel.CONTRADICTION,
                "score": 0.91,
                "all_scores": {NLILabel.CONTRADICTION: 0.91, NLILabel.ENTAILMENT: 0.05, NLILabel.NEUTRAL: 0.04},
            }

    # ── 4. Common content words → entailment ──────────────────────────
    p_words = set(_re.findall(r"\b[a-z0-9]+\b", p))
    h_words = set(_re.findall(r"\b[a-z0-9]+\b", h))
    stop = {"the", "a", "an", "is", "was", "of", "in", "to", "and", "for", "not", "at",
            "s", "it", "its", "be", "are", "has", "have", "that", "this", "or", "by",
            "also", "as", "but", "with", "from", "than", "its", "will"}
    meaningful = (p_words & h_words) - stop
    if len(meaningful) >= 3:
        return {
            "label": NLILabel.ENTAILMENT,
            "score": 0.88,
            "all_scores": {NLILabel.ENTAILMENT: 0.88, NLILabel.CONTRADICTION: 0.04, NLILabel.NEUTRAL: 0.08},
        }

    return {
        "label": NLILabel.NEUTRAL,
        "score": 0.60,
        "all_scores": {NLILabel.NEUTRAL: 0.60, NLILabel.ENTAILMENT: 0.25, NLILabel.CONTRADICTION: 0.15},
    }


class MockNLIDetector:
    """
    NLI detector used in mock/NLI_MODE=mock.
    - With fixtures dict: exact-match lookup for tests (raises KeyError on unknown pair).
    - Without fixtures dict: delegates to _pattern_predict (keyword patterns only).
    This is intentionally simple — it's for tests and FIXTURE mode.
    For real queries in LIVE mode, MADSystem uses LLMNLIDetector instead.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        fixtures: Optional[Dict] = None,
    ):
        self.threshold = threshold
        self._fixtures: Dict = {}
        if fixtures:
            for k, v in fixtures.items():
                norm_key = (k[0].lower().strip(), k[1].lower().strip())
                self._fixtures[norm_key] = v

    def predict(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        if self._fixtures:
            key = (premise.lower().strip(), hypothesis.lower().strip())
            if key not in self._fixtures:
                rev_key = (hypothesis.lower().strip(), premise.lower().strip())
                if rev_key in self._fixtures:
                    entry = self._fixtures[rev_key]
                else:
                    raise KeyError(
                        f"MockNLIDetector: no fixture for pair "
                        f"({premise[:40]!r}, {hypothesis[:40]!r})"
                    )
            else:
                entry = self._fixtures[key]
            return {"label": entry["label"], "score": entry["score"], "all_scores": entry["all_scores"]}

        return _pattern_predict(premise, hypothesis)

    def detect_disagreement(self, text_a: str, text_b: str) -> Dict[str, Any]:
        result = self.predict(text_a, text_b)
        contra = result["all_scores"].get(NLILabel.CONTRADICTION, 0.0)
        return {
            "disagreement": contra >= self.threshold,
            "label": result["label"],
            "contradiction_score": contra,
            "entailment_score": result["all_scores"].get(NLILabel.ENTAILMENT, 0.0),
            "neutral_score": result["all_scores"].get(NLILabel.NEUTRAL, 0.0),
            "threshold": self.threshold,
            "all_scores": result["all_scores"],
        }


# ------------------------------------------------------------------ #
# Factory                                                              #
# ------------------------------------------------------------------ #

def get_nli_detector():
    """Return MockNLIDetector in mock/demo mode, NLIDetector in transformers mode."""
    if settings.nli_mode == "mock" or settings.demo_mode:
        logger.info("Using MockNLIDetector (mock/demo mode)")
        return MockNLIDetector(threshold=settings.disagreement_threshold)
    logger.info(f"Using NLIDetector (model={settings.nli_model})")
    return NLIDetector(
        model_name=settings.nli_model,
        device=settings.nli_device,
        threshold=settings.disagreement_threshold,
    )
