"""
Evidence comparator for comparing claims from different agents.
"""
from typing import List, Dict, Any, Optional, Tuple
from itertools import product
import asyncio
from loguru import logger

from evidence.models import (
    Evidence, Claim, EvidenceComparison, NLILabel, ConflictType
)
from config import settings


class EvidenceComparator:
    """
    Compares evidence from two agents to identify agreements and disagreements.
    Uses NLI (Natural Language Inference) to determine relationships between claims.
    """
    
    def __init__(self, nli_detector=None):
        self.nli_detector = nli_detector
        self.threshold = settings.disagreement_threshold
    
    def compare(self, evidence_a: Evidence, evidence_b: Evidence) -> EvidenceComparison:
        """
        Compare evidence from two agents.
        
        Args:
            evidence_a: Evidence from agent A
            evidence_b: Evidence from agent B
            
        Returns:
            EvidenceComparison with detailed results
        """
        comparison = EvidenceComparison(
            agent_a_id=evidence_a.agent_id,
            agent_b_id=evidence_b.agent_id,
        )
        
        # Compare all claim pairs
        claim_pairs = []
        nli_results = []
        conflicts = []
        
        contradiction_scores = []
        entailment_scores = []
        neutral_scores = []
        
        for claim_a in evidence_a.claims:
            for claim_b in evidence_b.claims:
                pair = {
                    "claim_a_id": claim_a.id,
                    "claim_a_text": claim_a.text,
                    "claim_b_id": claim_b.id,
                    "claim_b_text": claim_b.text,
                }
                claim_pairs.append(pair)
                
                # Get NLI prediction if detector available
                if self.nli_detector:
                    result = self.nli_detector.predict(claim_a.text, claim_b.text)
                    nli_results.append({
                        "claim_a_id": claim_a.id,
                        "claim_b_id": claim_b.id,
                        "label": result["label"],
                        "score": result["score"],
                        "all_scores": result.get("all_scores", {}),
                    })
                    
                    # Track scores for aggregate metrics
                    if result["label"] == NLILabel.CONTRADICTION:
                        contradiction_scores.append(result["score"])
                        # Record as conflict
                        conflicts.append({
                            "claim_a": claim_a.text,
                            "claim_b": claim_b.text,
                            "relationship": "contradiction",
                            "score": result["score"],
                            "claim_a_id": claim_a.id,
                            "claim_b_id": claim_b.id,
                        })
                    elif result["label"] == NLILabel.ENTAILMENT:
                        entailment_scores.append(result["score"])
                    else:
                        neutral_scores.append(result["score"])
        
        comparison.claim_pairs = claim_pairs
        comparison.nli_results = nli_results
        
        # Calculate aggregate scores
        comparison.contradiction_score = max(contradiction_scores) if contradiction_scores else 0.0
        comparison.entailment_score = max(entailment_scores) if entailment_scores else 0.0
        comparison.neutral_score = max(neutral_scores) if neutral_scores else 0.0
        
        # Determine if there's disagreement
        comparison.has_disagreement = comparison.contradiction_score >= self.threshold
        comparison.conflicts = conflicts
        
        logger.info(
            f"Comparison {evidence_a.agent_id} vs {evidence_b.agent_id}: "
            f"contradiction={comparison.contradiction_score:.3f}, "
            f"entailment={comparison.entailment_score:.3f}, "
            f"disagreement={comparison.has_disagreement}"
        )
        
        return comparison
    
    async def compare_async(self, evidence_a: Evidence, evidence_b: Evidence) -> EvidenceComparison:
        """Async version of compare (for consistency with async pipeline)."""
        return self.compare(evidence_a, evidence_b)


def find_most_contradictory_pair(comparison: EvidenceComparison) -> Optional[Tuple[Claim, Claim, float]]:
    """
    Find the most contradictory claim pair from a comparison.
    
    Returns:
        Tuple of (claim_a, claim_b, score) or None if no contradiction
    """
    if not comparison.conflicts:
        return None
    
    # Sort by contradiction score descending
    sorted_conflicts = sorted(comparison.conflicts, key=lambda x: x["score"], reverse=True)
    top = sorted_conflicts[0]
    
    # We need to find the actual Claim objects - this would need access to original evidence
    # For now, return the text and score
    return (top["claim_a"], top["claim_b"], top["score"])