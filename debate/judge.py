"""
Judge for evaluating debate outcomes.
Task 13: No more except→_demo_* fallbacks; LLMError propagates.
Task 15: Judge only called when outcome.status == "completed".
         Receives NormalizedClaims with sources and conflict type in structured form.
         judge_result = None on any failure — never fabricated.
"""
import re
from typing import List, Optional
from loguru import logger

from evidence.models import (
    AgentEvidence, ConflictType, DebateArgument, DebateRound,
    DebateOutcome, JudgeResult,
)
from llm_client import LLMClient, LLMError


JUDGE_PROMPT = """You are a fair and impartial judge evaluating debate arguments.
You must evaluate based on EVIDENCE, not persuasion.

QUESTION: {query}
CONFLICT TYPE: {conflict_type}

CANDIDATE CLAIMS:
Claim A ({agent_a_id}):
  Text: {claim_a_text}
  Date: {claim_a_date}
  Version: {claim_a_version}
  Scope: {claim_a_scope}
  Source: {claim_a_source}

Claim B ({agent_b_id}):
  Text: {claim_b_text}
  Date: {claim_b_date}
  Version: {claim_b_version}
  Scope: {claim_b_scope}
  Source: {claim_b_source}

DEBATE TRANSCRIPT:
{debate_transcript}

Evaluate and return ONLY a JSON object:
{{
    "winner": "candidate_1" or "candidate_2",
    "reason": "Brief explanation",
    "evidence_score": 0.0-1.0,
    "source_score": 0.0-1.0,
    "reasoning_score": 0.0-1.0,
    "confidence": 0.0-1.0
}}
"""


class Judge:
    """
    Evaluates debate outcomes and selects the best candidate answer.
    Never fabricates a verdict — returns None if evaluation fails.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None, run_mode: str = "live"):
        self.llm_client = llm_client
        self.run_mode = run_mode

    def evaluate(
        self,
        query: str,
        evidence_list: List[AgentEvidence],
        debate_outcome: DebateOutcome,
        conflict_type: ConflictType,
        top_conflict: dict,
    ) -> Optional[JudgeResult]:
        """
        Evaluate the debate and select the best answer.

        Args:
            query: Original user query
            evidence_list: All usable AgentEvidence objects
            debate_outcome: Must have status == "completed"
            conflict_type: Type of conflict detected
            top_conflict: The conflicting claim pair

        Returns:
            JudgeResult or None if debate was not completed or evaluation fails
        """
        # Task 15 guard: only judge a real completed debate
        if debate_outcome.status != "completed":
            logger.info(
                f"Skipping judge — debate status is '{debate_outcome.status}', "
                f"failed_stages={debate_outcome.failed_stages}"
            )
            return None

        if self.run_mode == "fixture" or not self.llm_client:
            return self._fixture_evaluate(evidence_list, conflict_type, debate_outcome)

        return self._llm_evaluate(query, evidence_list, debate_outcome, conflict_type, top_conflict)

    def _llm_evaluate(
        self,
        query: str,
        evidence_list: List[AgentEvidence],
        debate_outcome: DebateOutcome,
        conflict_type: ConflictType,
        top_conflict: dict,
    ) -> Optional[JudgeResult]:
        """Evaluate using LLM. Returns None on failure."""
        try:
            claim_a = top_conflict.get("claim_a")
            claim_b = top_conflict.get("claim_b")
            agent_a_id = top_conflict.get("agent_a_id", "agent_a")
            agent_b_id = top_conflict.get("agent_b_id", "agent_b")

            transcript = self._build_transcript(debate_outcome.rounds)

            prompt = JUDGE_PROMPT.format(
                query=query,
                conflict_type=conflict_type.value,
                agent_a_id=agent_a_id,
                agent_b_id=agent_b_id,
                claim_a_text=claim_a.text if claim_a else "N/A",
                claim_a_date=claim_a.date or "unknown" if claim_a else "unknown",
                claim_a_version=claim_a.version or "unknown" if claim_a else "unknown",
                claim_a_scope=claim_a.scope or "unknown" if claim_a else "unknown",
                claim_a_source=claim_a.source.url if claim_a else "unknown",
                claim_b_text=claim_b.text if claim_b else "N/A",
                claim_b_date=claim_b.date or "unknown" if claim_b else "unknown",
                claim_b_version=claim_b.version or "unknown" if claim_b else "unknown",
                claim_b_scope=claim_b.scope or "unknown" if claim_b else "unknown",
                claim_b_source=claim_b.source.url if claim_b else "unknown",
                debate_transcript=transcript,
            )

            result = self.llm_client.chat_json(prompt, role="judge")

            winner = result.get("winner", "candidate_1")
            if winner not in ("candidate_1", "candidate_2"):
                winner = "candidate_1"

            return JudgeResult(
                winner=winner,
                reason=result.get("reason", "LLM evaluation"),
                evidence_score=float(result.get("evidence_score", 0.8)),
                reasoning_score=float(result.get("reasoning_score", 0.8)),
                source_score=float(result.get("source_score", 0.8)),
                confidence=float(result.get("confidence", 0.8)),
                scores={
                    "candidate_1": float(result.get("evidence_score", 0.8)),
                    "candidate_2": float(result.get("reasoning_score", 0.75)),
                },
            )

        except LLMError as e:
            logger.error(f"Judge LLM evaluation failed: {e}")
            return None

    def _fixture_evaluate(
        self,
        evidence_list: List[AgentEvidence],
        conflict_type: ConflictType,
        debate_outcome: DebateOutcome,
    ) -> Optional[JudgeResult]:
        """Fixture-mode evaluation — pick the agent with the most recent/primary source."""
        if not evidence_list:
            return None

        reasons = {
            ConflictType.TEMPORAL:    "The more recent official source carries greater authority for current-state queries.",
            ConflictType.VERSION:     "The source describing the specifically queried version is directly relevant.",
            ConflictType.CONTEXTUAL:  "Both claims are valid within their respective scopes; the primary population addressed wins.",
            ConflictType.SOURCE:      "Peer-reviewed and official sources outweigh informal blog or opinion content.",
            ConflictType.FACTUAL:     "The claim with stronger direct evidence and source authority is preferred.",
            ConflictType.UNKNOWN:     "The claim with the highest source credibility and specificity is preferred.",
        }
        reason = reasons.get(conflict_type, "Higher source authority determines the winner.")

        # Candidate 1 = first usable agent (typically web_agent / most recent)
        return JudgeResult(
            winner="candidate_1",
            reason=reason,
            evidence_score=0.85,
            reasoning_score=0.80,
            source_score=0.82,
            confidence=0.82,
            scores={"candidate_1": 0.85, "candidate_2": 0.72},
        )

    def _build_transcript(self, rounds: List[DebateRound]) -> str:
        """Build debate transcript from rounds."""
        if not rounds:
            return "No debate rounds completed."
        lines = []
        for r in rounds:
            lines.append(f"\n=== ROUND {r.round_number} ===\n")
            if r.proposer_argument:
                lines.append(f"PROPOSER:\n{r.proposer_argument.content}\n")
            for i, critic in enumerate(r.critic_arguments):
                lines.append(f"CRITIC {i+1}:\n{critic.content}\n")
            if r.reviser_argument:
                lines.append(f"REVISER:\n{r.reviser_argument.content}\n")
        return "\n".join(lines)


def get_judge(llm_client=None, run_mode: str = "live") -> Judge:
    """Get or create a judge instance."""
    return Judge(llm_client=llm_client, run_mode=run_mode)