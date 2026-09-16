"""
Reviser agent for the debate module.
Task 13: No more except→_demo_* fallback; LLMError propagates.
"""
from typing import List, Optional
from loguru import logger

from evidence.models import AgentEvidence, ConflictType, DebateArgument
from llm_client import LLMClient, LLMError


class Reviser:
    """
    Revises arguments by incorporating critique feedback.
    LLMError propagates — no silent fallback.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None, run_mode: str = "live"):
        self.llm_client = llm_client
        self.run_mode = run_mode

    def revise(
        self,
        query: str,
        proposal: DebateArgument,
        critiques: List[DebateArgument],
        evidence_list: List[AgentEvidence],
        conflict_type: ConflictType,
        top_conflict: dict,
        round_number: int,
    ) -> DebateArgument:
        """
        Generate a revised argument incorporating critiques.

        Returns:
            Revised DebateArgument

        Raises:
            LLMError: if LLM call fails
        """
        if self.run_mode == "fixture" or not self.llm_client:
            return self._fixture_revise(query, proposal, critiques, conflict_type, round_number)

        return self._llm_revise(
            query, proposal, critiques, evidence_list, conflict_type, top_conflict, round_number
        )

    def _llm_revise(
        self,
        query: str,
        proposal: DebateArgument,
        critiques: List[DebateArgument],
        evidence_list: List[AgentEvidence],
        conflict_type: ConflictType,
        top_conflict: dict,
        round_number: int,
    ) -> DebateArgument:
        """Generate revision using LLM. Raises LLMError on failure."""
        claim_a = top_conflict.get("claim_a")
        claim_b = top_conflict.get("claim_b")
        claim_a_text = claim_a.text if claim_a else "Claim A"
        claim_b_text = claim_b.text if claim_b else "Claim B"
        agent_a_id = top_conflict.get("agent_a_id", "agent_a")
        agent_b_id = top_conflict.get("agent_b_id", "agent_b")

        critiques_text = "\n\n".join(
            f"Critique {i+1}:\n{c.content}" for i, c in enumerate(critiques)
        )

        prompt = f"""You are a debate reviser. Produce an improved argument that addresses the critiques.

QUESTION: {query}
CONFLICT TYPE: {conflict_type.value}

ORIGINAL CLAIMS:
- {agent_a_id}: {claim_a_text}
- {agent_b_id}: {claim_b_text}

ORIGINAL PROPOSAL:
{proposal.content}

CRITIQUES TO ADDRESS:
{critiques_text}

Generate a revised argument (Round {round_number}) that:
1. Directly addresses each critique
2. Incorporates valid points from critiques
3. Strengthens weak areas
4. Provides a well-reasoned final position
5. References specific evidence

Your revised proposal:"""

        response = self.llm_client.chat(prompt, role="reviser")

        return DebateArgument(
            round_number=round_number,
            role="reviser",
            agent_id="reviser",
            content=response,
            evidence_refs=[agent_a_id, agent_b_id],
            conflict_type=conflict_type,
            strategy_used=f"{conflict_type.value}_revision",
        )

    def _fixture_revise(
        self,
        query: str,
        proposal: DebateArgument,
        critiques: List[DebateArgument],
        conflict_type: ConflictType,
        round_number: int,
    ) -> DebateArgument:
        """Realistic revision for FIXTURE mode."""
        if conflict_type == ConflictType.TEMPORAL:
            content = (
                f"Revised Position (Round {round_number}): "
                f"Incorporating both critiques — the conflict is temporal, not factual. "
                f"Both dates (1986 and 2020) are historically accurate. "
                f"The current operative answer is 2020, as this is the most recent and active policy. "
                f"The 1986 policy is superseded. Resolution: answer with the 2020 date, "
                f"noting 1986 as the original framework."
            )
        elif conflict_type == ConflictType.VERSION:
            content = (
                f"Revised Position (Round {round_number}): "
                f"Incorporating critique — the query specifically targets Python 3.13. "
                f"Source B's information about Python 3.14 is accurate but out of scope. "
                f"Resolution: answer with Python 3.13 features (free-threaded mode, new REPL), "
                f"and note 3.14 as a separate upcoming release."
            )
        elif conflict_type == ConflictType.CONTEXTUAL:
            content = (
                f"Revised Position (Round {round_number}): "
                f"Both critiques are valid. The apparent contradiction dissolves under scope analysis: "
                f"the drug / substance is safe and effective for adults, but contraindicated "
                f"for children due to documented physiological differences. "
                f"Resolution: both claims are correct — the answer must specify the patient population."
            )
        else:
            content = (
                f"Revised Position (Round {round_number}): "
                f"After weighing critiques, the resolution favours the claim supported by "
                f"higher-authority, peer-reviewed evidence. "
                f"The conflicting claim from lower-credibility sources is noted but outweighed. "
                f"Resolution: adopt the evidence-backed position while flagging the source conflict."
            )
        return DebateArgument(
            round_number=round_number, role="reviser", agent_id="reviser",
            content=content, conflict_type=conflict_type,
            strategy_used=f"{conflict_type.value}_revision",
        )