"""
Critic agent for the debate module.
Task 13: No more except→_demo_* fallback; LLMError propagates.
"""
from typing import List, Optional
from loguru import logger

from evidence.models import AgentEvidence, ConflictType, DebateArgument
from llm_client import LLMClient, LLMError


class Critic:
    """
    Critiques proposals by identifying logical weaknesses,
    unsupported claims, and reasoning gaps.
    LLMError propagates — no silent fallback.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None, run_mode: str = "live"):
        self.llm_client = llm_client
        self.run_mode = run_mode

    def critique(
        self,
        query: str,
        proposal: DebateArgument,
        evidence_list: List[AgentEvidence],
        conflict_type: ConflictType,
        top_conflict: dict,
        round_number: int,
    ) -> List[DebateArgument]:
        """
        Generate critiques of a proposal.

        Returns:
            List of critique DebateArguments

        Raises:
            LLMError: if LLM call fails
        """
        if self.run_mode == "fixture" or not self.llm_client:
            return self._fixture_critique(query, proposal, conflict_type, round_number)

        return self._llm_critique(
            query, proposal, evidence_list, conflict_type, top_conflict, round_number
        )

    def _llm_critique(
        self,
        query: str,
        proposal: DebateArgument,
        evidence_list: List[AgentEvidence],
        conflict_type: ConflictType,
        top_conflict: dict,
        round_number: int,
    ) -> List[DebateArgument]:
        """Generate critiques using LLM. Raises LLMError on failure."""
        claim_a = top_conflict.get("claim_a")
        claim_b = top_conflict.get("claim_b")
        claim_a_text = claim_a.text if claim_a else "Claim A"
        claim_b_text = claim_b.text if claim_b else "Claim B"
        agent_a_id = top_conflict.get("agent_a_id", "agent_a")
        agent_b_id = top_conflict.get("agent_b_id", "agent_b")

        prompt = f"""You are a critical analyst in a structured debate.
Also return a JSON field "substantive_objection": true/false at the end,
indicating whether you found a real logical flaw (true) or the proposal is already sound (false).

QUESTION: {query}
CONFLICT TYPE: {conflict_type.value}

ORIGINAL CLAIMS:
- {agent_a_id}: {claim_a_text}
- {agent_b_id}: {claim_b_text}

PROPOSAL TO CRITIQUE:
{proposal.content}

Generate TWO distinct critiques that:
1. Identify logical weaknesses or unsupported claims
2. Challenge assumptions the proposer made
3. Point out missing evidence or reasoning gaps
4. Suggest specific improvements

Format each critique clearly as CRITIQUE 1 and CRITIQUE 2.
After the critiques, add a final line: SUBSTANTIVE_OBJECTION: true or false"""

        response = self.llm_client.chat(prompt, role="critic")

        # Extract substantive_objection flag
        import re
        sub_obj = True  # default assume yes
        sub_match = re.search(r"SUBSTANTIVE_OBJECTION:\s*(true|false)", response, re.I)
        if sub_match:
            sub_obj = sub_match.group(1).lower() == "true"

        # Split into two critiques
        parts = response.split("CRITIQUE 2")
        content_1 = parts[0].replace("CRITIQUE 1", "").strip() if len(parts) == 2 else response[: len(response) // 2]
        content_2 = parts[1].strip() if len(parts) == 2 else response[len(response) // 2:]

        return [
            DebateArgument(
                round_number=round_number,
                role="critic",
                agent_id="critic_1",
                content=content_1,
                evidence_refs=[agent_a_id],
                conflict_type=conflict_type,
                strategy_used="critique" + ("_substantive" if sub_obj else "_no_objection"),
            ),
            DebateArgument(
                round_number=round_number,
                role="critic",
                agent_id="critic_2",
                content=content_2,
                evidence_refs=[agent_b_id],
                conflict_type=conflict_type,
                strategy_used="critique",
            ),
        ]

    def _fixture_critique(
        self,
        query: str,
        proposal: DebateArgument,
        conflict_type: ConflictType,
        round_number: int,
    ) -> List[DebateArgument]:
        """Realistic critiques for FIXTURE mode."""
        if conflict_type == ConflictType.TEMPORAL:
            c1 = (
                f"Critique 1 (Round {round_number}): The proposal correctly identifies the temporal gap "
                f"but does not weigh source authority. The more recent official source (2020) "
                f"reflects current policy, while the 1986 reference describes a superseded framework. "
                f"The proposal should explicitly state which is the operative answer."
            )
            c2 = (
                f"Critique 2 (Round {round_number}): The proposal omits that both dates may be correct "
                f"— 1986 was a real policy, 2020 is the current one. "
                f"The answer depends entirely on whether the query asks about the original or current policy."
            )
        elif conflict_type == ConflictType.VERSION:
            c1 = (
                f"Critique 1 (Round {round_number}): The proposal identifies the version mismatch "
                f"but does not confirm whether both sources are accurate for their stated versions. "
                f"Version 3.13 and 3.14 are distinct releases — both claims can be simultaneously correct."
            )
            c2 = (
                f"Critique 2 (Round {round_number}): The proposal should clarify that the query asked "
                f"specifically about 3.13. Source B's information about 3.14 is technically accurate "
                f"but out of scope for this query."
            )
        elif conflict_type == ConflictType.CONTEXTUAL:
            c1 = (
                f"Critique 1 (Round {round_number}): The proposal acknowledges scope differences "
                f"but does not explicitly state the conditions. "
                f"The answer should specify: safe for adults (18+), contraindicated for children."
            )
            c2 = (
                f"Critique 2 (Round {round_number}): The proposal could strengthen the argument "
                f"by citing the specific mechanism — e.g. Reye's syndrome risk in children — "
                f"rather than treating this as a simple disagreement."
            )
        else:
            c1 = (
                f"Critique 1 (Round {round_number}): The proposal identifies the conflict "
                f"but does not evaluate source credibility. "
                f"Peer-reviewed sources should be weighted higher than informal publications."
            )
            c2 = (
                f"Critique 2 (Round {round_number}): The proposal lacks a concrete resolution. "
                f"It should recommend the claim supported by the highest-authority source "
                f"while acknowledging the conflicting claim exists."
            )
        return [
            DebateArgument(
                round_number=round_number, role="critic", agent_id="critic_1",
                content=c1, conflict_type=conflict_type, strategy_used="critique_substantive",
            ),
            DebateArgument(
                round_number=round_number, role="critic", agent_id="critic_2",
                content=c2, conflict_type=conflict_type, strategy_used="critique_substantive",
            ),
        ]