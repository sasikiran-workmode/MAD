"""
Proposer agent for the debate module.
Task 13: No more except→_demo_* fallback; LLMError propagates.
Task 16: Records strategy_used and strategy_text_preview.
Task 18: Temporal strategy receives query period + claim dates.
Task 19: Version strategy checks whether both claims share a version.
Task 20: Contextual strategy checks scope fields before declaring winner.
"""
from typing import List, Optional
from loguru import logger

from evidence.models import (
    AgentEvidence, ConflictType, DebateArgument, EvidenceState,
)
from llm_client import LLMClient, LLMError
from config import settings


# ------------------------------------------------------------------ #
# Conflict-specific strategy texts (DO NOT DELETE — Task 30 note)     #
# ------------------------------------------------------------------ #

CONFLICT_STRATEGIES = {
    ConflictType.FACTUAL: """You are arguing about a factual disagreement.
Focus on direct evidence. Compare the competing factual claims directly against the retrieved evidence.
Identify which claim has stronger direct support. Cite specific sources.""",

    ConflictType.TEMPORAL: """You are arguing about a temporal disagreement.
IMPORTANT: The query may specify a particular date or time period (see QUERY PERIOD below).
For each claim, check its recorded date. If a claim is dated AFTER the query period, it must be
excluded from consideration — it is irrelevant to what was true at the requested time.
Construct a timeline. Determine which claim best answers the user's question given the requested period.""",

    ConflictType.VERSION: """You are arguing about a version disagreement.
IMPORTANT: First check whether the two claims actually refer to the SAME version.
If they refer to different versions (see VERSION fields below), the disagreement may be fully
explained by the version difference — in that case, report "both correct for their respective versions"
rather than declaring one side wrong.
If the query specifies a particular version, only claims for that version are relevant.""",

    ConflictType.CONTEXTUAL: """You are arguing about a contextual disagreement.
IMPORTANT: First check whether the two claims apply to different populations, platforms, or editions
(see SCOPE fields below). If the scopes are different, the contradiction may dissolve —
report the conditions under which each claim holds rather than picking a winner.""",

    ConflictType.SOURCE: """You are arguing about a source disagreement.
Evaluate source authority, provenance, specificity, and reliability.
Prefer stronger primary evidence where appropriate.""",

    ConflictType.UNKNOWN: """You are evaluating conflicting evidence.
Analyse both claims carefully. Identify which has stronger, more specific, and more authoritative
support. Provide a balanced assessment.""",
}


def _extract_query_year(query: str) -> Optional[str]:
    """Extract an explicit year or 'as of X' from the query."""
    import re
    # Explicit year: "in 2024", "as of 2024", "during 2024"
    m = re.search(r"\b(?:in|as of|during|for|by)\s+((?:19|20)\d{2})\b", query, re.I)
    if m:
        return m.group(1)
    # Bare 4-digit year at end of query
    m = re.search(r"\b((?:19|20)\d{2})\b", query)
    if m:
        return m.group(1)
    return None


def is_valid_argument(arg: Optional[DebateArgument]) -> bool:
    """Validate that a debate argument is real content (Task 14)."""
    return (
        arg is not None
        and len(arg.content.strip()) >= 50
        and not arg.content.startswith("Demo response")
    )


class Proposer:
    """
    Generates initial arguments in the debate.
    Uses conflict-type-specific strategies.
    LLMError propagates — no silent fallback.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None, run_mode: str = "live"):
        self.llm_client = llm_client
        self.run_mode = run_mode  # "live" | "fixture"

    def propose(
        self,
        query: str,
        evidence_list: List[AgentEvidence],
        conflict_type: ConflictType,
        conflict_explanation: str,
        top_conflict: dict,
        round_number: int = 1,
    ) -> DebateArgument:
        """
        Generate a proposal argument.

        Args:
            query: Original user query
            evidence_list: All usable AgentEvidence objects
            conflict_type: Type of conflict detected
            conflict_explanation: Brief explanation of the conflict
            top_conflict: The highest-scoring contradictory pair from EvidenceDecision
            round_number: Current debate round

        Returns:
            DebateArgument

        Raises:
            LLMError: if LLM call fails (caller should catch and record stage_failure)
        """
        strategy = CONFLICT_STRATEGIES.get(conflict_type, CONFLICT_STRATEGIES[ConflictType.UNKNOWN])

        if self.run_mode == "fixture" or not self.llm_client:
            return self._fixture_propose(
                query, evidence_list, conflict_type, strategy, top_conflict, round_number
            )

        return self._llm_propose(
            query, evidence_list, conflict_type,
            conflict_explanation, strategy, top_conflict, round_number
        )

    def _llm_propose(
        self,
        query: str,
        evidence_list: List[AgentEvidence],
        conflict_type: ConflictType,
        conflict_explanation: str,
        strategy: str,
        top_conflict: dict,
        round_number: int,
    ) -> DebateArgument:
        """Generate proposal using LLM. Raises LLMError on failure."""
        claim_a = top_conflict.get("claim_a")
        claim_b = top_conflict.get("claim_b")
        claim_a_text = claim_a.text if claim_a else "N/A"
        claim_b_text = claim_b.text if claim_b else "N/A"

        # Build metadata context for temporal/version/contextual strategies
        meta_lines = []
        query_year = _extract_query_year(query)
        if conflict_type == ConflictType.TEMPORAL and query_year:
            meta_lines.append(f"QUERY PERIOD: {query_year}")
        if claim_a and claim_a.date:
            meta_lines.append(f"Claim A date: {claim_a.date}")
        if claim_b and claim_b.date:
            meta_lines.append(f"Claim B date: {claim_b.date}")
        if conflict_type == ConflictType.VERSION:
            if claim_a and claim_a.version:
                meta_lines.append(f"Claim A version: {claim_a.version}")
            if claim_b and claim_b.version:
                meta_lines.append(f"Claim B version: {claim_b.version}")
        if conflict_type == ConflictType.CONTEXTUAL:
            if claim_a and claim_a.scope:
                meta_lines.append(f"Claim A scope: {claim_a.scope}")
            if claim_b and claim_b.scope:
                meta_lines.append(f"Claim B scope: {claim_b.scope}")
        meta_block = ("\n" + "\n".join(meta_lines)) if meta_lines else ""

        agent_a_id = top_conflict.get("agent_a_id", "agent_a")
        agent_b_id = top_conflict.get("agent_b_id", "agent_b")

        prompt = f"""{strategy}
{meta_block}
QUESTION: {query}

CONFLICT DETECTED:
- Claim A ({agent_a_id}): {claim_a_text}
- Claim B ({agent_b_id}): {claim_b_text}

Conflict Type: {conflict_type.value}
Explanation: {conflict_explanation}

Generate a structured proposal (Round {round_number}) that:
1. Analyses the evidence from both sides
2. Identifies the core issue in the conflict
3. Proposes an initial position supported by evidence
4. References specific sources

Provide your proposal:"""

        response = self.llm_client.chat(prompt, role="proposer")

        return DebateArgument(
            round_number=round_number,
            role="proposer",
            agent_id="proposer",
            content=response,
            evidence_refs=[agent_a_id, agent_b_id],
            conflict_type=conflict_type,
            strategy_used=conflict_type.value,
            strategy_text_preview=strategy[:200],
        )

    def _fixture_propose(
        self,
        query: str,
        evidence_list: List[AgentEvidence],
        conflict_type: ConflictType,
        strategy: str,
        top_conflict: dict,
        round_number: int,
    ) -> DebateArgument:
        """Generate a realistic proposal for FIXTURE mode."""
        claim_a = top_conflict.get("claim_a")
        claim_b = top_conflict.get("claim_b")
        claim_a_text = claim_a.text if claim_a else "Claim A"
        claim_b_text = claim_b.text if claim_b else "Claim B"

        if conflict_type == ConflictType.TEMPORAL:
            content = (
                f"Proposal (Round {round_number}): Analysing the temporal conflict.\n\n"
                f"Source A states: \"{claim_a_text}\"\n"
                f"Source B states: \"{claim_b_text}\"\n\n"
                f"These claims refer to different time points. "
                f"The date metadata confirms this is a genuine temporal disagreement, not a factual error. "
                f"To resolve this, we must determine which date is most relevant to the query. "
                f"The more recent and primary source should take precedence unless the query "
                f"specifically targets a historical date."
            )
        elif conflict_type == ConflictType.VERSION:
            content = (
                f"Proposal (Round {round_number}): Analysing the version conflict.\n\n"
                f"Source A states: \"{claim_a_text}\"\n"
                f"Source B states: \"{claim_b_text}\"\n\n"
                f"These claims describe different versions of the same software/product. "
                f"Both claims may be correct within their respective version scope. "
                f"The key question is whether the query targets a specific version or the latest. "
                f"If no version is specified, both claims should be presented with their version context."
            )
        elif conflict_type == ConflictType.CONTEXTUAL:
            content = (
                f"Proposal (Round {round_number}): Analysing the contextual conflict.\n\n"
                f"Source A states: \"{claim_a_text}\"\n"
                f"Source B states: \"{claim_b_text}\"\n\n"
                f"These claims apply to different populations or conditions. "
                f"The scope metadata reveals this is not a factual contradiction — "
                f"both statements are correct within their respective contexts. "
                f"The resolution is to clarify under which conditions each claim holds."
            )
        elif conflict_type == ConflictType.SOURCE:
            content = (
                f"Proposal (Round {round_number}): Analysing the source credibility conflict.\n\n"
                f"Source A states: \"{claim_a_text}\"\n"
                f"Source B states: \"{claim_b_text}\"\n\n"
                f"These claims originate from sources with different levels of authority. "
                f"Peer-reviewed research and official bodies carry greater evidential weight "
                f"than blog posts or opinion pieces. "
                f"The claim backed by higher-quality evidence should be preferred."
            )
        else:
            content = (
                f"Proposal (Round {round_number}): Analysing the factual conflict.\n\n"
                f"Source A states: \"{claim_a_text}\"\n"
                f"Source B states: \"{claim_b_text}\"\n\n"
                f"These claims make directly incompatible factual assertions. "
                f"Evaluating source recency, authority, and specificity will determine which claim "
                f"has stronger evidentiary support."
            )

        return DebateArgument(
            round_number=round_number,
            role="proposer",
            agent_id="proposer",
            content=content,
            evidence_refs=[
                top_conflict.get("agent_a_id", ""),
                top_conflict.get("agent_b_id", ""),
            ],
            conflict_type=conflict_type,
            strategy_used=conflict_type.value,
            strategy_text_preview=strategy[:200],
        )