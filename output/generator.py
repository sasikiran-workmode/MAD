"""
Final answer generator (Task 21).
Drives output from EvidenceState — four distinct shapes, no bullet dumps.
Reports independent_agents_agreeing separately from source_count.
"""
from typing import List, Optional
from loguru import logger

from evidence.models import (
    AgentEvidence, EvidenceState, EvidenceDecision,
    Conflict, DebateOutcome, JudgeResult, FinalAnswer, Source,
)
from llm_client import LLMClient, LLMError
from config import settings


class AnswerGenerator:
    """Generates final answers driven by EvidenceState."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client

    def generate(
        self,
        query: str,
        decision: EvidenceDecision,
        agent_evidence: List[AgentEvidence],
        conflict: Optional[Conflict] = None,
        debate_outcome: Optional[DebateOutcome] = None,
        judge_result: Optional[JudgeResult] = None,
    ) -> FinalAnswer:
        """
        Generate the final answer driven by decision.state.

        AGREEMENT    → single LLM call producing a direct answer
        DISAGREEMENT → conflict type + judge resolution (or "debate did not complete")
        UNCERTAIN    → "evidence neither clearly agreed nor conflicted"
        INSUFFICIENT → explicit statement naming failed agents; no answer content
        """
        state = decision.state

        if state == EvidenceState.AGREEMENT:
            return self._generate_agreement(query, decision, agent_evidence)

        if state == EvidenceState.DISAGREEMENT:
            return self._generate_disagreement(
                query, decision, agent_evidence, conflict, debate_outcome, judge_result
            )

        if state == EvidenceState.UNCERTAIN:
            return self._generate_uncertain(query, decision, agent_evidence)

        # INSUFFICIENT
        return self._generate_insufficient(query, decision)

    # ------------------------------------------------------------------ #
    # AGREEMENT                                                            #
    # ------------------------------------------------------------------ #

    def _generate_agreement(
        self,
        query: str,
        decision: EvidenceDecision,
        agent_evidence: List[AgentEvidence],
    ) -> FinalAnswer:
        usable = [e for e in agent_evidence if e.is_usable]
        all_sources = [s for e in usable for s in e.sources]
        unique_sources = self._deduplicate_sources(all_sources)

        # Build context for LLM
        claims_text = "\n".join(
            f"- [{e.agent_id}] {e.claims[0].text}" for e in usable if e.claims
        )

        if self.llm_client:
            prompt = (
                f"Query: {query}\n\n"
                f"Multiple independent sources agree on the following:\n{claims_text}\n\n"
                f"Write a concise, direct answer to the query using the agreed evidence. "
                f"Do NOT list bullet points or repeat raw snippets. Write 1-3 sentences."
            )
            try:
                answer_text = self.llm_client.chat(prompt, role="answer")
            except LLMError as e:
                logger.warning(f"LLM answer generation failed: {e}; using agreed claim directly")
                answer_text = usable[0].claims[0].text if usable and usable[0].claims else "No answer."
        else:
            answer_text = usable[0].claims[0].text if usable and usable[0].claims else "No answer."

        reasoning = (
            f"{len(usable)} independent agent(s) retrieved consistent evidence. "
            f"{decision.rationale}"
        )

        return FinalAnswer(
            query=query,
            answer=answer_text,
            reasoning=reasoning,
            evidence_state=EvidenceState.AGREEMENT,
            sources=unique_sources,
            debate_triggered=False,
            independent_agents_agreeing=len(usable),
            source_count=len(unique_sources),
            metadata={"evidence_state": "agreement"},
        )

    # ------------------------------------------------------------------ #
    # DISAGREEMENT                                                         #
    # ------------------------------------------------------------------ #

    def _generate_disagreement(
        self,
        query: str,
        decision: EvidenceDecision,
        agent_evidence: List[AgentEvidence],
        conflict: Optional[Conflict],
        debate_outcome: Optional[DebateOutcome],
        judge_result: Optional[JudgeResult],
    ) -> FinalAnswer:
        usable = [e for e in agent_evidence if e.is_usable]
        all_sources = [s for e in usable for s in e.sources]
        unique_sources = self._deduplicate_sources(all_sources)

        if judge_result is None or (debate_outcome and debate_outcome.status != "completed"):
            # Debate did not complete — never claim resolution
            conflict_desc = f" ({conflict.type.value})" if conflict else ""
            answer_text = (
                f"A conflict{conflict_desc} was detected between sources, but the structured debate "
                f"did not complete successfully"
            )
            if debate_outcome and debate_outcome.failed_stages:
                answer_text += f" (failed at: {', '.join(debate_outcome.failed_stages)})"
            answer_text += ". The conflict remains unresolved. Please consult the sources directly."

            return FinalAnswer(
                query=query,
                answer=answer_text,
                reasoning=decision.rationale,
                evidence_state=EvidenceState.DISAGREEMENT,
                sources=unique_sources,
                debate_triggered=True,
                conflict=conflict,
                debate_outcome=debate_outcome,
                judge_result=None,
                independent_agents_agreeing=0,
                source_count=len(unique_sources),
                metadata={
                    "evidence_state": "disagreement",
                    "debate_completed": False,
                },
            )

        # Debate completed with a judge verdict
        conflict_type_str = conflict.type.value.upper() if conflict else "UNKNOWN"
        winning_claim = (
            conflict.claim_a if judge_result.winner == "candidate_1" else conflict.claim_b
        )
        answer_text = (
            f"A {conflict_type_str} conflict was detected and resolved through structured debate.\n\n"
            f"Resolution: {winning_claim.text}\n\n"
            f"Judge's reasoning: {judge_result.reason}"
        )

        return FinalAnswer(
            query=query,
            answer=answer_text,
            reasoning=decision.rationale,
            evidence_state=EvidenceState.DISAGREEMENT,
            sources=unique_sources,
            debate_triggered=True,
            conflict=conflict,
            debate_outcome=debate_outcome,
            judge_result=judge_result,
            independent_agents_agreeing=0,
            source_count=len(unique_sources),
            metadata={
                "evidence_state": "disagreement",
                "conflict_type": conflict.type.value if conflict else "unknown",
                "debate_completed": True,
                "winner": judge_result.winner,
                "confidence": judge_result.confidence,
            },
        )

    # ------------------------------------------------------------------ #
    # UNCERTAIN                                                            #
    # ------------------------------------------------------------------ #

    def _generate_uncertain(
        self,
        query: str,
        decision: EvidenceDecision,
        agent_evidence: List[AgentEvidence],
    ) -> FinalAnswer:
        usable = [e for e in agent_evidence if e.is_usable]
        all_sources = [s for e in usable for s in e.sources]
        unique_sources = self._deduplicate_sources(all_sources)

        findings = "\n".join(
            f"- [{e.agent_id}]: {e.claims[0].text}" for e in usable if e.claims
        )
        answer_text = (
            f"Evidence neither clearly agreed nor conflicted for this query. "
            f"Different sources found:\n{findings}\n\n"
            f"No single authoritative answer could be determined. "
            f"Consider consulting the sources directly."
        )

        return FinalAnswer(
            query=query,
            answer=answer_text,
            reasoning=decision.rationale,
            evidence_state=EvidenceState.UNCERTAIN,
            sources=unique_sources,
            debate_triggered=False,
            independent_agents_agreeing=0,
            source_count=len(unique_sources),
            metadata={"evidence_state": "uncertain"},
        )

    # ------------------------------------------------------------------ #
    # INSUFFICIENT                                                         #
    # ------------------------------------------------------------------ #

    def _generate_insufficient(
        self,
        query: str,
        decision: EvidenceDecision,
    ) -> FinalAnswer:
        failed_info = "; ".join(
            f"{f['agent_id']} ({f['reason']})" for f in decision.failed_agents
        )
        answer_text = (
            f"Insufficient evidence was retrieved to answer this query. "
            f"Failed agents: {failed_info if failed_info else 'all agents returned no usable results'}."
        )

        return FinalAnswer(
            query=query,
            answer=answer_text,
            reasoning=decision.rationale,
            evidence_state=EvidenceState.INSUFFICIENT,
            sources=[],
            debate_triggered=False,
            independent_agents_agreeing=0,
            source_count=0,
            metadata={
                "evidence_state": "insufficient",
                "failed_agents": decision.failed_agents,
            },
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _deduplicate_sources(self, sources: List[Source]) -> List[Source]:
        """Remove duplicate sources by URL."""
        seen: set = set()
        unique = []
        for s in sources:
            if s.url not in seen:
                seen.add(s.url)
                unique.append(s)
        return unique


def get_answer_generator(llm_client=None) -> AnswerGenerator:
    """Get or create an answer generator instance."""
    return AnswerGenerator(llm_client=llm_client)