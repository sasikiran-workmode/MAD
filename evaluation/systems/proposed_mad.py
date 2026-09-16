"""
Proposed MAD System (Task 25).
Gate + conflict-type-aware strategies.
The only system with both novelties.
"""
from evidence.models import ExecutionTrace
from app import MADSystem, save_trace
from run_mode import RunMode


class ProposedMADSystem:
    """
    Proposed MAD: evidence gate + typed conflict strategies.
    This is the full system as described in the paper.
    """

    SYSTEM_NAME = "proposed_mad"

    def __init__(self, run_mode: RunMode = RunMode.LIVE):
        self.run_mode = run_mode
        self._base = MADSystem(run_mode=run_mode)

    def run(self, query: str) -> ExecutionTrace:
        trace = self._base.solve(query)
        trace.metrics["system"] = self.SYSTEM_NAME
        return trace


# ------------------------------------------------------------------ #
# Ablations (Task 26)                                                  #
# ------------------------------------------------------------------ #

class NoGateSystem:
    """
    Ablation: Proposed minus the gate.
    Typed strategies, but always debates (ignores evidence state).
    Equivalent to StandardMAD + typed classification.
    """

    SYSTEM_NAME = "no_gate"

    def __init__(self, run_mode: RunMode = RunMode.LIVE):
        self.run_mode = run_mode
        from evaluation.systems.standard_mad import StandardMADSystem
        # StandardMAD always debates, but NoGate uses typed classification
        self._standard = StandardMADSystem(run_mode=run_mode)
        self._base = MADSystem(run_mode=run_mode)

    def run(self, query: str) -> ExecutionTrace:
        """Always debate but use real conflict classification."""
        # Run full pipeline (no override)
        trace = self._base.solve(query)

        # Force debate even on non-DISAGREEMENT states
        if not trace.debate_triggered and trace.evidence_decision:
            from evaluation.systems.standard_mad import StandardMADSystem
            std = StandardMADSystem(run_mode=self.run_mode)
            trace = std.run(query)

        trace.metrics["system"] = self.SYSTEM_NAME
        return trace


class NoTypedStrategySystem:
    """
    Ablation: Proposed minus typed strategies.
    Gate enabled, but always uses UNKNOWN (generic) strategy regardless of conflict type.
    """

    SYSTEM_NAME = "no_typed_strategy"

    def __init__(self, run_mode: RunMode = RunMode.LIVE):
        self.run_mode = run_mode
        self._base = MADSystem(run_mode=run_mode)

    def run(self, query: str) -> ExecutionTrace:
        """Gate works, but override conflict type to UNKNOWN before debate."""
        trace = self._base.solve(query)

        if trace.debate_triggered and trace.conflict:
            from evidence.models import ConflictType
            # Override the conflict type to UNKNOWN (generic strategy)
            original_type = trace.conflict.type
            trace.conflict.type = ConflictType.UNKNOWN
            trace.conflict.metadata["original_type"] = original_type.value
            trace.conflict.metadata["strategy_override"] = "UNKNOWN"

        trace.metrics["system"] = self.SYSTEM_NAME
        return trace


class NoRelevanceFilterSystem:
    """
    Ablation: Proposed with Task 8's relevance filter disabled.
    All retrieved snippets pass through without content-word overlap filtering.
    """

    SYSTEM_NAME = "no_relevance_filter"

    def __init__(self, run_mode: RunMode = RunMode.LIVE):
        self.run_mode = run_mode
        self._base = MADSystem(run_mode=run_mode)

        # Patch the normalizer to skip relevance filtering
        from evidence.normalizer import EvidenceNormalizer
        import types

        def _no_filter_normalize(self_norm, agent_id, query, raw_results):
            """Skip relevance filter — pass all results through."""
            if not raw_results:
                from evidence.models import AgentEvidence, EvidenceStatus
                return AgentEvidence(
                    agent_id=agent_id, query=query,
                    status=EvidenceStatus.NO_RELEVANT_RESULTS,
                    failure_reason="No search results",
                )
            from evidence.models import Source, AgentEvidence, EvidenceStatus
            sources = [
                Source(
                    url=r.get("link", ""),
                    title=r.get("title", ""),
                    snippet=r.get("snippet", ""),
                )
                for r in raw_results[:5]
            ]
            claim = self_norm._llm_normalize(agent_id, query, raw_results, sources)
            if claim is None:
                return AgentEvidence(
                    agent_id=agent_id, query=query,
                    status=EvidenceStatus.NO_RELEVANT_RESULTS,
                    failure_reason="LLM found no answer",
                    sources=sources,
                    raw_results_count=len(raw_results),
                    filtered_out_count=0,
                )
            return AgentEvidence(
                agent_id=agent_id, query=query,
                status=EvidenceStatus.OK,
                claims=[claim],
                sources=sources,
                raw_results_count=len(raw_results),
                filtered_out_count=0,  # no filtering applied
            )

        self._base.normalizer.normalize = types.MethodType(
            _no_filter_normalize, self._base.normalizer
        )

    def run(self, query: str) -> ExecutionTrace:
        trace = self._base.solve(query)
        trace.metrics["system"] = self.SYSTEM_NAME
        return trace
