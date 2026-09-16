"""
Tests for debate components and the evidence gate.
Task 29: Fully synchronous; uses new API (AgentEvidence, NormalizedClaim, DebateOutcome).
         No asyncio. No DemoLLMClient. No Evidence/Claim in debate calls.
"""
import pytest
from evidence.models import (
    Source, NormalizedClaim, AgentEvidence, EvidenceStatus,
    Conflict, ConflictType, DebateRound, DebateOutcome, EvidenceState,
)
from debate.proposer import Proposer
from debate.critic import Critic
from debate.reviser import Reviser
from debate.manager import DebateManager
from debate.judge import Judge


# ------------------------------------------------------------------ #
# Shared helpers                                                       #
# ------------------------------------------------------------------ #

def make_source(suffix="a"):
    return Source(
        url=f"https://example.com/{suffix}",
        title=f"Source {suffix.upper()}",
        snippet="Test snippet",
    )


def make_agent_evidence(agent_id, claim_text, source, date=None, version=None, scope=None):
    nc = NormalizedClaim(
        text=claim_text,
        source=source,
        date=date,
        version=version,
        scope=scope,
    )
    return AgentEvidence(
        agent_id=agent_id,
        query="Test query",
        status=EvidenceStatus.OK,
        claims=[nc],
        sources=[source],
    )


def make_conflict(conflict_type=ConflictType.FACTUAL):
    source_a = make_source("a")
    source_b = make_source("b")
    claim_a = NormalizedClaim(text="Claim A text", source=source_a)
    claim_b = NormalizedClaim(text="Claim B text", source=source_b)
    return (
        claim_a,
        claim_b,
        Conflict(
            type=conflict_type,
            confidence=0.85,
            explanation=f"{conflict_type.value} conflict",
            claim_a=claim_a,
            claim_b=claim_b,
            metadata={
                "agent_a_id": "agent_a",
                "agent_b_id": "agent_b",
            },
        ),
    )


TOP_CONFLICT = {
    "agent_a_id": "agent_a",
    "agent_b_id": "agent_b",
    "claim_a": None,   # filled in per test
    "claim_b": None,
}


# ------------------------------------------------------------------ #
# Proposer                                                             #
# ------------------------------------------------------------------ #

class TestProposer:
    """Test the Proposer agent (fixture mode)."""

    def test_proposer_creates_argument(self):
        proposer = Proposer(run_mode="fixture")
        source_a = make_source("a")
        source_b = make_source("b")
        claim_a = NormalizedClaim(text="Claim A", source=source_a)
        claim_b = NormalizedClaim(text="Claim B", source=source_b)
        top_conflict = {
            "agent_a_id": "agent_a", "agent_b_id": "agent_b",
            "claim_a": claim_a, "claim_b": claim_b,
        }
        ev_a = make_agent_evidence("agent_a", "Claim A", source_a)
        ev_b = make_agent_evidence("agent_b", "Claim B", source_b)

        argument = proposer.propose(
            query="Test query",
            evidence_list=[ev_a, ev_b],
            conflict_type=ConflictType.TEMPORAL,
            conflict_explanation="Temporal conflict",
            top_conflict=top_conflict,
            round_number=1,
        )

        assert argument is not None
        assert argument.role == "proposer"
        assert argument.round_number == 1
        assert len(argument.content) > 0
        assert argument.strategy_text_preview is not None

    def test_proposer_records_strategy(self):
        """Task 16: strategy_text_preview should be set."""
        proposer = Proposer(run_mode="fixture")
        claim_a = NormalizedClaim(text="A", source=make_source())
        claim_b = NormalizedClaim(text="B", source=make_source("b"))
        top_conflict = {"agent_a_id": "a", "agent_b_id": "b", "claim_a": claim_a, "claim_b": claim_b}
        ev = make_agent_evidence("a", "A", make_source())

        arg = proposer.propose(
            query="q", evidence_list=[ev], conflict_type=ConflictType.FACTUAL,
            conflict_explanation="factual", top_conflict=top_conflict, round_number=1,
        )
        assert arg.strategy_text_preview is not None
        assert len(arg.strategy_text_preview) > 0


# ------------------------------------------------------------------ #
# Critic                                                               #
# ------------------------------------------------------------------ #

class TestCritic:
    """Test the Critic agent (fixture mode)."""

    def test_critic_creates_critiques(self):
        claim_a, claim_b, conflict = make_conflict(ConflictType.FACTUAL)
        top_conflict = {
            "agent_a_id": "agent_a", "agent_b_id": "agent_b",
            "claim_a": claim_a, "claim_b": claim_b,
        }
        source = make_source()
        ev = make_agent_evidence("agent_a", "Claim A text", source)
        proposer = Proposer(run_mode="fixture")
        proposal = proposer.propose(
            query="q", evidence_list=[ev], conflict_type=ConflictType.FACTUAL,
            conflict_explanation="factual", top_conflict=top_conflict, round_number=1,
        )

        critic = Critic(run_mode="fixture")
        critiques = critic.critique(
            query="q", proposal=proposal, evidence_list=[ev],
            conflict_type=ConflictType.FACTUAL,
            top_conflict=top_conflict, round_number=1,
        )

        assert isinstance(critiques, list)
        assert len(critiques) > 0
        assert all(c.role == "critic" for c in critiques)


# ------------------------------------------------------------------ #
# Reviser                                                              #
# ------------------------------------------------------------------ #

class TestReviser:
    """Test the Reviser agent (fixture mode)."""

    def test_reviser_creates_revision(self):
        claim_a, claim_b, conflict = make_conflict(ConflictType.VERSION)
        top_conflict = {
            "agent_a_id": "agent_a", "agent_b_id": "agent_b",
            "claim_a": claim_a, "claim_b": claim_b,
        }
        source = make_source()
        ev = make_agent_evidence("agent_a", "Claim A text", source)

        proposer = Proposer(run_mode="fixture")
        proposal = proposer.propose(
            query="q", evidence_list=[ev], conflict_type=ConflictType.VERSION,
            conflict_explanation="version", top_conflict=top_conflict, round_number=1,
        )

        critic = Critic(run_mode="fixture")
        critiques = critic.critique(
            query="q", proposal=proposal, evidence_list=[ev],
            conflict_type=ConflictType.VERSION, top_conflict=top_conflict, round_number=1,
        )

        reviser = Reviser(run_mode="fixture")
        revision = reviser.revise(
            query="q", proposal=proposal, critiques=critiques,
            evidence_list=[ev], conflict_type=ConflictType.VERSION,
            top_conflict=top_conflict, round_number=1,
        )

        assert revision is not None
        assert revision.role == "reviser"
        assert len(revision.content) > 0


# ------------------------------------------------------------------ #
# DebateManager                                                        #
# ------------------------------------------------------------------ #

class TestDebateManager:
    """Test the DebateManager (Task 14)."""

    def test_manager_returns_debate_outcome(self):
        """run_debate should return DebateOutcome, not a list."""
        _, _, conflict = make_conflict(ConflictType.FACTUAL)
        source = make_source()
        ev_a = make_agent_evidence("agent_a", "Claim A text", source)
        ev_b = make_agent_evidence("agent_b", "Claim B text", source)

        manager = DebateManager(max_rounds=2, run_mode="fixture")
        outcome = manager.run_debate(
            query="Test query",
            evidence_list=[ev_a, ev_b],
            conflict=conflict,
        )

        assert isinstance(outcome, DebateOutcome)
        assert outcome.status in ("completed", "partial", "failed")
        assert outcome.stop_reason in ("converged", "no_objection", "max_rounds", "stage_failure")
        assert isinstance(outcome.rounds_used, int)

    def test_completed_debate_has_rounds(self):
        _, _, conflict = make_conflict(ConflictType.FACTUAL)
        source = make_source()
        ev = make_agent_evidence("agent_a", "A", source)

        manager = DebateManager(max_rounds=1, run_mode="fixture")
        outcome = manager.run_debate("q", [ev, ev], conflict)

        # At least one round should exist
        assert len(outcome.rounds) >= 1

    def test_each_round_has_structure(self):
        _, _, conflict = make_conflict()
        source = make_source()
        ev = make_agent_evidence("agent_a", "Claim text", source)

        manager = DebateManager(max_rounds=2, run_mode="fixture")
        outcome = manager.run_debate("q", [ev, ev], conflict)

        for r in outcome.rounds:
            assert isinstance(r, DebateRound)
            assert r.proposer_argument is not None
            assert len(r.critic_arguments) > 0
            assert r.reviser_argument is not None


# ------------------------------------------------------------------ #
# Judge                                                                #
# ------------------------------------------------------------------ #

class TestJudge:
    """Test the Judge (Task 15)."""

    def test_judge_returns_none_on_failed_debate(self):
        """Task 15: judge must return None if debate did not complete."""
        claim_a, claim_b, conflict = make_conflict(ConflictType.TEMPORAL)
        source = make_source()
        ev = make_agent_evidence("agent_a", "Claim A", source)

        failed_outcome = DebateOutcome(
            rounds=[], status="failed", stop_reason="stage_failure",
            failed_stages=["proposer"], rounds_used=0,
        )

        judge = Judge(run_mode="fixture")
        top_conflict = {
            "agent_a_id": "agent_a", "agent_b_id": "agent_b",
            "claim_a": claim_a, "claim_b": claim_b,
        }
        result = judge.evaluate(
            query="q", evidence_list=[ev],
            debate_outcome=failed_outcome,
            conflict_type=ConflictType.TEMPORAL,
            top_conflict=top_conflict,
        )
        assert result is None

    def test_judge_evaluates_completed_debate(self):
        _, _, conflict = make_conflict(ConflictType.TEMPORAL)
        source = make_source()
        ev = make_agent_evidence("agent_a", "Claim A", source)

        manager = DebateManager(max_rounds=1, run_mode="fixture")
        outcome = manager.run_debate("q", [ev, ev], conflict)

        claim_a, claim_b, _ = make_conflict()
        top_conflict = {
            "agent_a_id": "agent_a", "agent_b_id": "agent_b",
            "claim_a": claim_a, "claim_b": claim_b,
        }
        judge = Judge(run_mode="fixture")
        result = judge.evaluate(
            query="q", evidence_list=[ev],
            debate_outcome=outcome,
            conflict_type=ConflictType.TEMPORAL,
            top_conflict=top_conflict,
        )

        if outcome.status == "completed":
            assert result is not None
            assert result.winner in ("candidate_1", "candidate_2")
            assert 0 <= result.evidence_score <= 1
        else:
            assert result is None


# ------------------------------------------------------------------ #
# MADSystem integration                                                #
# ------------------------------------------------------------------ #

class TestMADSystemIntegration:
    """Integration tests using MADSystem (FIXTURE mode)."""

    def test_solve_returns_trace(self):
        """Task 4: solve() should be synchronous and return ExecutionTrace."""
        from app import MADSystem
        from run_mode import RunMode
        system = MADSystem(run_mode=RunMode.FIXTURE)
        trace = system.solve("What is the capital of Australia?")

        assert trace is not None
        assert trace.query == "What is the capital of Australia?"
        assert "debate_triggered" in trace.metrics
        assert isinstance(trace.metrics["debate_triggered"], bool)
        assert "total_llm_calls_ok" in trace.metrics

    def test_solve_reports_evidence_state(self):
        """Task 9: evidence_state must appear in metrics."""
        from app import MADSystem
        from run_mode import RunMode
        system = MADSystem(run_mode=RunMode.FIXTURE)
        trace = system.solve("What is the capital of Australia?")

        assert "evidence_state" in trace.metrics
        assert trace.metrics["evidence_state"] in (
            "agreement", "disagreement", "uncertain", "insufficient"
        )