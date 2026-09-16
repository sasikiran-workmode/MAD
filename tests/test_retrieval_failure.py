"""
test_retrieval_failure.py — Task 23
Must assert:
  - 1/2/3 agents failing
  - failed agents appear in decision.failed_agents with reasons
  - no fabricated evidence anywhere in the trace
"""
import pytest
from app import solve, RunMode


def test_one_agent_fails():
    trace = solve("What is the population of Tokyo?", run_mode=RunMode.FIXTURE)
    
    # In fixtures, "one_agent_fails" case has reference_agent failing.
    assert trace.evidence_decision is not None
    assert trace.evidence_decision.state.value == "agreement"
    
    # Assert failed agent is recorded
    failed = trace.evidence_decision.failed_agents
    assert len(failed) == 1
    assert failed[0]["agent_id"] == "reference_agent"
    assert "timeout" in failed[0]["reason"].lower()


def test_two_agents_fail():
    trace = solve("What is the GDP of Iceland?", run_mode=RunMode.FIXTURE)
    
    # In fixtures, "two_agents_fail" case has 2 failing agents.
    assert trace.evidence_decision is not None
    assert trace.evidence_decision.state.value == "insufficient"
    
    # Assert failed agents are recorded
    failed = trace.evidence_decision.failed_agents
    assert len(failed) == 2
    failed_ids = [f["agent_id"] for f in failed]
    assert "reference_agent" in failed_ids
    assert "scholarly_agent" in failed_ids
    
    # Assert no answer content produced
    assert "insufficient evidence" in trace.final_answer.answer.lower()
    
    # Assert NO fabricated evidence (no mock claims)
    for agent_ev in trace.agent_evidence:
        if agent_ev.status == "retrieval_failed":
            assert len(agent_ev.claims) == 0


def test_all_agents_fail():
    trace = solve("Latest AI breakthrough today", run_mode=RunMode.FIXTURE)
    
    # In fixtures, "all_agents_fail" case has 3 failing agents.
    assert trace.evidence_decision is not None
    assert trace.evidence_decision.state.value == "insufficient"
    
    # Assert failed agents are recorded
    failed = trace.evidence_decision.failed_agents
    assert len(failed) == 3
    
    # Assert no answer content produced
    assert "insufficient evidence" in trace.final_answer.answer.lower()
    assert trace.debate_triggered is False
