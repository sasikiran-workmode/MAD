"""
test_baselines.py — Task 23
Must assert:
  - Single-LLM trace has agents == [] and exactly 1 LLM call (or 1 mock call in fixture mode)
  - Standard-MAD always debates
  - Proposed gates (agreements don't debate, disagreements do)
"""
import pytest
from evaluation.systems.single_llm import SingleLLMSystem
from evaluation.systems.standard_mad import StandardMADSystem
from evaluation.systems.proposed_mad import ProposedMADSystem
from run_mode import RunMode


def test_single_llm_baseline():
    import sys
    system = SingleLLMSystem(run_mode=RunMode.FIXTURE)
    trace = system.run("What is the capital of Australia?")
    
    assert trace.metrics["system"] == "single_llm"
    assert trace.debate_triggered is False
    assert trace.metrics["debate_triggered"] is False
    # No agents retrieved
    assert not trace.agent_evidence
    # 1 LLM call
    assert trace.metrics["total_llm_calls_ok"] == 1
    
    # Assert SingleLLMSystem is isolated
    with open("evaluation/systems/single_llm.py") as f:
        content = f.read()
    assert "from agents." not in content
    assert "from conflict." not in content
    assert "from debate." not in content


def test_standard_mad_baseline():
    system = StandardMADSystem(run_mode=RunMode.FIXTURE)
    # Using an agreement case (agree_capital)
    trace = system.run("What is the capital of Australia?")
    
    assert trace.metrics["system"] == "standard_mad"
    # Even on agreement, Standard-MAD force-triggers a debate
    assert trace.debate_triggered is True
    assert trace.metrics["debate_triggered"] is True
    assert trace.conflict is not None
    assert trace.conflict.type.value == "unknown"  # Force generic strategy


def test_proposed_mad_baseline():
    system = ProposedMADSystem(run_mode=RunMode.FIXTURE)
    
    # Agreement case -> NO debate
    trace_agree = system.run("What is the capital of Australia?")
    assert trace_agree.metrics["system"] == "proposed_mad"
    assert trace_agree.debate_triggered is False
    assert trace_agree.evidence_decision.state.value == "agreement"
    
    # Disagreement case -> debate
    trace_disagree = system.run("When was India's National Education Policy introduced?")
    assert trace_disagree.debate_triggered is True
    assert trace_disagree.evidence_decision.state.value == "disagreement"
    assert trace_disagree.conflict is not None
    assert trace_disagree.conflict.type.value == "temporal"
