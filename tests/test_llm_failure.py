"""
test_llm_failure.py — Task 23
Must assert:
  - Mock client raising on proposer/critic/reviser/judge
  - status != "completed"
  - judge_result is None
  - trace never reports 3 successful rounds
"""
import pytest
from llm_client import LLMClient, LLMError
from evidence.models import ConflictType
from debate.manager import DebateManager
from debate.proposer import Proposer
from debate.critic import Critic
from debate.reviser import Reviser
from tests.test_debate import make_agent_evidence, make_conflict, make_source


class FailingLLMClient(LLMClient):
    """An LLM client that always fails."""
    def __init__(self, fail_on_role=None):
        super().__init__(api_key="fake")
        self.fail_on_role = fail_on_role
        
    def chat(self, prompt, system_prompt=None, *, role: str) -> str:
        if self.fail_on_role is None or role == self.fail_on_role:
            raise LLMError(f"Simulated failure for role {role}")
        return '{"claim": "Simulated", "reasoning": "Sim", "sources": [], "strategy_text_preview": "preview"}'
        
    def chat_json(self, prompt, system_prompt=None, *, role: str) -> dict:
        if self.fail_on_role is None or role == self.fail_on_role:
            raise LLMError(f"Simulated JSON failure for role {role}")
        return {"substantive_objection": True, "critiques": ["Simulated critique"]}


def test_debate_manager_handles_proposer_failure():
    """Test when the proposer fails, the debate halts and is marked failed."""
    failing_client = FailingLLMClient(fail_on_role="proposer")
    manager = DebateManager(max_rounds=3, run_mode="live")
    # inject the failing client directly into components
    manager.proposer = Proposer(llm_client=failing_client)
    manager.critic = Critic(llm_client=failing_client)
    manager.reviser = Reviser(llm_client=failing_client)
    
    _, _, conflict = make_conflict(ConflictType.FACTUAL)
    source = make_source()
    ev_a = make_agent_evidence("agent_a", "Claim A", source)
    ev_b = make_agent_evidence("agent_b", "Claim B", source)
    
    outcome = manager.run_debate("q", [ev_a, ev_b], conflict)
    
    # Assert status != "completed"
    assert outcome.status == "failed"
    assert outcome.stop_reason == "stage_failure"
    assert "proposer" in outcome.failed_stages
    
    # Assert trace never reports 3 successful rounds
    assert len(outcome.rounds) == 1
    assert outcome.rounds_used == 0


def test_debate_manager_handles_critic_failure():
    """Test when the critic fails, the debate halts."""
    failing_client = FailingLLMClient(fail_on_role="critic")
    manager = DebateManager(max_rounds=3, run_mode="live")
    manager.proposer = Proposer(llm_client=failing_client)
    manager.critic = Critic(llm_client=failing_client)
    manager.reviser = Reviser(llm_client=failing_client)
    
    _, _, conflict = make_conflict(ConflictType.FACTUAL)
    source = make_source()
    ev_a = make_agent_evidence("agent_a", "Claim A", source)
    ev_b = make_agent_evidence("agent_b", "Claim B", source)
    
    outcome = manager.run_debate("q", [ev_a, ev_b], conflict)
    
    assert outcome.status == "failed"
    assert outcome.stop_reason == "stage_failure"
    assert "critic" in outcome.failed_stages
    
    # Proposer succeeded, but round didn't complete
    assert len(outcome.rounds) == 1
    assert outcome.rounds_used == 0


def test_judge_failure():
    """Test when judge fails, judge_result is None (Task 15/23)."""
    from debate.judge import Judge
    
    failing_client = FailingLLMClient(fail_on_role="judge")
    judge = Judge(llm_client=failing_client, run_mode="live")
    
    _, _, conflict = make_conflict(ConflictType.FACTUAL)
    source = make_source()
    ev = make_agent_evidence("agent_a", "Claim A", source)
    
    # Need a completed debate outcome to pass to judge
    manager = DebateManager(max_rounds=1, run_mode="fixture")
    outcome = manager.run_debate("q", [ev, ev], conflict)
    
    claim_a, claim_b, _ = make_conflict()
    top_conflict = {
        "agent_a_id": "agent_a", "agent_b_id": "agent_b",
        "claim_a": claim_a, "claim_b": claim_b,
    }
    
    result = judge.evaluate(
        query="q", evidence_list=[ev, ev],
        debate_outcome=outcome,
        conflict_type=ConflictType.FACTUAL,
        top_conflict=top_conflict,
    )
    
    # Assert judge_result is None on failure
    assert result is None
