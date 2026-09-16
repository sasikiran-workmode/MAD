"""
test_relevance.py — Task 23
Must assert:
  - Unrelated snippets filtered
  - agent with only unrelated hits → NO_RELEVANT_RESULTS
"""
import pytest
from evidence.normalizer import EvidenceNormalizer, _is_relevant
from evidence.models import EvidenceStatus


def test_unrelated_snippets_filtered():
    query = "What is the capital of Australia?"
    
    res1 = {"title": "Australia Capital", "snippet": "Canberra is the capital.", "link": "https://a.com"}
    res2 = {"title": "Minecraft Tips", "snippet": "How to build a base.", "link": "https://b.com"}
    
    # Test internal relevance filter
    assert _is_relevant(query, res1) is True
    assert _is_relevant(query, res2) is False


def test_agent_with_only_unrelated_hits_is_no_relevant_results():
    normalizer = EvidenceNormalizer(llm_client=None)
    query = "What is the capital of Australia?"
    
    # All irrelevant raw results
    raw_results = [
        {"title": "Minecraft Tips", "snippet": "How to build a base.", "link": "https://b.com"},
        {"title": "Africa Safaris", "snippet": "Visit the Sahara.", "link": "https://c.com"},
    ]
    
    # Normalizer should return NO_RELEVANT_RESULTS
    agent_ev = normalizer.normalize("agent_a", query, raw_results)
    
    assert agent_ev.status == EvidenceStatus.NO_RELEVANT_RESULTS
    assert agent_ev.raw_results_count == 2
    assert agent_ev.filtered_out_count == 2
    assert len(agent_ev.claims) == 0
