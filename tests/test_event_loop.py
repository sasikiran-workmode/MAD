"""
test_event_loop.py — Task 23
Must assert:
  - solve() 5x in one process, no exception (regression test for Task 4)
"""
import pytest
from app import solve, RunMode


def test_solve_multiple_times_no_event_loop_closed_error():
    """
    Test that running solve() multiple times sequentially doesn't raise
    'Event loop is closed' (regression for Task 4).
    """
    for _ in range(5):
        trace = solve("What is the capital of Australia?", run_mode=RunMode.FIXTURE)
        assert trace.error is None
        assert trace.final_answer is not None
        assert trace.evidence_decision.state.value == "agreement"
