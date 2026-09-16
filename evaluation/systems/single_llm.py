"""
Single LLM Baseline System (Task 25).
Imports NOTHING from agents/, evidence/, conflict/, debate/.
One LLM call, no retrieval, no NLI, no debate.
This is a genuine isolated baseline — not a stripped MADSystem.
"""
import time
from typing import Optional
from loguru import logger

from evidence.models import ExecutionTrace, FinalAnswer, EvidenceState
from llm_client import LLMClient, LLMError, get_llm_client
from run_mode import RunMode


ANSWER_PROMPT = """Answer the following question directly and concisely.
Provide only the answer without preamble.

Question: {query}

Answer:"""


class SingleLLMSystem:
    """
    True single-LLM baseline: one LLM call, zero retrieval.
    Imports nothing from agents/, evidence/, conflict/, debate/.
    """

    SYSTEM_NAME = "single_llm"

    def __init__(self, run_mode: RunMode = RunMode.LIVE, llm_client: Optional[LLMClient] = None):
        self.run_mode = run_mode
        self.llm_client = llm_client or (
            None if run_mode == RunMode.FIXTURE else get_llm_client()
        )

    def run(self, query: str) -> ExecutionTrace:
        """Run single-LLM query. Exactly 1 LLM call, 0 retrieval."""
        t0 = time.time()
        trace = ExecutionTrace(query=query)

        try:
            if self.run_mode == RunMode.FIXTURE:
                answer_text = f"[FIXTURE] Single LLM answer for: {query}"
                llm_calls_ok = 1
                llm_calls_failed = 0
            elif not self.llm_client:
                raise LLMError("No LLM client configured for SingleLLMSystem")
            else:
                answer_text = self.llm_client.chat(
                    ANSWER_PROMPT.format(query=query),
                    role="single",
                )
                report = self.llm_client.call_report()
                llm_calls_ok = sum(v["ok"] for v in report.values())
                llm_calls_failed = sum(v["failed"] for v in report.values())

            final_answer = FinalAnswer(
                query=query,
                answer=answer_text,
                reasoning="Single LLM direct answer — no retrieval or debate",
                evidence_state=EvidenceState.INSUFFICIENT,   # no evidence retrieved
                sources=[],
                debate_triggered=False,
                independent_agents_agreeing=0,
                source_count=0,
                metadata={"system": self.SYSTEM_NAME},
            )
            trace.final_answer = final_answer
            trace.debate_triggered = False
            trace.metrics = {
                "system": self.SYSTEM_NAME,
                "latency_seconds": time.time() - t0,
                "debate_triggered": False,
                "total_llm_calls_ok": 1 if self.run_mode == RunMode.FIXTURE else llm_calls_ok,
                "total_llm_calls_failed": 0 if self.run_mode == RunMode.FIXTURE else llm_calls_failed,
                "evidence_state": "insufficient",
                "usable_agents": 0,
                "source_count": 0,
            }

        except LLMError as e:
            trace.error = str(e)
            report = self.llm_client.call_report() if self.llm_client else {}
            trace.metrics = {
                "system": self.SYSTEM_NAME,
                "latency_seconds": time.time() - t0,
                "debate_triggered": False,
                "total_llm_calls_ok": sum(v["ok"] for v in report.values()),
                "total_llm_calls_failed": sum(v["failed"] for v in report.values()),
            }

        return trace
