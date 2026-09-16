"""
Metrics calculator for evaluating the Multi-Agent Debate system.
Tasks applied:
  Task 22: is_trace_ok flag
  Task 24: total_llm_calls_ok / total_llm_calls_failed split
  Task 25: per-role breakdown; retrieval failure rate
  Task 27: Two accuracy metrics:
             1. LLM-as-judge (primary, cached, temperature=0)
             2. Key-fact matching (required_facts / forbidden_facts)
  Task 28: All honest metrics — unnecessary_debate_rate, missed_conflict_rate,
            conflict_classification_accuracy only over LLM_SUCCESS,
            latency_p50/p95, insufficient_evidence_rate, per-role LLM calls,
            avg_debate_rounds, FALLBACK rate.
            Zero hardcoded values anywhere.
"""
import json
import re
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
import pandas as pd
from loguru import logger

from evidence.models import ConflictType, ExecutionTrace, EvidenceState


# ------------------------------------------------------------------ #
# LLM-as-judge grading prompt (Task 27)                               #
# ------------------------------------------------------------------ #

JUDGE_GRADING_PROMPT = """You are evaluating whether a system answer correctly answers a question.

Question: {query}
Reference answer: {reference}
System answer: {answer}

Is the system answer correct? Reply with ONLY "CORRECT" or "INCORRECT" and a one-sentence reason.
Do not add any other text."""


class MetricsCalculator:
    """Calculates evaluation metrics for the MAD system."""

    def __init__(self, llm_client=None):
        self.results: List[ExecutionTrace] = []
        self.llm_client = llm_client      # optional, for LLM-as-judge
        self._judge_cache: Dict[str, bool] = {}   # (query, answer) → bool

    def add_result(self, trace: ExecutionTrace) -> None:
        self.results.append(trace)

    def add_results(self, traces: List[ExecutionTrace]) -> None:
        self.results.extend(traces)

    # ------------------------------------------------------------------ #
    # Task 22: per-trace validity                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def is_trace_ok(trace: ExecutionTrace) -> bool:
        """Return True if the trace produced a real answer without error."""
        return trace.error is None and trace.final_answer is not None

    # ------------------------------------------------------------------ #
    # Task 27: key-fact accuracy (deterministic)                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def key_fact_correct(answer: str, required_facts: List[str], forbidden_facts: List[str]) -> bool:
        """
        Return True iff all required_facts appear in the answer
        and no forbidden_facts appear.
        Case-insensitive substring match.
        """
        a = answer.lower()
        for fact in required_facts:
            if fact.lower() not in a:
                return False
        for fact in forbidden_facts:
            if fact.lower() in a:
                return False
        return True

    # ------------------------------------------------------------------ #
    # Task 27: LLM-as-judge (primary accuracy metric)                    #
    # ------------------------------------------------------------------ #

    def llm_judge_correct(
        self,
        query: str,
        answer: str,
        reference: str,
    ) -> Optional[bool]:
        """
        Return True/False if LLM judge says CORRECT/INCORRECT.
        Returns None if no LLM client or call fails.
        Results are cached.
        """
        if not self.llm_client:
            return None

        cache_key = f"{query}||{answer}"
        if cache_key in self._judge_cache:
            return self._judge_cache[cache_key]

        try:
            from llm_client import LLMError
            prompt = JUDGE_GRADING_PROMPT.format(
                query=query,
                reference=reference,
                answer=answer,
            )
            response = self.llm_client.chat(prompt, role="judge")
            result = "CORRECT" in response.upper().split("\n")[0]
            self._judge_cache[cache_key] = result
            return result
        except Exception as e:
            logger.warning(f"LLM judge failed for '{query[:40]}': {e}")
            return None

    # ------------------------------------------------------------------ #
    # Main metric calculation                                              #
    # ------------------------------------------------------------------ #

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate all evaluation metrics."""
        if not self.results:
            return {"error": "No results to calculate metrics"}

        total = len(self.results)
        ok_traces = [r for r in self.results if self.is_trace_ok(r)]
        ok_count = len(ok_traces)
        error_rate = 1.0 - ok_count / total if total else 1.0

        # ── Task 27: Key-fact accuracy ────────────────────────────────
        kf_correct = 0
        kf_total = 0
        llm_judge_correct = 0
        llm_judge_total = 0

        for trace in ok_traces:
            answer = trace.final_answer.answer if trace.final_answer else ""
            required = trace.router_result.get("required_facts", [])
            forbidden = trace.router_result.get("forbidden_facts", [])
            reference = trace.router_result.get("expected_answer", "")

            if required:
                kf_total += 1
                if self.key_fact_correct(answer, required, forbidden):
                    kf_correct += 1

            if reference:
                verdict = self.llm_judge_correct(trace.query, answer, reference)
                if verdict is not None:
                    llm_judge_total += 1
                    if verdict:
                        llm_judge_correct += 1

        key_fact_accuracy = kf_correct / kf_total if kf_total else 0.0
        llm_judge_accuracy = llm_judge_correct / llm_judge_total if llm_judge_total else 0.0

        # Legacy token-recall for backward compat
        answer_correct = 0
        answer_total = 0
        for trace in ok_traces:
            expected = trace.router_result.get("expected_answer")
            actual = trace.final_answer.answer if trace.final_answer else ""
            if expected:
                answer_total += 1
                e_toks = set(re.findall(r"[a-z0-9]+", expected.lower()))
                a_toks = set(re.findall(r"[a-z0-9]+", actual.lower()))
                if e_toks and len(e_toks & a_toks) / len(e_toks) >= 0.5:
                    answer_correct += 1
        token_recall_accuracy = answer_correct / answer_total if answer_total else 0.0

        # ── Evidence state distribution ───────────────────────────────
        evidence_states = Counter(
            r.evidence_decision.state.value for r in ok_traces if r.evidence_decision
        )
        debate_triggered = sum(1 for r in self.results if r.debate_triggered)
        insufficient_count = evidence_states.get("insufficient", 0)
        insufficient_rate = insufficient_count / ok_count if ok_count else 0.0

        # ── Task 28: Conflict detection ───────────────────────────────
        tp = fp = fn = tn = 0
        for r in self.results:
            expected_conflict = r.router_result.get("expected_conflict", None)
            if expected_conflict is None:
                continue
            detected = r.debate_triggered
            if detected and expected_conflict:
                tp += 1
            elif detected and not expected_conflict:
                fp += 1
            elif not detected and expected_conflict:
                fn += 1
            else:
                tn += 1

        total_labelled = tp + fp + fn + tn
        conflict_detection_acc = (tp + tn) / total_labelled if total_labelled else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        # Task 28: Unnecessary debate (debated when expected_conflict=False)
        unnecessary_debate_count = fp
        unnecessary_debate_rate = fp / (fp + tn) if (fp + tn) else 0.0

        # Task 28: Missed conflict (did not debate when expected_conflict=True)
        missed_conflict_rate = fn / (fn + tp) if (fn + tp) else 0.0

        # ── Task 28: Conflict classification — only LLM_SUCCESS ───────
        correct_cls_llm = 0
        total_cls_llm = 0
        correct_cls_fallback = 0
        total_cls_fallback = 0
        for r in self.results:
            if r.conflict and r.router_result.get("expected_conflict_type"):
                status = r.metrics.get("classification_status", "")
                correct = r.conflict.type.value == r.router_result["expected_conflict_type"]
                if status == "llm_success":
                    total_cls_llm += 1
                    if correct:
                        correct_cls_llm += 1
                elif status == "fallback":
                    total_cls_fallback += 1
                    if correct:
                        correct_cls_fallback += 1
        cls_acc_llm = correct_cls_llm / total_cls_llm if total_cls_llm else 0.0
        cls_acc_fallback = correct_cls_fallback / total_cls_fallback if total_cls_fallback else 0.0

        # ── Task 28: Latency p50/p95 ──────────────────────────────────
        latencies = sorted(r.metrics.get("latency_seconds", 0) for r in ok_traces)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p50 = statistics.median(latencies) if latencies else 0.0
        p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0

        # ── Task 24: LLM calls (ok/failed split) ─────────────────────
        llm_ok_counts = [r.metrics.get("total_llm_calls_ok", 0) for r in ok_traces]
        llm_fail_counts = [r.metrics.get("total_llm_calls_failed", 0) for r in ok_traces]
        avg_llm_ok = sum(llm_ok_counts) / len(llm_ok_counts) if llm_ok_counts else 0.0
        avg_llm_failed = sum(llm_fail_counts) / len(llm_fail_counts) if llm_fail_counts else 0.0

        # Per-role breakdown (Task 24)
        role_totals: Dict[str, Dict[str, int]] = {}
        for r in ok_traces:
            by_role = r.metrics.get("llm_calls_by_role", {})
            for role, counts in by_role.items():
                if role not in role_totals:
                    role_totals[role] = {"ok": 0, "failed": 0}
                role_totals[role]["ok"] += counts.get("ok", 0)
                role_totals[role]["failed"] += counts.get("failed", 0)

        # ── Debate rounds ─────────────────────────────────────────────
        rounds_used = [
            r.metrics.get("debate_rounds_used", 0)
            for r in self.results if r.debate_triggered
        ]
        avg_debate_rounds = sum(rounds_used) / len(rounds_used) if rounds_used else 0.0

        # Debate stop reason distribution (Task 30)
        stop_reasons = Counter(
            r.metrics.get("debate_stop_reason") or "none"
            for r in self.results if r.debate_triggered
        )

        # ── Task 25/28: Retrieval health ──────────────────────────────
        retrieval_failure_rates = [r.metrics.get("retrieval_failure_rate", 0.0) for r in ok_traces]
        avg_retrieval_failure = sum(retrieval_failure_rates) / len(retrieval_failure_rates) if retrieval_failure_rates else 0.0
        irrelevant_counts = [r.metrics.get("irrelevant_evidence_rate", 0) for r in ok_traces]
        avg_irrelevant = sum(irrelevant_counts) / len(irrelevant_counts) if irrelevant_counts else 0.0
        agents_usable = [r.metrics.get("usable_agents", 0) for r in ok_traces]
        avg_usable_agents = sum(agents_usable) / len(agents_usable) if agents_usable else 0.0

        # ── Task 28: Efficiency headline ─────────────────────────────
        # debates_avoided × avg LLM calls saved per avoided debate
        debates_triggered = debate_triggered
        total_queries = total
        debates_avoided = total_queries - debates_triggered
        avg_calls_per_debate = avg_llm_ok  # approximate
        efficiency_calls_saved = debates_avoided * avg_calls_per_debate if debates_triggered > 0 else 0.0

        # ── Classification status distribution ────────────────────────
        cls_status = Counter(
            r.metrics.get("classification_status") or "none"
            for r in self.results if r.debate_triggered
        )

        # ── Conflict type distribution ────────────────────────────────
        conflict_types = [r.conflict.type.value for r in self.results if r.conflict]

        return {
            # Counts
            "total_samples": total,
            "ok_count": ok_count,
            "error_rate": error_rate,

            # Task 27: Primary accuracy metrics
            "key_fact_accuracy": key_fact_accuracy,
            "llm_judge_accuracy": llm_judge_accuracy,
            "llm_judge_samples_evaluated": llm_judge_total,
            # Legacy / secondary
            "token_recall_accuracy": token_recall_accuracy,
            "answer_accuracy": token_recall_accuracy,   # backward compat

            # Task 28: Debate trigger
            "debate_trigger_rate": debates_triggered / total if total else 0.0,
            "evidence_agreement_rate": (total - debates_triggered) / total if total else 0.0,
            "debate_triggered": debates_triggered,
            "debates_avoided": debates_avoided,
            "unnecessary_debate_rate": unnecessary_debate_rate,
            "missed_conflict_rate": missed_conflict_rate,
            "insufficient_evidence_rate": insufficient_rate,

            # Confusion matrix
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "conflict_detection_accuracy": conflict_detection_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,

            # Task 28: Classification accuracy (LLM_SUCCESS only)
            "conflict_classification_accuracy_llm_success": cls_acc_llm,
            "conflict_classification_accuracy_fallback": cls_acc_fallback,
            "conflict_classification_llm_success_count": total_cls_llm,
            "conflict_classification_fallback_count": total_cls_fallback,

            # Latency (Task 28)
            "avg_latency_seconds": avg_latency,
            "latency_p50": p50,
            "latency_p95": p95,

            # Task 24: LLM calls split
            "avg_llm_calls_ok": avg_llm_ok,
            "avg_llm_calls_failed": avg_llm_failed,
            "avg_llm_calls": avg_llm_ok,   # backward compat
            "llm_calls_by_role_totals": role_totals,

            # Debate structure (Task 30)
            "avg_debate_rounds": avg_debate_rounds,
            "debate_stop_reason_distribution": dict(stop_reasons),

            # Task 28: Efficiency
            "efficiency_llm_calls_saved": efficiency_calls_saved,

            # Retrieval health
            "avg_retrieval_failure_rate": avg_retrieval_failure,
            "avg_irrelevant_evidence_count": avg_irrelevant,
            "avg_usable_agents_per_query": avg_usable_agents,

            # Distributions
            "conflict_type_distribution": dict(Counter(conflict_types)),
            "evidence_state_distribution": dict(evidence_states),
            "classification_status_distribution": dict(cls_status),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = []
        for trace in self.results:
            row = {
                "query": trace.query,
                "trace_ok": self.is_trace_ok(trace),
                "evidence_state": (
                    trace.evidence_decision.state.value if trace.evidence_decision else "unknown"
                ),
                "debate_triggered": trace.debate_triggered,
                "conflict_type": trace.conflict.type.value if trace.conflict else None,
                "classification_status": trace.metrics.get("classification_status"),
                "debate_status": trace.metrics.get("debate_status"),
                "debate_stop_reason": trace.metrics.get("debate_stop_reason"),
                "debate_rounds_used": trace.metrics.get("debate_rounds_used", 0),
                "latency": trace.metrics.get("latency_seconds", 0),
                "llm_calls_ok": trace.metrics.get("total_llm_calls_ok", 0),
                "llm_calls_failed": trace.metrics.get("total_llm_calls_failed", 0),
                "retrieval_failure_rate": trace.metrics.get("retrieval_failure_rate", 0),
                "irrelevant_evidence_count": trace.metrics.get("irrelevant_evidence_rate", 0),
                "usable_agents": trace.metrics.get("usable_agents", 0),
                "independent_agents_agreeing": trace.metrics.get("independent_agents_agreeing", 0),
                "source_count": trace.metrics.get("source_count", 0),
                "contradiction_score": trace.contradiction_score,
                "judge_winner": trace.judge_result.winner if trace.judge_result else None,
                "judge_confidence": trace.judge_result.confidence if trace.judge_result else None,
                "error": trace.error,
            }
            data.append(row)
        return pd.DataFrame(data)

    def save_metrics(self, filepath: str) -> None:
        metrics = self.calculate_metrics()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Metrics saved to {filepath}")

    def save_results_csv(self, filepath: str) -> None:
        df = self.to_dataframe()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Results saved to {filepath}")

    def print_comparison_table(
        self,
        single_llm_metrics: Dict[str, Any],
        standard_mad_metrics: Dict[str, Any],
        proposed_metrics: Dict[str, Any],
        ablations: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        """Print a comparison table using only measured values."""
        lines = [
            "",
            "=" * 90,
            "SYSTEM COMPARISON (all values from measured traces)",
            "=" * 90,
            "",
            f"{'System':<28} {'KF Acc':<10} {'Debate%':<10} {'LLM ok':<10} {'Latency(s)':<12} {'Ret.Fail%':<10}",
            "-" * 90,
        ]

        def _row(name: str, m: Dict[str, Any]) -> str:
            kf = m.get("key_fact_accuracy", 0) * 100
            dr = m.get("debate_trigger_rate", 0) * 100
            llm = m.get("avg_llm_calls_ok", m.get("avg_llm_calls", 0))
            lat = m.get("avg_latency_seconds", 0)
            rf = m.get("avg_retrieval_failure_rate", 0) * 100
            return f"{name:<28} {kf:>5.1f}%    {dr:>5.1f}%    {llm:>6.1f}    {lat:>8.2f}s    {rf:>5.1f}%"

        lines.append(_row("Single LLM", single_llm_metrics))
        lines.append(_row("Standard MAD", standard_mad_metrics))
        lines.append(_row("Proposed MAD", proposed_metrics))
        if ablations:
            for name, m in ablations.items():
                lines.append(_row(f"  ↳ {name}", m))
        lines += [
            "-" * 90,
            "",
            "PROPOSED MAD — DETAILED METRICS:",
            f"  Key-Fact Accuracy:      {proposed_metrics.get('key_fact_accuracy', 0)*100:.1f}%",
            f"  LLM-Judge Accuracy:     {proposed_metrics.get('llm_judge_accuracy', 0)*100:.1f}%  (n={proposed_metrics.get('llm_judge_samples_evaluated', 0)})",
            f"  Conflict Detection:     P={proposed_metrics.get('precision', 0)*100:.1f}% R={proposed_metrics.get('recall', 0)*100:.1f}% F1={proposed_metrics.get('f1', 0)*100:.1f}%",
            f"  Cls Acc (LLM_SUCCESS):  {proposed_metrics.get('conflict_classification_accuracy_llm_success', 0)*100:.1f}%  (n={proposed_metrics.get('conflict_classification_llm_success_count', 0)})",
            f"  Cls Acc (FALLBACK):     {proposed_metrics.get('conflict_classification_accuracy_fallback', 0)*100:.1f}%  (n={proposed_metrics.get('conflict_classification_fallback_count', 0)})",
            f"  Unnecessary Debates:    {proposed_metrics.get('unnecessary_debate_rate', 0)*100:.1f}%",
            f"  Missed Conflicts:       {proposed_metrics.get('missed_conflict_rate', 0)*100:.1f}%",
            f"  Latency p50/p95:        {proposed_metrics.get('latency_p50', 0):.2f}s / {proposed_metrics.get('latency_p95', 0):.2f}s",
            f"  Insufficient Evidence:  {proposed_metrics.get('insufficient_evidence_rate', 0)*100:.1f}%",
            f"  LLM Calls Saved:        {proposed_metrics.get('efficiency_llm_calls_saved', 0):.1f}",
            "",
        ]
        return "\n".join(lines)


def get_metrics_calculator(llm_client=None) -> MetricsCalculator:
    """Get or create metrics calculator instance."""
    return MetricsCalculator(llm_client=llm_client)