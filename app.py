"""
Main application for the Multi-Agent Debate framework.
Tasks applied:
  Task 2:  validate() called on startup
  Task 4:  fully synchronous pipeline (no asyncio.run)
  Task 6:  _create_default_evidence deleted
  Task 9:  EvidenceAggregator replaces [0]/[1] comparison
  Task 10: INSUFFICIENT retry logic; UNCERTAIN skips debate
  Task 11: ConflictClassifier returns ClassificationResult with status
  Task 12: Conflict built from guaranteed NormalizedClaims (top_conflict)
  Task 22: RunMode enum; FixtureStore raises on unknown query; no mixed modes
  Task 24: _count_llm_calls deleted; use llm_client.call_report()
"""
import json
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
from loguru import logger

from config import settings
from run_mode import RunMode, get_fixture_store, FixtureNotFoundError
from agents import router
from evidence.models import (
    AgentEvidence, EvidenceState, EvidenceStatus,
    Conflict, ConflictType, ExecutionTrace, FinalAnswer, NormalizedClaim, Source,
)
from evidence.aggregator import EvidenceAggregator
from evidence.normalizer import EvidenceNormalizer
from evidence.disagreement_detector import MockNLIDetector, get_nli_detector
from conflict.classifier import get_conflict_classifier
from debate.manager import get_debate_manager
from debate.judge import get_judge
from output.generator import get_answer_generator
from llm_client import LLMClient, LLMError, get_llm_client


class MADSystem:
    """
    Multi-Agent Debate system with evidence-triggered debate
    and conflict-type-aware debate strategy.
    Task 22: RunMode is passed explicitly to every component — no component
             reads settings.demo_mode on its own.
    """

    def __init__(self, run_mode: RunMode = RunMode.LIVE):
        # Accept legacy bool for backward compat (demo_mode=True → FIXTURE)
        if isinstance(run_mode, bool):
            run_mode = RunMode.FIXTURE if run_mode else RunMode.LIVE
        self.run_mode: RunMode = run_mode
        self.demo_mode = (run_mode == RunMode.FIXTURE)  # kept for compat

        # Validate configuration (LIVE only)
        if self.run_mode == RunMode.LIVE:
            settings.validate()

        # Fixture store (FIXTURE only)
        self.fixture_store = get_fixture_store() if self.run_mode == RunMode.FIXTURE else None

        # Shared LLM client — None in FIXTURE mode; real in LIVE
        self.llm_client: Optional[LLMClient] = (
            None if self.run_mode == RunMode.FIXTURE else get_llm_client()
        )

        # NLI detector — MockNLIDetector in FIXTURE, real model in LIVE
        if self.run_mode == RunMode.FIXTURE or settings.nli_mode == "mock":
            self.nli_detector = MockNLIDetector(threshold=settings.disagreement_threshold)
        else:
            self.nli_detector = get_nli_detector()

        # Pipeline components — all receive run_mode explicitly
        self.normalizer = EvidenceNormalizer(llm_client=self.llm_client)
        self.aggregator = EvidenceAggregator(nli_detector=self.nli_detector)
        self.conflict_classifier = get_conflict_classifier(self.llm_client)
        self.debate_manager = get_debate_manager(
            self.llm_client, run_mode=self.run_mode
        )
        self.judge = get_judge(self.llm_client, run_mode=self.run_mode)
        self.answer_generator = get_answer_generator(self.llm_client)

        # Initialize router (sync)
        router.initialize()

    def solve(self, query: str) -> ExecutionTrace:
        """
        Main solve pipeline (fully synchronous).

        Args:
            query: User question

        Returns:
            ExecutionTrace with complete results
        """
        trace = ExecutionTrace(query=query)
        start_time = time.time()

        try:
            logger.info(f"Solving: {query!r}")

            if self.run_mode == RunMode.FIXTURE:
                # Task 22: FIXTURE mode intercepts retrieval completely
                fixture_case = self.fixture_store.get_by_query(query)
                routing_result = {
                    "decision": {"agents": list(fixture_case["agents"].keys()), "reasoning": "Fixture mode bypass"},
                    "evidence": []
                }
                trace.router_result = routing_result
                
                agent_evidence: List[AgentEvidence] = []
                for agent_id, data in fixture_case["agents"].items():
                    status_str = data.get("status", "ok")
                    if status_str == "ok":
                        status = EvidenceStatus.OK
                    elif status_str == "no_relevant_results":
                        status = EvidenceStatus.NO_RELEVANT_RESULTS
                    else:
                        status = EvidenceStatus.RETRIEVAL_FAILED
                        
                    if status == EvidenceStatus.OK:
                        src_dict = data.get("source") or {}
                        src = Source(
                            url=src_dict.get("url", ""),
                            title=src_dict.get("title", ""),
                            snippet=src_dict.get("snippet", "")
                        )
                        nc = NormalizedClaim(
                            text=data.get("claim", ""),
                            source=src,
                            date=data.get("date"),
                            version=data.get("version"),
                            scope=data.get("scope")
                        )
                        ev = AgentEvidence(
                            agent_id=agent_id, query=query, status=status,
                            claims=[nc], sources=[src]
                        )
                    else:
                        ev = AgentEvidence(
                            agent_id=agent_id, query=query, status=status,
                            failure_reason=data.get("failure_reason", "failed")
                        )
                    agent_evidence.append(ev)
                trace.agent_evidence = agent_evidence
            else:
                # ----------------------------------------------------------------
                # 1. Route query to agents and retrieve raw results
                # ----------------------------------------------------------------
                routing_result = router.execute(query)
                trace.router_result = routing_result

                # ----------------------------------------------------------------
                # 2. Normalize each agent's results → AgentEvidence
                # ----------------------------------------------------------------
                agent_evidence: List[AgentEvidence] = []
                for result in routing_result.get("evidence", []):
                    agent_id = result.get("agent_id", "unknown")

                    if "error" in result:
                        agent_evidence.append(AgentEvidence(
                            agent_id=agent_id,
                            query=query,
                            status=EvidenceStatus.RETRIEVAL_FAILED,
                            failure_reason=result["error"],
                        ))
                        continue

                    raw_ev_dict = result.get("evidence", {})
                    raw_results = []
                    for src in raw_ev_dict.get("sources", []):
                        raw_results.append({
                            "title": src.get("title", ""),
                            "snippet": src.get("snippet", ""),
                            "link": src.get("url", ""),
                        })

                    normalized = self.normalizer.normalize(agent_id, query, raw_results)
                    agent_evidence.append(normalized)

                trace.agent_evidence = agent_evidence

            # ----------------------------------------------------------------
            # 3. Aggregate decision (Task 9)
            # ----------------------------------------------------------------
            decision = self.aggregator.decide(agent_evidence)
            trace.evidence_decision = decision
            trace.disagreement_detected = decision.state == EvidenceState.DISAGREEMENT
            trace.contradiction_score = (
                max((p.contradiction_score for p in decision.pairwise), default=0.0)
            )

            # ----------------------------------------------------------------
            # 4. Recovery for INSUFFICIENT (Task 10)
            # ----------------------------------------------------------------
            if decision.state == EvidenceState.INSUFFICIENT:
                decision = self._retry_failed_agents(query, agent_evidence, decision)
                trace.evidence_decision = decision

            # ----------------------------------------------------------------
            # 5. Branch on decision state
            # ----------------------------------------------------------------
            if decision.state == EvidenceState.AGREEMENT:
                # --- AGREE: generate direct answer, no debate
                logger.info("Evidence AGREEMENT — skipping debate")
                trace.debate_triggered = False
                final_answer = self.answer_generator.generate(
                    query=query,
                    decision=decision,
                    agent_evidence=agent_evidence,
                )
                trace.final_answer = final_answer

            elif decision.state == EvidenceState.DISAGREEMENT:
                # --- DISAGREE: classify conflict, run debate, judge
                logger.info("Evidence DISAGREEMENT — triggering debate")
                trace.debate_triggered = True

                # Task 11 guard: only classify on DISAGREEMENT
                top_conflict = decision.top_conflict
                claim_a_text = top_conflict["claim_a"].text if top_conflict else ""
                claim_b_text = top_conflict["claim_b"].text if top_conflict else ""

                classification = self.conflict_classifier.classify(
                    claim_a=claim_a_text,
                    claim_b=claim_b_text,
                    query=query,
                )

                # Task 12: Conflict built from guaranteed NormalizedClaims
                conflict = Conflict(
                    type=classification.type,
                    confidence=classification.confidence,
                    explanation=classification.explanation,
                    claim_a=top_conflict["claim_a"],
                    claim_b=top_conflict["claim_b"],
                    metadata={
                        "classification_status": classification.status.value,
                        "agent_a_id": top_conflict.get("agent_a_id", ""),
                        "agent_b_id": top_conflict.get("agent_b_id", ""),
                    },
                )
                trace.conflict = conflict

                # Run debate
                usable = [e for e in agent_evidence if e.is_usable]
                debate_outcome = self.debate_manager.run_debate(
                    query=query,
                    evidence_list=usable,
                    conflict=conflict,
                )
                trace.debate_outcome = debate_outcome
                trace.debate_rounds = debate_outcome.rounds

                # Judge (Task 15: only if completed)
                judge_result = self.judge.evaluate(
                    query=query,
                    evidence_list=usable,
                    debate_outcome=debate_outcome,
                    conflict_type=conflict.type,
                    top_conflict=top_conflict,
                )
                trace.judge_result = judge_result

                final_answer = self.answer_generator.generate(
                    query=query,
                    decision=decision,
                    agent_evidence=agent_evidence,
                    conflict=conflict,
                    debate_outcome=debate_outcome,
                    judge_result=judge_result,
                )
                trace.final_answer = final_answer

            elif decision.state == EvidenceState.UNCERTAIN:
                # --- UNCERTAIN: skip debate, report findings
                logger.info("Evidence UNCERTAIN — skipping debate (do not debate unrelated claims)")
                trace.debate_triggered = False
                final_answer = self.answer_generator.generate(
                    query=query,
                    decision=decision,
                    agent_evidence=agent_evidence,
                )
                trace.final_answer = final_answer

            else:
                # --- INSUFFICIENT (after retry)
                logger.info("Evidence INSUFFICIENT — reporting failure")
                trace.debate_triggered = False
                final_answer = self.answer_generator.generate(
                    query=query,
                    decision=decision,
                    agent_evidence=agent_evidence,
                )
                trace.final_answer = final_answer

            # ----------------------------------------------------------------
            # 6. Record metrics (Task 24)
            # ----------------------------------------------------------------
            end_time = time.time()
            call_report = self.llm_client.call_report() if self.llm_client else {}
            total_ok = sum(v["ok"] for v in call_report.values()) if call_report else 0
            total_failed = sum(v["failed"] for v in call_report.values()) if call_report else 0

            trace.metrics = {
                "latency_seconds": end_time - start_time,
                "evidence_state": decision.state.value,
                "debate_triggered": trace.debate_triggered,
                "conflict_type": trace.conflict.type.value if trace.conflict else None,
                "classification_status": (
                    trace.conflict.metadata.get("classification_status")
                    if trace.conflict else None
                ),
                "debate_status": (
                    trace.debate_outcome.status if trace.debate_outcome else None
                ),
                "debate_stop_reason": (
                    trace.debate_outcome.stop_reason if trace.debate_outcome else None
                ),
                "debate_rounds_used": (
                    trace.debate_outcome.rounds_used if trace.debate_outcome else 0
                ),
                "total_llm_calls_ok": total_ok,
                "total_llm_calls_failed": total_failed,
                "llm_calls_by_role": call_report,
                "retrieval_failure_rate": sum(
                    1 for e in agent_evidence
                    if e.status == EvidenceStatus.RETRIEVAL_FAILED
                ) / max(len(agent_evidence), 1),
                "irrelevant_evidence_rate": sum(e.filtered_out_count for e in agent_evidence),
                "usable_agents": len(decision.usable_agents),
                "failed_agents": len(decision.failed_agents),
                "independent_agents_agreeing": (
                    trace.final_answer.independent_agents_agreeing
                    if trace.final_answer else 0
                ),
                "source_count": (
                    trace.final_answer.source_count
                    if trace.final_answer else 0
                ),
            }

            trace.completed_at = datetime.now()
            save_trace(trace)

            logger.info(
                f"Solved in {trace.metrics['latency_seconds']:.2f}s | "
                f"state={decision.state.value} | "
                f"llm_ok={total_ok} | llm_failed={total_failed}"
            )

        except Exception as e:
            logger.error(f"Solve pipeline failed: {e}", exc_info=True)
            trace.error = f"{type(e).__name__}: {e}"
            trace.completed_at = datetime.now()
            save_trace(trace)

        return trace

    # ------------------------------------------------------------------ #
    # Recovery                                                             #
    # ------------------------------------------------------------------ #

    def _retry_failed_agents(self, query: str, agent_evidence: List[AgentEvidence], decision) -> object:
        """
        Task 10: Retry failed/empty agents once; re-run aggregator.
        Task 22: Skip retry in FIXTURE mode.
        Returns updated EvidenceDecision.
        """
        if self.run_mode == RunMode.FIXTURE:
            logger.info("FIXTURE mode: skipping retry for failed agents")
            return decision

        from agents.base_agent import registry
        failed_ids = {f["agent_id"] for f in decision.failed_agents}
        if not failed_ids:
            return decision

        logger.info(f"INSUFFICIENT: retrying {failed_ids}")

        updated_evidence = [e for e in agent_evidence if e.agent_id not in failed_ids]

        for agent_id in failed_ids:
            agent = registry.get(agent_id)
            if agent is None:
                continue
            try:
                raw_ev = agent.retrieve(query)
                raw_results = [
                    {"title": s.title, "snippet": s.snippet, "link": s.url}
                    for s in raw_ev.sources
                ]
                normalized = self.normalizer.normalize(agent_id, query, raw_results)
                updated_evidence.append(normalized)
            except Exception as exc:
                logger.warning(f"Retry failed for {agent_id}: {exc}")
                updated_evidence.append(AgentEvidence(
                    agent_id=agent_id,
                    query=query,
                    status=EvidenceStatus.RETRIEVAL_FAILED,
                    failure_reason=f"Retry also failed: {exc}",
                ))

        return self.aggregator.decide(updated_evidence)


# ------------------------------------------------------------------ #
# Save trace                                                           #
# ------------------------------------------------------------------ #

def save_trace(trace: ExecutionTrace, results_dir: str = "results") -> str:
    """Save execution trace to JSON file."""
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)

    filename = f"trace_{trace.trace_id[:8]}.json"
    filepath = results_path / filename

    with open(filepath, "w") as f:
        json.dump(trace.model_dump(mode="json"), f, indent=2, default=str)

    logger.info(f"Trace saved to {filepath}")
    return str(filepath)


# ------------------------------------------------------------------ #
# Top-level solve function                                             #
# ------------------------------------------------------------------ #

def solve(query: str, demo_mode: bool = False, run_mode: "RunMode | None" = None) -> ExecutionTrace:
    """
    Main solve function (synchronous).

    Args:
        query: User question
        run_mode: RunMode.FIXTURE or RunMode.LIVE (overrides demo_mode)
        demo_mode: Backward-compat bool; True = FIXTURE mode

    Returns:
        ExecutionTrace with results
    """
    if run_mode is None:
        run_mode = RunMode.FIXTURE if demo_mode else RunMode.LIVE
    system = MADSystem(run_mode=run_mode)
    return system.solve(query)


# ------------------------------------------------------------------ #
# Interactive CLI                                                      #
# ------------------------------------------------------------------ #

def run_interactive():
    """Run interactive CLI mode."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    import sys

    console = Console()
    console.print(Panel.fit(
        "[bold blue]Evidence-MAD: Multi-Agent Debate System[/]\n"
        "[dim]Independent retrieval → Evidence comparison → Conflict-aware debate[/]",
        border_style="blue",
    ))

    use_demo = "--demo" in sys.argv
    run_mode = RunMode.FIXTURE if use_demo else RunMode.LIVE

    if use_demo:
        console.print("[yellow]● DEMO MODE — pre-loaded evidence fixtures[/]")
        console.print("[dim]Demo queries: 'What is the capital of Australia?' / "
                      "\"When was India's National Education Policy introduced?\" / "
                      "'Is aspirin safe to take?' / 'What new features does Python 3.13 have?'[/]")
    else:
        console.print("[green]● LIVE MODE — real search + LLM[/]")

    try:
        system = MADSystem(run_mode=run_mode)
    except Exception as e:
        console.print(f"[red bold]Startup error:[/] {e}")
        sys.exit(1)

    while True:
        console.print("\n[bold]Enter your query (or 'quit' to exit):[/]")
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/]")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/]")
            break

        with console.status("[bold green]Retrieving evidence and reasoning…[/]"):
            trace = system.solve(query)

        # ── Error ────────────────────────────────────────────────────
        if trace.error:
            err = trace.error
            if "FixtureNotFoundError" in err or "unknown query" in err:
                console.print(
                    "[yellow]No demo fixture for that query.[/]\n"
                    "[dim]Try: 'What is the capital of Australia?' or "
                    "\"When was India's National Education Policy introduced?\"[/]"
                )
            else:
                console.print(f"[red]Error:[/] {err}")
            continue

        console.print()

        # ════════════════════════════════════════════════════════════
        # STEP 1 — What each agent retrieved
        # ════════════════════════════════════════════════════════════
        console.print("[bold cyan]━━━  STEP 1: Independent Agent Evidence  ━━━[/]")

        agent_table = Table(box=box.ROUNDED, show_lines=True, expand=True)
        agent_table.add_column("Agent",        style="cyan bold",  width=18, no_wrap=True)
        agent_table.add_column("Claim",        style="white",      ratio=3)
        agent_table.add_column("Date/Version", style="dim",        width=14, no_wrap=True)
        agent_table.add_column("Status",       style="bold",       width=14, no_wrap=True)

        agent_provider = {
            "web_agent":       "Web (DuckDuckGo)",
            "reference_agent": "Reference (Wikipedia)",
            "scholarly_agent": "Scholarly (Crossref)",
        }

        if trace.agent_evidence:
            for ev in trace.agent_evidence:
                provider = agent_provider.get(ev.agent_id, ev.agent_id)
                if ev.is_usable and ev.claims:
                    claim_text = ev.claims[0].text
                    date_ver = ev.claims[0].date or ev.claims[0].version or "—"
                    status_str = "[green]✓ OK[/]"
                elif ev.status.value == "retrieval_failed":
                    claim_text = f"[red]{ev.failure_reason or 'Failed'}[/]"
                    date_ver = "—"
                    status_str = "[red]✗ FAILED[/]"
                else:
                    claim_text = f"[dim]{ev.failure_reason or 'No relevant results'}[/]"
                    date_ver = "—"
                    status_str = "[yellow]~ NO DATA[/]"
                agent_table.add_row(provider, claim_text, date_ver, status_str)

        console.print(agent_table)

        # ════════════════════════════════════════════════════════════
        # STEP 2 — Evidence decision
        # ════════════════════════════════════════════════════════════
        console.print()
        console.print("[bold cyan]━━━  STEP 2: Evidence Comparison & Gate Decision  ━━━[/]")

        decision = trace.evidence_decision
        if decision:
            state_colors = {
                "agreement":   ("green",  "✓ AGREEMENT — sources agree, debate skipped"),
                "disagreement":("yellow", "⚡ DISAGREEMENT — conflict detected, debate triggered"),
                "uncertain":   ("blue",   "~ UNCERTAIN — sources unrelated, debate skipped"),
                "insufficient":("red",    "✗ INSUFFICIENT — not enough evidence retrieved"),
            }
            state_val = decision.state.value
            color, label = state_colors.get(state_val, ("white", state_val.upper()))
            console.print(f"  [{color}]{label}[/{color}]")

            # Show pairwise NLI scores
            if decision.pairwise:
                console.print()
                nli_table = Table(box=box.SIMPLE, show_header=True)
                nli_table.add_column("Agent A",        style="cyan",  width=22)
                nli_table.add_column("Agent B",        style="cyan",  width=22)
                nli_table.add_column("Contradiction ↑", style="red",  width=16, justify="right")
                nli_table.add_column("Entailment ↑",   style="green", width=14, justify="right")
                nli_table.add_column("Verdict",        style="bold",  width=14)
                for p in decision.pairwise:
                    a_name = agent_provider.get(p.agent_a_id, p.agent_a_id)
                    b_name = agent_provider.get(p.agent_b_id, p.agent_b_id)
                    verdict = (
                        "[red]CONFLICT[/]" if p.contradiction_score >= 0.70
                        else "[green]AGREE[/]" if p.entailment_score >= 0.70
                        else "[blue]NEUTRAL[/]"
                    )
                    nli_table.add_row(
                        a_name, b_name,
                        f"{p.contradiction_score:.2f}",
                        f"{p.entailment_score:.2f}",
                        verdict,
                    )
                console.print(nli_table)

            if decision.failed_agents:
                failed = ", ".join(f"{f['agent_id']} ({f['reason']})" for f in decision.failed_agents)
                console.print(f"  [dim]Failed: {failed}[/]")

        # ════════════════════════════════════════════════════════════
        # STEP 3 — Debate (only if triggered)
        # ════════════════════════════════════════════════════════════
        if trace.debate_triggered:
            console.print()
            console.print("[bold cyan]━━━  STEP 3: Conflict Classification & Debate  ━━━[/]")

            if trace.conflict:
                console.print(f"  [yellow]Conflict Type: [bold]{trace.conflict.type.value.upper()}[/bold][/]")
                console.print(f"  [dim]Explanation: {trace.conflict.explanation}[/]")

                # Show the two conflicting claims side by side
                conflict_table = Table(box=box.ROUNDED, show_lines=True, expand=True)
                conflict_table.add_column("Claim A", style="red",   ratio=1)
                conflict_table.add_column("Claim B", style="yellow", ratio=1)
                conflict_table.add_row(
                    trace.conflict.claim_a.text,
                    trace.conflict.claim_b.text,
                )
                console.print(conflict_table)

            if trace.debate_outcome:
                outcome = trace.debate_outcome
                status_color = "green" if outcome.status == "completed" else "red"
                console.print(
                    f"\n  Debate: [{status_color}]{outcome.status.upper()}[/{status_color}] | "
                    f"{outcome.rounds_used} round(s) | "
                    f"stop reason: {outcome.stop_reason}"
                )

                # Show each debate round
                for rnd in outcome.rounds:
                    console.print(f"\n  [bold]Round {rnd.round_number}:[/]")
                    if rnd.proposer_argument:
                        console.print(f"    [cyan]Proposer →[/] {rnd.proposer_argument.content[:120]}…")
                    for i, c in enumerate(rnd.critic_arguments, 1):
                        console.print(f"    [red]Critic {i}  →[/] {c.content[:100]}…")
                    if rnd.reviser_argument:
                        console.print(f"    [green]Reviser  →[/] {rnd.reviser_argument.content[:120]}…")

            if trace.judge_result:
                jr = trace.judge_result
                console.print(
                    f"\n  [bold]Judge verdict:[/] winner={jr.winner} | "
                    f"confidence={jr.confidence:.0%} | "
                    f"evidence_score={jr.evidence_score:.2f}"
                )
                console.print(f"  [dim]Reason: {jr.reason}[/]")

        # ════════════════════════════════════════════════════════════
        # STEP 4 — Final answer
        # ════════════════════════════════════════════════════════════
        console.print()
        console.print("[bold cyan]━━━  STEP 4: Final Answer  ━━━[/]")

        m = trace.metrics or {}
        console.print(
            f"[dim]Latency: {m.get('latency_seconds', 0):.2f}s | "
            f"Usable agents: {m.get('usable_agents', 0)} | "
            f"LLM calls: {m.get('total_llm_calls_ok', 0)} ok / "
            f"{m.get('total_llm_calls_failed', 0)} failed[/]"
        )
        console.print()

        if trace.final_answer:
            console.print(Panel(
                trace.final_answer.answer,
                title="[bold green]Final Answer[/]",
                border_style="green",
                padding=(1, 2),
            ))
            if trace.final_answer.sources:
                console.print(f"\n[bold]Sources ({trace.final_answer.source_count}):[/]")
                for i, src in enumerate(trace.final_answer.sources[:5], 1):
                    console.print(f"  {i}. {src.title or src.url}")
                    if src.url:
                        console.print(f"     [link={src.url}][dim]{src.url[:80]}[/dim][/link]")
        else:
            console.print("[dim]No answer generated.[/]")


def main():
    """Main entry point."""
    import sys

    if "--api" in sys.argv:
        import uvicorn
        from api import app
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        run_interactive()


if __name__ == "__main__":
    main()