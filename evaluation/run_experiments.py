"""
Experiment runner for evaluating the Multi-Agent Debate system.

Tasks applied:
  Task 25: Uses SingleLLMSystem / StandardMADSystem / ProposedMADSystem —
           no async, no post-hoc label editing, no mock conflict fabrication.
  Task 26: All 5-6 systems (+ ablations) run from one command.
  Task 28: generate_graphs uses only measured trace data —
           zero hardcoded bar values anywhere.
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from evidence.models import ExecutionTrace
from evaluation.dataset import EvaluationDataset, EvaluationSample, get_evaluation_dataset
from evaluation.metrics import MetricsCalculator, get_metrics_calculator
from evaluation.systems.single_llm import SingleLLMSystem
from evaluation.systems.standard_mad import StandardMADSystem
from evaluation.systems.proposed_mad import (
    ProposedMADSystem,
    NoGateSystem,
    NoTypedStrategySystem,
    NoRelevanceFilterSystem,
)
from app import save_trace
from run_mode import RunMode
from config import settings


# ------------------------------------------------------------------ #
# Per-system runner helper                                             #
# ------------------------------------------------------------------ #

def _run_system(
    system,
    samples: List[EvaluationSample],
    results_dir: Path,
) -> List[ExecutionTrace]:
    """
    Run a single system over all samples and save per-sample traces.

    Args:
        system:      Any object with .run(query) -> ExecutionTrace and .SYSTEM_NAME
        samples:     EvaluationSample list
        results_dir: Root results directory; traces go into results_dir/system_name/

    Returns:
        List of ExecutionTrace (one per sample, including errors)
    """
    system_name = system.SYSTEM_NAME
    system_dir = results_dir / system_name
    system_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running system: {system_name} ({len(samples)} samples)")
    traces: List[ExecutionTrace] = []

    for i, sample in enumerate(samples, 1):
        logger.debug(f"  [{i}/{len(samples)}] {system_name}: {sample.query[:60]!r}")
        try:
            trace = system.run(sample.query)
        except Exception as exc:
            logger.error(f"  {system_name} raised for sample {sample.id!r}: {exc}")
            trace = ExecutionTrace(query=sample.query, error=str(exc))

        # Attach evaluation metadata so MetricsCalculator can score it
        if trace.router_result is None:
            trace.router_result = {}
        trace.router_result.update({
            "expected_conflict": sample.expected_conflict,
            "expected_conflict_type": sample.expected_conflict_type,
            "expected_answer": sample.expected_answer or "",
            "required_facts": sample.required_facts,
            "forbidden_facts": sample.forbidden_facts,
        })
        if "system" not in trace.metrics:
            trace.metrics["system"] = system_name

        save_trace(trace, str(system_dir))
        traces.append(trace)

    logger.info(f"  {system_name}: {len(traces)} traces written to {system_dir}")
    return traces


# ------------------------------------------------------------------ #
# Main experiment runner                                               #
# ------------------------------------------------------------------ #

class ExperimentRunner:
    """
    Runs all systems over the evaluation dataset.
    Pure synchronous — no asyncio.run anywhere.
    """

    def __init__(self, run_mode: RunMode = RunMode.LIVE):
        self.run_mode = run_mode
        self.dataset = get_evaluation_dataset()
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

    def run_all_experiments(
        self,
        samples: Optional[List[EvaluationSample]] = None,
    ) -> Dict[str, Any]:
        """
        Run all 6 systems over the dataset and save results.

        Returns:
            Dictionary with per-system metrics.
        """
        samples = samples or self.dataset.samples
        logger.info(
            f"Starting full experiment suite — {len(samples)} samples, "
            f"mode={self.run_mode.value}"
        )

        # ── Instantiate all systems ───────────────────────────────────
        systems = {
            "single_llm":          SingleLLMSystem(run_mode=self.run_mode),
            "standard_mad":        StandardMADSystem(run_mode=self.run_mode),
            "proposed_mad":        ProposedMADSystem(run_mode=self.run_mode),
            "no_gate":             NoGateSystem(run_mode=self.run_mode),
            "no_typed_strategy":   NoTypedStrategySystem(run_mode=self.run_mode),
            "no_relevance_filter": NoRelevanceFilterSystem(run_mode=self.run_mode),
        }

        # ── Run each system ───────────────────────────────────────────
        all_traces: Dict[str, List[ExecutionTrace]] = {}
        for name, system in systems.items():
            all_traces[name] = _run_system(system, samples, self.results_dir)

        # ── Calculate metrics per system ──────────────────────────────
        all_metrics: Dict[str, Any] = {}
        for name, traces in all_traces.items():
            calc = get_metrics_calculator()
            calc.add_results(traces)
            m = calc.calculate_metrics()
            all_metrics[name] = m

            # Save per-system metrics and CSV
            system_dir = self.results_dir / name
            calc.save_metrics(str(system_dir / "metrics.json"))
            calc.save_results_csv(str(system_dir / "results.csv"))

        all_metrics["dataset_statistics"] = self.dataset.get_statistics()

        # ── Save combined metrics ─────────────────────────────────────
        metrics_path = self.results_dir / "experiment_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        logger.info(f"Combined metrics saved to {metrics_path}")

        # ── Print comparison table ────────────────────────────────────
        calc_single = get_metrics_calculator()
        calc_single.add_results(all_traces["single_llm"])
        calc_std = get_metrics_calculator()
        calc_std.add_results(all_traces["standard_mad"])
        calc_prop = get_metrics_calculator()
        calc_prop.add_results(all_traces["proposed_mad"])

        table = calc_prop.print_comparison_table(
            calc_single.calculate_metrics(),
            calc_std.calculate_metrics(),
            calc_prop.calculate_metrics(),
        )
        print(table)

        # ── Generate figures ──────────────────────────────────────────
        self.generate_graphs(all_metrics)

        logger.info("All experiments completed.")
        return all_metrics

    # ------------------------------------------------------------------ #
    # Task 28 — honest graphs: zero hardcoded values                      #
    # ------------------------------------------------------------------ #

    def generate_graphs(self, metrics: Dict[str, Any]) -> None:
        """
        Generate evaluation graphs.

        Every bar and data point comes from measured trace data.
        No hardcoded literals (0, 100, 1, 5, 10 ...) anywhere.
        If a metric is missing the bar is simply absent.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")          # headless — no display needed
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as e:
            logger.warning(f"Cannot generate graphs (missing dependency): {e}")
            return

        figures_dir = self.results_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        # Canonical system display order
        system_display = {
            "single_llm":          "Single LLM",
            "standard_mad":        "Standard MAD",
            "proposed_mad":        "Proposed MAD",
            "no_gate":             "No Gate",
            "no_typed_strategy":   "No Typed Strategy",
            "no_relevance_filter": "No Relevance Filter",
        }
        # Only include systems that have measured data
        present = [k for k in system_display if k in metrics and isinstance(metrics[k], dict)]

        colors = plt.cm.tab10(np.linspace(0, 0.8, len(present)))

        def _bar_chart(metric_key: str, ylabel: str, title: str, filename: str,
                       scale: float = 1.0, ylim_top: Optional[float] = None) -> None:
            """Generic helper: build a bar chart from measured metric values only."""
            names, values = [], []
            for k in present:
                v = metrics[k].get(metric_key)
                if v is not None:
                    names.append(system_display[k])
                    values.append(float(v) * scale)

            if not values:
                logger.warning(f"No data for {metric_key}; skipping figure {filename}")
                return

            fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))
            bars = ax.bar(names, values, color=colors[: len(names)])
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            if ylim_top is not None:
                ax.set_ylim(0, ylim_top)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (ylim_top or max(values)) * 0.01,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=9,
                )
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(figures_dir / filename, dpi=150)
            plt.close()
            logger.info(f"Saved figure: {figures_dir / filename}")

        # ── Figure 1: Key-fact accuracy ───────────────────────────────
        _bar_chart(
            metric_key="key_fact_accuracy",
            ylabel="Key-Fact Accuracy (%)",
            title="Key-Fact Accuracy by System",
            filename="key_fact_accuracy.png",
            scale=100.0,
            ylim_top=105.0,
        )

        # ── Figure 2: Debate trigger rate (measured, never hardcoded) ─
        _bar_chart(
            metric_key="debate_trigger_rate",
            ylabel="Debate Trigger Rate (%)",
            title="Debate Trigger Rate by System",
            filename="debate_trigger_rate.png",
            scale=100.0,
            ylim_top=110.0,
        )

        # ── Figure 3: Average LLM calls (ok only) ────────────────────
        _bar_chart(
            metric_key="avg_llm_calls_ok",
            ylabel="Avg. LLM Calls (OK)",
            title="Average Successful LLM Calls per Query",
            filename="llm_calls_ok.png",
        )

        # ── Figure 4: Latency p50 ─────────────────────────────────────
        _bar_chart(
            metric_key="latency_p50",
            ylabel="Latency p50 (seconds)",
            title="Median Latency per Query (p50)",
            filename="latency_p50.png",
        )

        # ── Figure 5: Conflict type distribution (proposed_mad only) ──
        conflict_dist = metrics.get("proposed_mad", {}).get("conflict_type_distribution", {})
        if conflict_dist:
            try:
                types = list(conflict_dist.keys())
                counts = list(conflict_dist.values())
                colors_pie = plt.cm.Set3(np.linspace(0, 1, len(types)))
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.pie(
                    counts, labels=types, colors=colors_pie,
                    autopct="%1.1f%%", startangle=90,
                )
                ax.set_title("Conflict Type Distribution (Proposed MAD)")
                plt.tight_layout()
                plt.savefig(figures_dir / "conflict_distribution.png", dpi=150)
                plt.close()
                logger.info(f"Saved figure: {figures_dir / 'conflict_distribution.png'}")
            except Exception as e:
                logger.warning(f"Conflict distribution chart failed: {e}")

        # ── Figure 6: Error / retrieval failure rate ──────────────────
        _bar_chart(
            metric_key="error_rate",
            ylabel="Error Rate (%)",
            title="Error Rate by System",
            filename="error_rates.png",
            scale=100.0,
            ylim_top=105.0,
        )


# ------------------------------------------------------------------ #
# Convenience entry points                                             #
# ------------------------------------------------------------------ #

def run_all_experiments(demo_mode: bool = False) -> Dict[str, Any]:
    """Synchronous convenience wrapper (replaces old async version)."""
    run_mode = RunMode.FIXTURE if demo_mode else RunMode.LIVE
    runner = ExperimentRunner(run_mode=run_mode)
    return runner.run_all_experiments()


if __name__ == "__main__":
    import sys
    demo_mode = "--demo" in sys.argv
    run_all_experiments(demo_mode=demo_mode)
