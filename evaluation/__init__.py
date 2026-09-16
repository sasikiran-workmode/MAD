"""
Evaluation package for the Multi-Agent Debate framework.
"""
from .dataset import EvaluationDataset, get_evaluation_dataset
from .metrics import MetricsCalculator, get_metrics_calculator
from .run_experiments import ExperimentRunner, run_all_experiments

__all__ = [
    "EvaluationDataset",
    "get_evaluation_dataset",
    "MetricsCalculator",
    "get_metrics_calculator",
    "ExperimentRunner",
    "run_all_experiments",
]