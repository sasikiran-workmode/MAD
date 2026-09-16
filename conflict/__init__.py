"""
Conflict package for classifying conflict types.
"""
from .classifier import ConflictClassifier, get_conflict_classifier

__all__ = [
    "ConflictClassifier",
    "get_conflict_classifier",
]