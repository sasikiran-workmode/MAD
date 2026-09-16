"""
RunMode enum and fixture loader for FIXTURE/LIVE mode separation.
Task 22: One RunMode passed explicitly into every constructor.
         FIXTURE mode uses data/fixtures.json — raises on unknown query.
         No component reads settings.demo_mode on its own.
"""
import json
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger


class RunMode(str, Enum):
    FIXTURE = "fixture"
    LIVE = "live"


FIXTURES_PATH = Path(__file__).parent / "data" / "fixtures.json"

# Required fixture case IDs per task-plan.md Task 22
REQUIRED_FIXTURE_IDS = {
    "agree_capital", "factual_boiling", "temporal_policy",
    "version_python", "contextual_drug", "source_blog_vs_paper",
    "unrelated_evidence", "one_agent_fails", "two_agents_fail", "all_agents_fail",
}


class FixtureStore:
    """
    Loads and serves fixture cases from data/fixtures.json.
    Raises on unknown queries in FIXTURE mode.
    """

    def __init__(self, path: Path = FIXTURES_PATH):
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._by_query: Dict[str, Dict[str, Any]] = {}
        self._load(path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Fixture file not found: {path}. "
                "Run `python -c 'from run_mode import FixtureStore; FixtureStore()'` "
                "after creating data/fixtures.json."
            )
        with open(path) as f:
            data = json.load(f)
        for case in data.get("cases", []):
            self._by_id[case["id"]] = case
            self._by_query[case["query"].strip().lower()] = case
        logger.info(f"Loaded {len(self._by_id)} fixture cases")

    def get_by_id(self, case_id: str) -> Dict[str, Any]:
        if case_id not in self._by_id:
            raise KeyError(f"Unknown fixture case ID: {case_id!r}")
        return self._by_id[case_id]

    def get_by_query(self, query: str) -> Dict[str, Any]:
        """
        Return the fixture case for the given query.
        Raises FixtureNotFoundError on unknown query — no fallback.
        """
        key = query.strip().lower()
        if key not in self._by_query:
            available = list(self._by_id.keys())
            raise FixtureNotFoundError(
                f"FIXTURE mode: unknown query {query!r}.\n"
                f"Available cases: {available}\n"
                "Add the query to data/fixtures.json or switch to LIVE mode."
            )
        return self._by_query[key]

    def list_ids(self) -> List[str]:
        return list(self._by_id.keys())

    def validate_required(self) -> None:
        """Raise if any required fixture case is missing."""
        missing = REQUIRED_FIXTURE_IDS - set(self._by_id.keys())
        if missing:
            raise ValueError(
                f"Missing required fixture cases: {sorted(missing)}. "
                "Add them to data/fixtures.json."
            )


class FixtureNotFoundError(KeyError):
    """Raised when a query has no fixture case in FIXTURE mode."""


# Module-level singleton (lazy-loaded)
_fixture_store: Optional[FixtureStore] = None


def get_fixture_store() -> FixtureStore:
    global _fixture_store
    if _fixture_store is None:
        _fixture_store = FixtureStore()
    return _fixture_store
