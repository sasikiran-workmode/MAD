"""Vercel entrypoint for the Evidence-MAD FastAPI application."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import app  # noqa: E402,F401
