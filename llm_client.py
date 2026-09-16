"""
LLM client wrapper for interacting with OpenAI-compatible APIs.
Fails loudly on any error, counts calls per role (ok/failed).
"""
import json
import re
import time
from collections import defaultdict
from typing import Optional, Dict, Any
from loguru import logger
import httpx

from config import settings


class LLMError(Exception):
    """Raised when any LLM call fails."""


def _extract_json_block(text: str) -> str:
    """Extract first JSON object or array from text."""
    # Try fenced code block first
    fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    # Fall back to first braced block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return text[start:end]
    raise ValueError("No JSON object found in response")


class LLMClient:
    """
    Client for OpenAI-compatible LLM APIs (Groq, OpenAI, etc.).
    - chat()      → returns str or raises LLMError
    - chat_json() → returns dict or raises LLMError
    - call_report() → per-role ok/failed breakdown
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        self.api_key = settings.llm_api_key if api_key is None else api_key
        self.base_url = (settings.llm_base_url if base_url is None else base_url).rstrip("/")
        self.model = settings.llm_model if model is None else model
        self.temperature = settings.llm_temperature if temperature is None else temperature
        self.client = httpx.Client(timeout=settings.llm_timeout)
        # Per-role call tracking
        self.calls: Dict[str, Dict[str, int]] = defaultdict(lambda: {"ok": 0, "failed": 0})

    def chat(self, prompt: str, system_prompt: Optional[str] = None, *, role: str) -> str:
        """
        Send a chat completion request.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            role: Required keyword — e.g. "classifier", "proposer", "judge"

        Returns:
            Generated text response

        Raises:
            LLMError: on any failure (missing key, network error, bad status, …)
        """
        if not self.api_key:
            self.calls[role]["failed"] += 1
            raise LLMError("No LLM API key configured")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_exc: Optional[Exception] = None
        for attempt in range(1, settings.llm_max_retries + 1):
            try:
                resp = self.client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": self.temperature,
                        "max_tokens": 1024,
                    },
                )
                # Only retry on rate-limit or server errors
                if resp.status_code == 400:
                    # Prompt bug — no point retrying
                    self.calls[role]["failed"] += 1
                    raise LLMError(f"{role}: 400 Bad Request — {resp.text[:200]}")
                resp.raise_for_status()
                self.calls[role]["ok"] += 1
                return resp.json()["choices"][0]["message"]["content"]

            except LLMError:
                raise
            except httpx.HTTPStatusError as e:
                last_exc = e
                if attempt < settings.llm_max_retries:
                    time.sleep(2 ** attempt)
            except Exception as e:
                last_exc = e
                if attempt < settings.llm_max_retries:
                    time.sleep(2 ** attempt)

        self.calls[role]["failed"] += 1
        raise LLMError(f"{role}: {last_exc}") from last_exc

    def chat_json(self, prompt: str, system_prompt: Optional[str] = None, *, role: str) -> Dict[str, Any]:
        """
        Send a chat completion request expecting JSON output.

        Returns:
            Parsed JSON dict

        Raises:
            LLMError: on any failure including unparseable JSON
        """
        raw = self.chat(prompt, system_prompt, role=role)
        try:
            return json.loads(_extract_json_block(raw))
        except Exception as e:
            self.calls[role]["failed"] += 1
            raise LLMError(f"{role} returned unparseable JSON: {e}") from e

    def call_report(self) -> Dict[str, Dict[str, int]]:
        """Return per-role ok/failed call counts."""
        return {r: dict(v) for r, v in self.calls.items()}

    def total_ok(self) -> int:
        """Total successful calls across all roles."""
        return sum(v["ok"] for v in self.calls.values())

    def total_failed(self) -> int:
        """Total failed calls across all roles."""
        return sum(v["failed"] for v in self.calls.values())

    def close(self):
        """Close the HTTP client."""
        self.client.close()


def get_llm_client() -> LLMClient:
    """Get a new LLM client instance."""
    return LLMClient()