"""
nexus/tools/llm_client.py
--------------------------
Ollama-based LLM client — the ONLY way the Nexus kernel talks to an LLM.
No OpenAI. No Anthropic API. No paid services.

Uses the `ollama` Python SDK which talks to your local Ollama daemon.
Gemma3 (or whichever model you have pulled) is used throughout.

Key features:
  - Async-first (asyncio compatible)
  - Structured output via Pydantic model parsing
  - Retry logic with exponential backoff
  - OpenTelemetry span wrapping for every LLM call
  - Streaming support for long generations
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, TypeVar

import ollama
import structlog
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from nexus.config.settings import get_settings

log = structlog.get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Message primitives — mirrors the standard chat message format
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None


# ---------------------------------------------------------------------------
# Core client
# ---------------------------------------------------------------------------


class OllamaClient:
    """
    Wraps the Ollama SDK with:
    - Automatic retry on transient failures
    - Structured JSON output parsing
    - Tracing hooks (OTel spans added in observability layer)

    Typical usage:
        client = OllamaClient()
        response = await client.chat([
            Message(role="system", content="You are a systems analyst."),
            Message(role="user", content="List all Kubernetes services."),
        ])
        print(response.content)
    """

    def __init__(self, model: str | None = None, fast: bool = False) -> None:
        cfg = get_settings().ollama
        self.model = model or (cfg.fast_model if fast else cfg.model)
        self.base_url = cfg.base_url
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.num_ctx = cfg.num_ctx
        self._client = ollama.AsyncClient(host=self.base_url)
        log.info("OllamaClient initialised", model=self.model, base_url=self.base_url)

    # ------------------------------------------------------------------
    # Primary async chat interface
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((ollama.ResponseError, ConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """
        Send a list of messages and return a single LLMResponse.

        Args:
            messages: Conversation history as Message objects.
            temperature: Override instance temperature if needed.
            max_tokens: Optional token cap on the response.
            system: Prepend a system prompt (shorthand — gets inserted at index 0).
        """
        payload = self._build_payload(messages, system)

        options: dict[str, Any] = {
            "temperature": temperature or self.temperature,
            "top_p": self.top_p,
            "num_ctx": self.num_ctx,
        }
        if max_tokens:
            options["num_predict"] = max_tokens

        t0 = time.monotonic()
        log.debug("LLM chat request", model=self.model, num_messages=len(payload))

        response = await self._client.chat(
            model=self.model,
            messages=payload,
            options=options,
        )

        latency_ms = (time.monotonic() - t0) * 1000
        content = response.message.content or ""

        log.debug(
            "LLM chat response",
            model=self.model,
            latency_ms=round(latency_ms, 1),
            chars=len(content),
        )

        return LLMResponse(
            content=content,
            model=self.model,
            latency_ms=round(latency_ms, 1),
        )

    # ------------------------------------------------------------------
    # Structured output — parse response directly into a Pydantic model
    # ------------------------------------------------------------------

    async def chat_structured(
        self,
        messages: list[Message],
        output_schema: type[T],
        system: str | None = None,
        max_retries: int = 3,
    ) -> T:
        """
        Chat, then parse the response as a JSON object matching `output_schema`.

        The model is instructed (via system prompt injection) to respond ONLY
        with valid JSON conforming to the schema. Retries on parse failure.

        Example:
            class TaskPlan(BaseModel):
                steps: list[str]
                priority: int

            plan = await client.chat_structured(
                messages=[Message(role="user", content="Plan a database migration.")],
                output_schema=TaskPlan,
            )
            print(plan.steps)
        """
        schema_str = json.dumps(output_schema.model_json_schema(), indent=2)
        json_instruction = (
            f"You MUST respond with ONLY valid JSON that conforms exactly to this schema. "
            f"No markdown, no explanation, no preamble — raw JSON only.\n\nSchema:\n{schema_str}"
        )

        # Prepend JSON instruction to the system prompt
        combined_system = f"{system}\n\n{json_instruction}" if system else json_instruction

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = await self.chat(messages, system=combined_system)
                # Strip markdown code fences if model wraps output
                raw = response.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                return output_schema.model_validate_json(raw.strip())
            except Exception as exc:
                last_error = exc
                log.warning(
                    "Structured output parse failed",
                    attempt=attempt + 1,
                    error=str(exc),
                )
                # Feedback the error so the model can self-correct
                messages = messages + [
                    Message(
                        role="assistant",
                        content=response.content if "response" in dir() else "",
                    ),
                    Message(
                        role="user",
                        content=f"Your response was not valid JSON. Error: {exc}. "
                        f"Please fix it and return ONLY the JSON object.",
                    ),
                ]

        raise ValueError(
            f"Failed to get structured output after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Streaming — useful for long code generation in MCP synthesis
    # ------------------------------------------------------------------

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream tokens as they are generated. Useful for watching MCP server
        code being written in real-time.

        Usage:
            async for token in client.stream(messages):
                print(token, end="", flush=True)
        """
        payload = self._build_payload(messages, system)
        async for chunk in await self._client.chat(
            model=self.model,
            messages=payload,
            stream=True,
            options={"temperature": self.temperature, "num_ctx": self.num_ctx},
        ):
            token = chunk.message.content or ""
            if token:
                yield token

    # ------------------------------------------------------------------
    # Health check — verifies Ollama is running and model is available
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        """
        Returns a health dict:
            {"status": "ok", "model": "gemma3", "available_models": [...]}
        or raises on failure.
        """
        try:
            models_response = await self._client.list()
            available = [m.model for m in models_response.models]
            model_ready = any(self.model in m for m in available)
            return {
                "status": "ok" if model_ready else "model_missing",
                "model": self.model,
                "model_ready": model_ready,
                "available_models": available,
                "base_url": self.base_url,
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
                "base_url": self.base_url,
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self, messages: list[Message], system: str | None
    ) -> list[dict[str, str]]:
        """Convert Message objects to the dict format Ollama SDK expects."""
        payload: list[dict[str, str]] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend({"role": m.role, "content": m.content} for m in messages)
        return payload


# ---------------------------------------------------------------------------
# Convenience singleton — use this in agents instead of creating new clients
# ---------------------------------------------------------------------------

_default_client: OllamaClient | None = None


def get_llm_client(fast: bool = False) -> OllamaClient:
    """
    Returns a module-level singleton OllamaClient.
    Pass fast=True to use the smaller/faster model for MCTS rollouts.
    """
    global _default_client
    if _default_client is None:
        _default_client = OllamaClient(fast=fast)
    return _default_client
