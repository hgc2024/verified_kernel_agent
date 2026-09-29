"""A single OpenAI-compatible chat client for vLLM, Ollama, and hosted APIs.

See README: Agent > Providers.

Configuration comes from ``VKA_BASE_URL``, ``VKA_API_KEY``, and ``VKA_MODEL``, read from the
process environment and, if present, a local ``.env`` file (which never overrides variables that
are already set).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from dotenv import load_dotenv

ENV_BASE_URL = "VKA_BASE_URL"
ENV_API_KEY = "VKA_API_KEY"
ENV_MODEL = "VKA_MODEL"


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    model: str
    # Local servers (vLLM, Ollama) usually accept any key; hosted APIs require one.
    api_key: str = field(default="", repr=False)

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None, *, use_dotenv: bool = True
    ) -> ProviderConfig:
        """Build a config from ``env`` (default: ``os.environ``, after loading ``.env``)."""
        if env is None:
            if use_dotenv:
                load_dotenv()
            env = os.environ
        missing = [name for name in (ENV_BASE_URL, ENV_MODEL) if not env.get(name)]
        if missing:
            raise ValueError(f"missing required environment variables: {', '.join(missing)}")
        return cls(
            base_url=env[ENV_BASE_URL].rstrip("/"),
            model=env[ENV_MODEL],
            api_key=env.get(ENV_API_KEY, ""),
        )


@dataclass(frozen=True)
class ChatResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    wall_time_s: float


class ChatClient:
    """Thin client over ``POST {base_url}/chat/completions``; logs tokens and wall time per call."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def complete(self, messages: list[dict[str, str]], **params: object) -> ChatResponse:
        # TODO(phase-5): send the request, record usage and wall time.
        raise NotImplementedError("TODO(phase-5): ChatClient.complete")
