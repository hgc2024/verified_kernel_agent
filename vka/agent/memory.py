"""Per-task attempt history, stored as JSONL.

The agent sees a compact summary of prior attempts rather than raw history, which keeps prompts
bounded as attempts accumulate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Attempt:
    task_id: str
    index: int
    plan: str
    code_sha256: str
    harness_result: dict[str, Any] = field(default_factory=dict)
    reviewer_notes: str = ""


class AttemptMemory:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, attempt: Attempt) -> None:
        # TODO(phase-5): append one JSON line.
        raise NotImplementedError("TODO(phase-5): AttemptMemory.append")

    def load(self) -> list[Attempt]:
        # TODO(phase-5): read all attempts.
        raise NotImplementedError("TODO(phase-5): AttemptMemory.load")

    def summary(self, max_attempts: int = 5) -> str:
        # TODO(phase-5): compact summary for prompts.
        raise NotImplementedError("TODO(phase-5): AttemptMemory.summary")
