"""The README results block exists and holds no hand-written numbers."""

from __future__ import annotations

from pathlib import Path

from vka.eval.make_tables import PLACEHOLDER, RESULTS_END, RESULTS_START

README = Path(__file__).resolve().parents[2] / "README.md"


def test_results_block_is_placeholder_until_real_runs() -> None:
    text = README.read_text(encoding="utf-8")
    assert text.count(RESULTS_START) == 1
    assert text.count(RESULTS_END) == 1
    block = text.split(RESULTS_START, 1)[1].split(RESULTS_END, 1)[0]
    # TODO(phase-6): once make_tables writes real tables, check the block matches its output.
    assert block.strip() == PLACEHOLDER
