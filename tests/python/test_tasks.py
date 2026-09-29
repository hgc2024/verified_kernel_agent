"""Task TOMLs are well-formed. Mirrors the Rust loader's validation in harness/src/task.rs."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from vka.reference import resolve

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_FILES = sorted((REPO_ROOT / "tasks").rglob("*.toml"))
REQUIRED_KEYS = {
    "id",
    "description",
    "reference",
    "baseline",
    "inputs",
    "outputs",
    "shapes",
    "tolerance",
}
DTYPES = {"f32", "f16", "bf16"}


def test_tasks_exist() -> None:
    assert TASK_FILES, "no task TOMLs found"


@pytest.mark.parametrize("path", TASK_FILES, ids=lambda p: p.stem)
def test_task_is_valid(path: Path) -> None:
    task = tomllib.loads(path.read_text(encoding="utf-8"))
    assert task.keys() >= REQUIRED_KEYS

    # The id encodes the file location so `make agent TASK=<id>` can find it.
    assert task["id"] == f"{path.parent.name}.{path.stem}"
    assert (REPO_ROOT / task["baseline"]).is_file()
    assert callable(resolve(task["reference"]))

    nominal = task["shapes"]["nominal"]
    fuzz = task["shapes"]["fuzz"]
    assert task["outputs"]
    for tensor in task["inputs"] + task["outputs"]:
        assert tensor["dtype"] in DTYPES
        for symbol in tensor["shape"]:
            assert nominal[symbol] >= 1
            lo, hi = fuzz[symbol]
            assert 1 <= lo <= hi

    assert task["tolerance"]["rtol"] >= 0
    assert task["tolerance"]["atol"] >= 0
