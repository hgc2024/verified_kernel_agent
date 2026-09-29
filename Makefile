# Convenience targets (README: Make targets). Intended for Linux / WSL2.

PYTHON    ?= python3
VENV      ?= .venv
PY        := $(VENV)/bin/python
BUILD_DIR ?= build
CMAKE_ARGS ?=
HARNESS   := --manifest-path harness/Cargo.toml
TASK      ?= level1.gemm_fp32

CPP_SOURCES = $(shell find kernels runner tests/cpp -name '*.cu' -o -name '*.cpp' -o -name '*.h' 2>/dev/null)

.PHONY: setup configure build test test-gpu bench agent tables lint fmt clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	cargo build $(HARNESS)
	cmake -S . -B $(BUILD_DIR) $(CMAKE_ARGS)

configure:
	cmake -S . -B $(BUILD_DIR) $(CMAKE_ARGS)

build: configure
	cmake --build $(BUILD_DIR) -j
	cargo build --release $(HARNESS)

test:
	cargo test $(HARNESS)
	$(PY) -m pytest -m "not gpu"

test-gpu:
	$(PY) -m pytest -m gpu

bench:
	$(PY) -m vka.eval.run_benchmark --agent baseline --tasks tasks/level1 --attempts 1

agent:
	$(PY) -m vka.agent.loop --task $(TASK)

tables:
	$(PY) -m vka.eval.make_tables

lint:
	cargo fmt $(HARNESS) --check
	cargo clippy $(HARNESS) --all-targets -- -D warnings
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	clang-format --dry-run --Werror $(CPP_SOURCES)

fmt:
	cargo fmt $(HARNESS)
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .
	clang-format -i $(CPP_SOURCES)

clean:
	rm -rf $(BUILD_DIR)
	cargo clean $(HARNESS)
