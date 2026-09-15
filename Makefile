# The executable definition of "a valid change". AGENTS.md points at `make
# check` as the gate, so an agent never has to guess what running the checks
# means. Nothing here uses `|| true`: a target that cannot fail is not a gate,
# and a suite allowed to go red is much harder to re-tighten later than to keep
# green from the start.

PYTHON ?= python
RUFF ?= ruff
VENV ?= .venv

.PHONY: check lint format-check format test setup

check: lint format-check test

lint:
	$(RUFF) check src tests

format-check:
	$(RUFF) format --check src tests

# Not part of `check` -- this one rewrites files.
format:
	$(RUFF) format src tests

test:
	$(PYTHON) -m pytest

# One idempotent entry point for a fresh checkout; the four steps of the
# README's quick start. The venv rule fires only when $(VENV) is absent, and
# pip skips what is already installed. `make -n setup` shows what would run.
setup: $(VENV)/bin/python
	$(VENV)/bin/python -m pip install --quiet -r requirements.txt
	$(VENV)/bin/python -m pip install --quiet -r requirements-dev.txt

$(VENV)/bin/python:
	python3 -m venv $(VENV)
