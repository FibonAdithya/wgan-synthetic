# The executable definition of "a valid change". AGENTS.md points at `make
# check` as the gate, so an agent never has to guess what running the checks
# means. Nothing here uses `|| true`: a target that cannot fail is not a gate,
# and a suite allowed to go red is much harder to re-tighten later than to keep
# green from the start.

PYTHON ?= python
RUFF ?= ruff

.PHONY: check lint format-check format test

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
