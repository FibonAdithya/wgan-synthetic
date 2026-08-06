"""Mechanical checks on the references in the human-maintained documentation.

The docs here are load-bearing: AGENTS.md routes an agent to whichever
document is authoritative for a question, and it does that by citing specific
places. Those citations used to be line numbers, and every insertion into
PROJECT_DOCUMENTATION.md pushed the ones below it out of true without any
signal -- seven of the nine in AGENTS.md were wrong by the time this module
was written, including both citations for the gate itself.

So documents are cited by anchor and code by symbol. Both survive edits above
them, and both fail loudly here when the thing they name is renamed or
removed.

What this does not check is semantic drift: prose that is simply untrue about
the code. That is not mechanically detectable, and it is the docs-review
workflow's job.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# The documents AGENTS.md calls human-maintained and authoritative. Everything
# under docs/superpowers/ is deliberately absent: docs/superpowers/README.md
# states those are snapshots kept for provenance and not updated as the code
# changes, so a reference that has gone stale in one of them is a fact about
# history rather than a defect.
AUTHORITATIVE_DOCS = [
    REPO_ROOT / name
    for name in (
        "AGENTS.md",
        "CLAUDE.md",
        "README.md",
        "PROJECT_DOCUMENTATION.md",
        "data/README.md",
    )
] + sorted((REPO_ROOT / "docs" / "datasets").glob("*.md"))

# A backticked reference, optionally suffixed by an anchor, a symbol, or a
# line number. The path must start with an alphanumeric or underscore, which
# is what excludes absolute paths on other machines -- doc prose naming a run
# config under /workspace on the GPU box is a true statement about tig-gpu,
# not a path this repo can resolve.
REFERENCE = re.compile(
    r"`(?P<path>[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:md|py|ya?ml|json|npy|txt|toml))"
    r"(?:#(?P<anchor>[A-Za-z0-9_-]+)"
    r"|::(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)"
    r"|:(?P<line>\d+(?:-\d+)?))?`"
)
# The absolute-path exclusion above was aimed at a FOLLOWUPS.md citation that
# has since moved into issue #17. Keep it: the constraint is right regardless,
# and doc prose naming a path on the GPU box will recur.

# Paths that are correctly absent from a clean checkout. runs/ and the .npy
# corpora are gitignored build products. The eval/ entries appear in
# PROJECT_DOCUMENTATION.md's "Run artifact structure" list, where they are
# relative to a run directory rather than to the repo root.
GENERATED = (
    re.compile(r"^runs/"),
    re.compile(r"^data/.*\.(?:npy|fvecs)$"),
    re.compile(r"^eval(?:_file_to_file|_embeddings)?/"),
)


def is_generated(path: str) -> bool:
    return any(pattern.match(path) for pattern in GENERATED)


def iter_refs(doc):
    """Yield (lineno, match) for each reference outside a fenced code block."""
    fenced = False
    for lineno, line in enumerate(doc.read_text().splitlines(), 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        for match in REFERENCE.finditer(line):
            yield lineno, match


def rel(doc) -> str:
    return str(doc.relative_to(REPO_ROOT))


@pytest.mark.parametrize("doc", AUTHORITATIVE_DOCS, ids=rel)
def test_path_references_resolve(doc):
    broken = [
        f"{rel(doc)}:{lineno} -> {match.group('path')}"
        for lineno, match in iter_refs(doc)
        if "/" in match.group("path")
        and not is_generated(match.group("path"))
        and not (REPO_ROOT / match.group("path")).exists()
    ]
    assert not broken, "references to paths that do not exist: " + "; ".join(broken)
