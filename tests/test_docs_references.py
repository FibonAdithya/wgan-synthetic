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

import ast
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


def test_authoritative_docs_has_the_expected_count():
    """Guard against the docs/datasets/*.md glob silently yielding nothing.

    If docs/datasets were renamed, the glob above would match zero files,
    six dataset pages would drop out of every check in this module, and the
    suite would stay green throughout. Pin the count -- 5 named root docs
    plus 6 dataset pages -- so that failure mode is loud instead of silent.
    """
    assert len(AUTHORITATIVE_DOCS) == 11


# A backticked reference, optionally suffixed by an anchor, a symbol, or a
# line number. The path must start with an alphanumeric, underscore, or dot
# -- the last so dot-rooted paths like `.github/workflows/ci.yml` are
# covered -- but never a slash, which is what excludes absolute paths on
# other machines. Doc prose naming a run config under /workspace on the GPU
# box is a true statement about tig-gpu, not a path this repo can resolve.
REFERENCE = re.compile(
    r"`(?P<path>[A-Za-z0-9_.][A-Za-z0-9_./-]*\.(?:md|py|ya?ml|json|npy|txt|toml))"
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


def slug(heading: str) -> str:
    """Slug a markdown heading the way GitHub does when it builds an anchor.

    Lowercase, drop everything that is not a letter, digit, space, hyphen or
    underscore, then spaces to hyphens. Dropped punctuation leaves its spaces
    behind, which is why `## ANN difficulty -- the gate` (with an em dash)
    slugs to `ann-difficulty--the-gate`, with two hyphens.
    """
    text = heading.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return text.replace(" ", "-")


HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def headings(path) -> set[str]:
    """Every heading in a markdown file, slugged, ignoring fenced code."""
    found = set()
    fenced = False
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        match = HEADING.match(line)
        if match:
            found.add(slug(match.group(2)))
    return found


def test_slug_matches_the_github_algorithm():
    assert slug("Documentation map") == "documentation-map"
    assert slug("Model architecture") == "model-architecture"
    assert slug("Metric definitions") == "metric-definitions"
    assert slug("`generator_type`") == "generator_type"
    assert slug("Model variants: the per-dataset ladder") == (
        "model-variants-the-per-dataset-ladder"
    )
    # The awkward one: the em dash is dropped and leaves its two spaces, so
    # the slug carries a double hyphen. Getting this wrong would give a test
    # that passes while the rendered link 404s.
    assert slug("ANN difficulty — the gate") == "ann-difficulty--the-gate"


@pytest.mark.parametrize("doc", AUTHORITATIVE_DOCS, ids=rel)
def test_anchor_references_resolve(doc):
    broken = []
    for lineno, match in iter_refs(doc):
        anchor = match.group("anchor")
        if anchor is None:
            continue
        target = REPO_ROOT / match.group("path")
        if not target.exists():
            broken.append(f"{rel(doc)}:{lineno} -> missing file {match.group('path')}")
        elif anchor not in headings(target):
            broken.append(f"{rel(doc)}:{lineno} -> {match.group('path')}#{anchor}")
    assert not broken, "anchors that match no heading: " + "; ".join(broken)


def module_symbols(path) -> set[str]:
    """Top-level function, class and assignment names in a Python module.

    Parsed rather than imported: importing src.eval modules pulls in torch and
    runs module-level code, which is far too much machinery for a docs test.
    """
    tree = ast.parse(path.read_text())
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


def test_module_symbols_finds_functions_classes_and_annotated_constants():
    """Pin module_symbols directly, the way test_slug_matches_the_github_algorithm pins slug.

    module_symbols has no live subjects in the docs today: no `::symbol`
    reference in an authoritative doc exercises it, so
    test_symbol_references_resolve stays green regardless of what this
    function does. A regression here -- say, losing the AnnAssign branch
    again -- would go unnoticed until the day a real citation happens to
    hit it. This test is the only thing pinning the resolver's behaviour
    until then.
    """
    ann_difficulty = module_symbols(REPO_ROOT / "src/eval/ann_difficulty.py")
    assert "lid_mle" in ann_difficulty
    assert "compute" in ann_difficulty
    assert "AnnMetrics" in ann_difficulty
    assert "not_a_real_symbol" not in ann_difficulty

    compare_variants = module_symbols(REPO_ROOT / "src/eval/compare_variants.py")
    assert "VARIANTS" in compare_variants


@pytest.mark.parametrize("doc", AUTHORITATIVE_DOCS, ids=rel)
def test_symbol_references_resolve(doc):
    broken = []
    for lineno, match in iter_refs(doc):
        symbol = match.group("symbol")
        if symbol is None:
            continue
        path = match.group("path")
        target = REPO_ROOT / path
        if not target.exists():
            broken.append(f"{rel(doc)}:{lineno} -> missing file {path}")
        elif symbol not in module_symbols(target):
            broken.append(f"{rel(doc)}:{lineno} -> {path}::{symbol}")
    assert not broken, "symbols that do not exist: " + "; ".join(broken)


@pytest.mark.parametrize("doc", AUTHORITATIVE_DOCS, ids=rel)
def test_no_line_number_citations(doc):
    """Line numbers rot silently; anchors and symbols do not.

    This is the check that keeps the other two meaningful. Without it a new
    `PROJECT_DOCUMENTATION.md:274` can be added at any time and nothing
    notices until it is pointing at a hyperparameter again.
    """
    offenders = [
        f"{rel(doc)}:{lineno} -> {match.group(0)}"
        for lineno, match in iter_refs(doc)
        if match.group("line") is not None
    ]
    assert not offenders, (
        "cite documents by anchor (`FILE.md#heading-slug`) and code by symbol "
        "(`module.py::name`) instead of by line number: " + "; ".join(offenders)
    )
