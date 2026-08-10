# Doc Review Bot and Doc Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a PR workflow that reviews documentation drift, and clear the accumulated citation rot in the authoritative docs behind a lint that stops it recurring.

**Architecture:** Three units sharing no code. A GitHub Actions workflow (`docs-review.yml`) that comments on PRs. A pytest module (`tests/test_docs_references.py`) that resolves every path, anchor and symbol reference in the human-maintained docs. And a one-time edit pass converting rot-prone `file:NNN` citations into anchors and `::symbol` refs, which is what makes the lint checkable.

**Tech Stack:** Python 3.12, pytest, `ast` and `re` from the standard library, ruff, GitHub Actions, `anthropics/claude-code-action@v1`.

## Global Constraints

- Run everything from the repo root on Python 3.12. This worktree has no `.venv`; use the interpreter at `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python`.
- `make check` is the gate: `ruff check src tests`, `ruff format --check src tests`, `python -m pytest`. No target may use `|| true`.
- `make format` rewrites files. Only format files you touched; never run it repo-wide.
- Ruff: `line-length = 88`, `target-version = "py312"`, rules `["E", "F", "I", "W", "UP"]`, `ignore = ["E501"]`, `known-first-party = ["src"]`.
- Never edit anything under `docs/superpowers/` other than adding this plan's own files. Those are snapshots, non-authoritative by written policy.
- Never change a number in `docs/datasets/*.md`. Those need the GPU box and corpora not present here. Flag, do not edit.
- Never touch gate bands, `data.real_path` values, or pins in `requirements.txt` — `AGENTS.md` reserves those for a human.
- Baseline before starting: 448 tests pass.

## Revision, after rebasing onto `7f8b69b`

The spec named an unmerged branch, `docs/followups-to-issues`, as a known
conflict. It merged as PR #24 while this plan was being written, so the
sweep now runs against its result:

- **`FOLLOWUPS.md` no longer exists.** Its eight entries are GitHub issues
  #15–#22 on the public mirror. It leaves `AUTHORITATIVE_DOCS`, and the four
  `l2_normalize` citations Task 4 was going to convert now live in issue #17.
- **The Issues claim is already corrected.** `AGENTS.md` now says Issues are
  the tracker on the public mirror and disabled on `upstream`. Task 6 loses
  that step.
- **`claude-review.yml` is now on `main`** (PR #13). Task 1 is unaffected —
  `docs-review.yml` is a separate file.
- **Nothing else moved.** All nine line-number citations in `AGENTS.md` are
  still present and still wrong, and every anchor target this plan names is
  still at the heading the plan assumed.

---

### Task 1: The documentation review workflow

Independent of every other task — no code, no test dependency. Can land first or last.

**Files:**
- Create: `.github/workflows/docs-review.yml`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing other tasks rely on.

- [ ] **Step 1: Create the workflow file**

```yaml
name: Docs review

on:
  pull_request:
    types: [opened, synchronize]

# A second push supersedes the review of the first; don't pay for both.
concurrency:
  group: docs-review-${{ github.ref }}
  cancel-in-progress: true

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
      id-token: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - uses: anthropics/claude-code-action@v1
        with:
          # Bills the Claude subscription rather than an API account, matching
          # claude-review.yml. The token comes from `claude setup-token`.
          claude_code_oauth_token: ${{ secrets.CLAUDE_CODE_AUTH_TOKEN }}
          # Without this the action mints its own token through the Claude
          # GitHub App, which is not installed here. Note the permissions block
          # above grants no `contents: write`: this job comments and never
          # pushes, so a confidently wrong rewrite costs a comment rather than
          # a commit.
          github_token: ${{ secrets.GITHUB_TOKEN }}
          prompt: |
            REPO: ${{ github.repository }}
            PR NUMBER: ${{ github.event.pull_request.number }}

            Review this pull request for documentation drift only. Another
            workflow reviews the code; do not duplicate it. The PR branch is
            already checked out.

            Read AGENTS.md first. It is the project's router: it names the
            order of authority between documents and lists five invariants
            that nothing in the test suite catches.

            Report, in this order:

            - A claim in an authoritative document that this diff has just
              made false. Authority runs: the code in src/ and the configs in
              configs/ win, then PROJECT_DOCUMENTATION.md, then README.md,
              data/README.md and docs/datasets/*.md.
            - Behaviour this diff changes with no matching doc update: a new
              or renamed CLI flag, a changed default, a renamed config key, a
              new entry point.
            - Drift touching the five invariants -- a change to what a variant
              number means, a diagnostic metric (MMD, cov_fro,
              pairwise_hist_l1) described as the gate instead of the four
              ANN-difficulty statistics, a cross-family comparison of variant
              numbers or measured statistics, or a checkpoint separated from
              its run_config.yaml.
            - Any newly added `file.md:123` style citation. This repo cites
              documents by anchor (`PROJECT_DOCUMENTATION.md#the-gate`) and
              code by symbol (`src/eval/ann_difficulty.py::lid_mle`), because
              line numbers silently rot when text is inserted above them.
              tests/test_docs_references.py enforces this; explain it rather
              than only naming the failure.

            Do not:
            - Rewrite documentation. Flag stale claims; the wording is a
              human's call.
            - Comment on anything under docs/superpowers/. Those are dated
              snapshots, non-authoritative by policy, and are not updated as
              the code changes.
            - Flag style that `make check` (ruff lint + format) enforces.
            - Propose tightening gate bands, repointing a config's data path,
              or bumping a pinned requirement. AGENTS.md reserves those.
            - Propose re-measuring or editing numbers in docs/datasets/*.md.

            Be concise and specific. If the documentation is consistent with
            the diff, say so in one short comment rather than manufacturing
            findings.

            Use `gh pr comment` for top-level feedback.
            Use `mcp__github_inline_comment__create_inline_comment` (with
            `confirmed: true`) for issues tied to specific lines.
            Only post GitHub comments -- don't return review text as a message.

          claude_args: |
            --allowedTools "mcp__github_inline_comment__create_inline_comment,Bash(gh pr comment:*),Bash(gh pr diff:*),Bash(gh pr view:*)"
```

- [ ] **Step 2: Verify the YAML parses**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/docs-review.yml')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Confirm the suite is unaffected**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest`
Expected: `448 passed`

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/docs-review.yml
git commit -m "ci: review pull requests for documentation drift

A sibling of claude-review.yml rather than an extension of it, so a doc
finding arrives in its own comment instead of buried under code findings,
and so the two prompts can be tuned separately.

Comment-only: the permissions block grants no contents: write, so the job
cannot push a rewrite of PROJECT_DOCUMENTATION.md onto the branch. A wrong
call costs a comment you ignore.

PRs only, no merge trigger -- every commit on main arrives through a PR, so
a post-merge pass would re-review the same diff at a second bill and report
after the cheapest moment to act has passed.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Lint scaffolding and the path check

Establishes the reference parser every later check builds on. Passes against the current tree — check 1 is a regression guard, not a fixer.

**Files:**
- Create: `tests/test_docs_references.py`

**Interfaces:**
- Consumes: nothing.
- Produces, for Tasks 3–5:
  - `REPO_ROOT: Path`
  - `AUTHORITATIVE_DOCS: list[Path]` — absolute paths to the human-maintained docs
  - `REFERENCE: re.Pattern` — named groups `path`, `anchor`, `symbol`, `line`
  - `iter_refs(doc: Path) -> Iterator[tuple[int, re.Match]]` — yields `(lineno, match)`, skipping fenced code blocks
  - `slug(heading: str) -> str`
  - `headings(path: Path) -> set[str]`
  - `module_symbols(path: Path) -> set[str]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_docs_references.py`:

```python
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
```

- [ ] **Step 2: Run it**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_docs_references.py -v`
Expected: PASS, 11 parametrized cases — 5 named docs plus 6 dataset pages. This check finds nothing today by design. If it fails, a path really is broken — read the failure before assuming the allowlist is wrong.

- [ ] **Step 3: Lint and format the new file**

Run: `ruff check tests/test_docs_references.py && ruff format tests/test_docs_references.py && ruff format --check tests/test_docs_references.py`
Expected: no errors; the format run may rewrite the file.

- [ ] **Step 4: Confirm the whole suite**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest`
Expected: `459 passed` (448 baseline + 11 parametrized cases)

- [ ] **Step 5: Commit**

```bash
git add tests/test_docs_references.py
git commit -m "test: resolve every path referenced by the authoritative docs

First of four checks. This one passes on the current tree and finds nothing:
the apparent breakages are false positives, and encoding why is the point.
They are run-directory-relative entries in PROJECT_DOCUMENTATION.md's run
artifact listing, not repo-relative paths.

So it is a regression guard rather than a fix -- it exists so the first
genuinely broken path fails make check instead of sitting there.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: The anchor check, and AGENTS.md's citations

The core of the sweep. The test is written first and fails against the un-swept `AGENTS.md`, which is the proof that the citations are wrong.

**Files:**
- Modify: `tests/test_docs_references.py`
- Modify: `AGENTS.md` (lines 19, 32, 49, 53, 63, 107, 112, 113, 115)

**Interfaces:**
- Consumes: `REPO_ROOT`, `AUTHORITATIVE_DOCS`, `iter_refs`, `rel` from Task 2.
- Produces: `slug(heading: str) -> str` and `headings(path: Path) -> set[str]`. Used only within this task; no later task consumes them.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_docs_references.py`:

```python
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
```

- [ ] **Step 2: Run to verify the slug test passes and the anchor test is vacuous**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_docs_references.py -v`
Expected: PASS. `test_anchor_references_resolve` passes trivially because no anchors exist yet — that is what Step 3 changes.

- [ ] **Step 3: Convert the nine citations in `AGENTS.md`**

Each is a `See ...` reference or a table cell. Apply these exact edits:

| Line | From | To |
|---|---|---|
| 19 | `` `README.md:10-26` `` | `` `README.md#documentation-map` `` |
| 32 | `` `docs/superpowers/README.md:6-13` `` | `` `docs/superpowers/README.md` `` |
| 49 | `` `PROJECT_DOCUMENTATION.md:274` `` | `` `PROJECT_DOCUMENTATION.md#ann-difficulty--the-gate` `` |
| 53 | `` `PROJECT_DOCUMENTATION.md:172` `` | `` `PROJECT_DOCUMENTATION.md#model-variants-the-per-dataset-ladder` `` |
| 63 | `` `PROJECT_DOCUMENTATION.md:216` `` | `` `PROJECT_DOCUMENTATION.md#generator_type` `` |
| 107 | `` `PROJECT_DOCUMENTATION.md:274` `` | `` `PROJECT_DOCUMENTATION.md#ann-difficulty--the-gate` `` |
| 112 | `` `PROJECT_DOCUMENTATION.md:146` `` | `` `PROJECT_DOCUMENTATION.md#model-architecture` `` |
| 113 | `` `PROJECT_DOCUMENTATION.md:172` `` | `` `PROJECT_DOCUMENTATION.md#model-variants-the-per-dataset-ladder` `` |
| 115 | `` `PROJECT_DOCUMENTATION.md:323` `` | `` `PROJECT_DOCUMENTATION.md#metric-definitions` `` |

The targets were re-verified after the rebase: the gate is at
`PROJECT_DOCUMENTATION.md:303`, the ladder at 201, `generator_type` at 245,
model architecture at 175, metric definitions at 365. Cite by anchor anyway —
those numbers are what rots.

Line 32 loses its range rather than gaining an anchor: it points at the "not the source of truth" paragraph, which sits under the H1 with no heading of its own, so the file alone is the honest citation.

Two of these were correct before the change (19 and 32); the other seven pointed at a hyperparameter, a blank line, a table separator, or the wrong section. Do not preserve the old numbers in a comment — they are wrong, and the anchor says what was meant.

Lines 107–115 are table rows. Keep the surrounding cell text and the `configs/<family>/` and `src/eval/` references intact; replace only the citation.

- [ ] **Step 4: Run to verify the anchors resolve**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_docs_references.py -v`
Expected: PASS. If `test_anchor_references_resolve` fails, the heading text in `PROJECT_DOCUMENTATION.md` differs from what the table above assumed — read the failure, check the real heading with `grep -nE '^#{2,3} ' PROJECT_DOCUMENTATION.md`, and fix the anchor rather than the slugger.

- [ ] **Step 5: Confirm the whole suite and lint**

Run: `ruff check tests && ruff format --check tests && /home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest`
Expected: `471 passed` (459 + 1 slug test + 11 anchor cases)

- [ ] **Step 6: Commit**

```bash
git add tests/test_docs_references.py AGENTS.md
git commit -m "docs: cite documents by anchor, and check the anchors resolve

Seven of the nine line-number citations in AGENTS.md were wrong. Invariant
1 -- ANN-difficulty is the gate, the most important claim in the project --
cited PROJECT_DOCUMENTATION.md:274, which reads \`- lambda_gp: 5.0\`. The
gate is at 303. The variant ladder cited a blank line, the checkpoint
invariant cited a table separator, generator_type cited the middle of a
different section.

They were right when written; the file grew above them and pushed every
citation below out of true, silently, because nothing looks at prose.

Correcting the numbers would fix today and rot on the next insertion, so
this changes the format instead. An anchor names what is cited rather than
where it currently sits, survives every edit that does not rename the
heading, and fails loudly here when someone does.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: The symbol check

The four `l2_normalize` citations this task was written to convert moved into
GitHub issue #17 when PR #24 landed, so there is nothing left in the docs to
convert. The check still goes in: it is the enforcement mechanism for the
`module.py::name` form that Task 1's workflow prompt tells authors to use, and
without it that instruction is unenforced. It passes vacuously today, and the
commit message says so rather than implying it caught something.

**Files:**
- Modify: `tests/test_docs_references.py`

**Interfaces:**
- Consumes: `REPO_ROOT`, `AUTHORITATIVE_DOCS`, `iter_refs`, `rel` from Task 2.
- Produces: `module_symbols(path: Path) -> set[str]`. Used only within this task; no later task consumes it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_docs_references.py`:

```python
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
    return names


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
```

- [ ] **Step 2: Verify the resolver works in both directions**

A vacuous pass is not evidence the check works — a resolver returning an empty
set would also pass. Confirm by hand before committing:

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -c "
import sys; sys.path.insert(0, 'tests')
from test_docs_references import module_symbols, REPO_ROOT
syms = module_symbols(REPO_ROOT / 'src/eval/ann_difficulty.py')
assert 'lid_mle' in syms and 'compute' in syms, 'should find functions'
assert 'AnnMetrics' in syms, 'should find classes'
assert 'no_such_symbol' not in syms
print('resolver ok:', len(syms), 'symbols')
"
```

Expected: `resolver ok: 12 symbols` (or more, if the module has grown).

- [ ] **Step 3: Run the suite**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_docs_references.py -v`
Expected: PASS. `test_symbol_references_resolve` passes vacuously — no `::`
reference exists in the docs today. Expected, not a failure to investigate.

- [ ] **Step 4: Confirm the whole suite and lint**

Run: `ruff check tests && ruff format --check tests && /home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest`
Expected: `482 passed` (471 + 11 symbol cases)

- [ ] **Step 5: Commit**

```bash
git add tests/test_docs_references.py
git commit -m "test: resolve symbol references by parsing, not importing

Enforcement for the module.py::name citation form. It passes vacuously right
now -- the four l2_normalize citations it was written for moved into issue
#17 when the FOLLOWUPS migration landed, so no doc currently uses the form.
It goes in anyway because docs-review.yml tells authors to use it, and an
unenforced instruction is one that drifts.

Parsing rather than importing: these are src/eval modules, and importing
them pulls in torch to answer a question about prose.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

<!-- Superseded by PR #24; retained for context on what the check was for.

**Original Step 3: Convert the four citations in `FOLLOWUPS.md`**

In the section "Fold the remaining `l2_normalize` copies onto `eda_report.maybe_l2_normalize`", lines 56–58 currently read:

```
carrying its own copy, but four remain: `evaluate_file_to_file.py:43`,
`plot_distance_cdf.py:30`, `plot_distance_cdf_pillow.py:46` and
`plot_embedding_clusters.py:33`. All four are byte-identical to each other
```

Replace with:

```
carrying its own copy, but four remain: `src/eval/evaluate_file_to_file.py::l2_normalize`,
`src/eval/plot_distance_cdf.py::l2_normalize`,
`src/eval/plot_distance_cdf_pillow.py::l2_normalize` and
`src/eval/plot_embedding_clusters.py::l2_normalize`. All four are
byte-identical to each other
```

Two defects fixed at once: the citations gave a bare filename with no
directory, and every line number was already off — the real definitions are at
45, 31, 47 and 34, not 43, 30, 46 and 33.

The follow-up itself is still open and must not be closed here: four copies do
still exist, and `src/eval/eda_report.py::maybe_l2_normalize` is still the one
they should fold onto.

-->

---

### Task 5: Ban line-number citations

Locks the format in. Should pass immediately after Tasks 3 and 4 — if it does not, a citation was missed.

**Files:**
- Modify: `tests/test_docs_references.py`

**Interfaces:**
- Consumes: `AUTHORITATIVE_DOCS`, `iter_refs`, `rel` from Task 2.
- Produces: nothing.

- [ ] **Step 1: Write the test**

Append to `tests/test_docs_references.py`:

```python
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
```

- [ ] **Step 2: Run it**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_docs_references.py -v`
Expected: PASS. A failure means Task 3 or 4 missed a citation — the message names the file, line and text, so fix that citation rather than weakening the test.

- [ ] **Step 3: Confirm the whole suite and lint**

Run: `make check`
Expected: green throughout, `493 passed` (482 + 11 format cases)

- [ ] **Step 4: Commit**

```bash
git add tests/test_docs_references.py
git commit -m "test: reject new line-number citations in the authoritative docs

The check that keeps the other two honest. Without it nothing stops a fresh
PROJECT_DOCUMENTATION.md:274 from being added, and nothing notices when it
starts pointing at a hyperparameter.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Delete the obsolete review pair

The Issues-claim correction this task originally carried is already done:
PR #24 rewrote `AGENTS.md` to say Issues are the tracker on the public mirror
and disabled on `upstream`, and deleted the `FOLLOWUPS.md` line that made the
same claim. Only the deletions remain.

**Files:**
- Modify: `README.md:28-34`
- Delete: `AGENTIC-REVIEW.md`
- Delete: `docs/ai-first-development-workflow.md`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing.

- [ ] **Step 1: Delete the two files**

```bash
git rm AGENTIC-REVIEW.md docs/ai-first-development-workflow.md
```

`AGENTIC-REVIEW.md`'s five surviving conclusions were: no `AGENTS.md`, no CI, a gate documented but not executable, an untested sampling path, and hard-coded run directories. Each has since been fixed — `AGENTS.md` exists, `.github/workflows/ci.yml` exists, `34c4fbf` made the gate executable, the generate-coverage PR covered sampling, and `f1d13ca` replaced the hard-coded run directories with a manifest. The file already carries its own note that its counts describe a stale base.

`docs/ai-first-development-workflow.md` is vendored from the sibling `tig-cpu` repo and says in its own header that it describes no part of this project and is kept only so `AGENTIC-REVIEW.md`'s citations resolve inside a fresh clone. With the review gone, its stated reason for existing is gone.

- [ ] **Step 2: Remove the README block that introduces them**

`README.md` lines 28–34 currently read:

```
Reviews and vendored external references, also **not** authoritative:

- `AGENTIC-REVIEW.md` — a cold-read review of how ready this repo is for
  autonomous agents. Written against one commit; its counts are a snapshot.
- `docs/ai-first-development-workflow.md` — a general AI-workflow guide copied
  from the sibling `tig-cpu` repository. It describes no part of this project
  and is kept only so the citations in `AGENTIC-REVIEW.md` resolve.
```

Delete all seven lines, and the blank line separating them from the preceding
block, so the documentation map ends with the `docs/superpowers/` entry.

- [ ] **Step 3: Verify nothing still references the deleted files**

Run: `grep -rn "AGENTIC-REVIEW\|ai-first-development-workflow" --include="*.md" --include="*.yml" --include="*.py" . | grep -v "docs/superpowers/"`
Expected: no output. Hits under `docs/superpowers/` are expected and must be left alone — those are dated snapshots that correctly record what existed when they were written.

- [ ] **Step 4: Confirm the suite**

Run: `make check`
Expected: green, `493 passed`. The doc lint's `AUTHORITATIVE_DOCS` list does not name either deleted file, so no test change is needed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs: delete the agentic review and its vendored citation target

AGENTIC-REVIEW.md's five surviving conclusions were: no AGENTS.md, no CI, a
gate documented but not executable, an untested sampling path, and hard-coded
run directories. All five are fixed -- AGENTS.md and ci.yml exist, 34c4fbf
made the gate executable, the generate-coverage PR covered sampling, f1d13ca
replaced the run directories with a manifest. What was left was 13KB that
every agent reads and then discounts.

docs/ai-first-development-workflow.md goes with it. Its own header says it
describes no part of this project and is kept only so the review's citations
resolve. Git history keeps both.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Audit the open issues and the documented commands

Verification, not repair. Its deliverable is a report plus any small corrections it justifies — it may end with no doc change at all, and that is a valid outcome.

**Files:**
- Modify: `README.md` / `PROJECT_DOCUMENTATION.md` (only if a documented flag is found wrong)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing.

- [ ] **Step 1: Check each open issue against the code**

The follow-ups are now GitHub issues #15–#23 on the public mirror, not a file
in the repo. Read them with `gh issue list --state open` and
`gh issue view <N>`. For each, run the stated check and record open/resolved:

| Issue | Check |
|---|---|
| #15 SIFT configs out of step | `grep -n "real_path\|output_dir" configs/sift/*.yaml` — open if they still name `data/sift_base.npy` and `runs/sift_gan_v*` |
| #16 Re-measure angular families | `grep -n "metric" src/eval/ann_difficulty.py` — open if it still measures L2 unconditionally |
| #17 Fold `l2_normalize` copies | `grep -rn "def l2_normalize" src/eval/` — open if more than zero remain |
| #18 `build_generator` rejects `sparse` | `grep -n "sparse\|gated" src/models/generator.py` — open if there is no `sparse` alias |
| #19 v2 checkpoint outside the repo | Not checkable from here; a claim about `tig-gpu`. Leave alone. |
| #20 DEEP ladder rests on one seed | Not checkable from here; needs GPU runs. Leave alone. |
| #21 `spectrum_reg_alpha` too small | Not checkable from here; needs GPU runs. Leave alone. |
| #22 `ann_difficulty.py` could inherit `--dataset` | `grep -n "dataset" src/eval/ann_difficulty.py` — open if it takes no dataset argument |
| #23 `eda.pipeline.run` builds `EdaConfig` early | `grep -n "EdaConfig" src/eval/eda/pipeline.py` — open if the config is still built before the output directory |

Note #17's title now names `eda.series.maybe_l2_normalize`, not
`eda_report.maybe_l2_normalize` — the eda split moved it. That is the issue's
own text and correct.

Report anything that looks resolved. **Do not close an issue.** Closing a
follow-up is a judgement about whether the underlying question is settled, and
`AGENTS.md` reserves that kind of call for a human. Report and stop.

- [ ] **Step 2: Verify the documented commands against argparse**

Run: `grep -rn 'add_argument(' src/data/fetch.py src/train/train_wgan_gp.py src/eval/evaluate_distribution.py src/eval/evaluate_file_to_file.py src/sample/generate.py src/eval/compare_variants.py`

Cross-check every flag appearing in `README.md`'s quick start and in
`PROJECT_DOCUMENTATION.md`'s walkthrough. The flags to confirm are
`--real-path`, `--checkpoint`, `--config`, `--output-dir`, `--synthetic-path`,
`--num-samples`, `--output-path`, `--dataset`, `--variants-manifest` and
`--allow-missing`.

Expected: every one exists. This was checked while writing the plan and found
clean; the step exists to catch a change since then. If a flag is missing,
correct the documented command to match the code — the code wins.

- [ ] **Step 3: Confirm the suite**

Run: `make check`
Expected: green

- [ ] **Step 4: Commit, or report no change**

If Steps 1 or 2 produced edits:

```bash
git add -A
git commit -m "docs: correct the follow-up and command claims the audit found stale

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

If they did not, make no commit and report which issues look resolved so a
human can decide whether to close them.

---

## Verification

After Task 7, confirm the whole change:

```bash
make check
git log --oneline origin/main..HEAD
```

Expected: `make check` green, and a commit per task. Then confirm the sweep
actually did what it claims:

```bash
grep -rnE '`[A-Za-z0-9_./-]+\.(md|py|ya?ml|json)(:[0-9]+)' AGENTS.md README.md PROJECT_DOCUMENTATION.md data/README.md docs/datasets/*.md
```

Expected: no output — every citation is now an anchor or a symbol.
