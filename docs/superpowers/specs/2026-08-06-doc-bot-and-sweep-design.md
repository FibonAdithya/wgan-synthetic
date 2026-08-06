# Documentation review bot and one-time doc sweep

Date: 2026-08-06
Status: design, approved for planning
Base: `origin/main` at `5f4db7a`

## Problem

The documentation in this repo is unusually good and unusually load-bearing.
`AGENTS.md` is a router: it tells an agent which document wins, which five
invariants are silent until violated, and which decisions need a human. Every
agent that touches this repo — including the PR reviewer added on
`ci/claude-pr-review` — is told to read it first.

It is also, right now, wrong in a way nothing can catch.

Seven of the nine `file:line` citations in `AGENTS.md` point at the wrong
line. The gate — invariant 1, the single most important claim in the project —
cites `PROJECT_DOCUMENTATION.md:274`, which reads `- lambda_gp: 5.0`. The gate
is actually at line 303. The variant ladder cites `:172`, a blank line; it is
at 201. `generator_type` cites `:146`, mid-sentence in a different section; it
is at 245. Metric definitions cite `:323`, another blank line; they are at 365.
The checkpoint/`run_config.yaml` invariant cites `:216`, a table separator; its
real home is the closing paragraph of the `generator_type` section at 258.

These citations were almost certainly correct when written.
`PROJECT_DOCUMENTATION.md` then grew above them, and every insertion pushed
every citation below it out of true. Nothing in `make check` looks at prose, so
the rot was free. `FOLLOWUPS.md` has drifted the same way, by one or two lines
each, on all four of its `l2_normalize` citations.

Two things follow, and this design does both: something must stop the rot from
recurring, and the accumulated rot must be cleared once.

## Non-goals

- **Re-measuring anything.** The numbers in `docs/datasets/*.md` need the GPU
  box and corpora that are not in this checkout. The sweep flags a number it
  cannot check; it never edits one.
- **Touching `docs/superpowers/`.** `docs/superpowers/README.md` states these
  are snapshots, kept for reasoning that is not recoverable from a diff, and
  deliberately not updated. Both the bot and the lint exclude the directory.
- **Rewriting docs automatically.** `AGENTS.md` already tells reviewers to flag
  stale claims rather than rewrite them. The bot comments; it does not commit.
- **Anything `AGENTS.md` reserves for a human** — gate bands, config
  re-pointing, pinned requirements.

## Architecture

Three units that share no code.

| Unit | File | Depends on |
|---|---|---|
| Doc review bot | `.github/workflows/docs-review.yml` | nothing |
| Doc lint | `tests/test_docs_references.py` | the citation format the sweep introduces |
| The sweep | edits to the authoritative docs | nothing |

The bot is independent and can land on its own, before or after the rest. The
lint and the sweep are coupled by convention rather than by code — the sweep
converts citations to a checkable format, the lint enforces that format — so
they must land in the same change, or the lint fails against un-swept docs.

## Unit 1 — the doc review bot

A sibling of `claude-review.yml`, not an extension of it. The code reviewer
keeps its existing third-priority doc check for things it notices in passing;
this workflow does the dedicated pass, in its own comment, so a doc finding is
not buried under code findings.

    on:
      pull_request:
        types: [opened, synchronize]

No merge or push trigger. Every commit on `main` since the ladder work arrived
through a PR, so a post-merge pass would re-review the diff the PR pass just
reviewed, at a second bill, and report after the cheapest moment to act has
passed. If direct-to-main pushes ever start happening, revisit.

Permissions are `contents: read`, `pull-requests: write`, `id-token: write` —
deliberately no `contents: write`. The bot cannot push to the branch, so a
confidently wrong rewrite of `PROJECT_DOCUMENTATION.md` costs a comment you
ignore rather than a commit you have to catch in review. Auth matches
`claude-review.yml`: `claude_code_oauth_token` for the subscription, plus the
job's own `github_token` because the Claude GitHub App is not installed here.
Concurrency group `docs-review-${{ github.ref }}` with `cancel-in-progress`.

The prompt directs the model to read `AGENTS.md` first, then check four things
against the diff:

1. A claim in an authoritative doc that the diff has just falsified. The order
   of authority is `AGENTS.md`'s: code and configs beat
   `PROJECT_DOCUMENTATION.md`, which beats `README.md`, `data/README.md` and
   `docs/datasets/*.md`.
2. Behaviour that changed with no corresponding doc update — a new CLI flag, a
   changed default, a renamed config key.
3. Drift touching the five invariants, which is the expensive kind: a diff that
   quietly changes what a variant number means, or that separates a checkpoint
   from its `run_config.yaml`.
4. Any newly introduced `file:NNN` citation. The lint catches these too; the
   bot explains why the anchor form is wanted.

And explicitly not: rewriting docs, style that `make check` already enforces,
anything under `docs/superpowers/`, or the human-reserved decisions above. If
the diff is clean it says so in one comment rather than manufacturing findings.

## Unit 2 — the doc lint

`tests/test_docs_references.py`, picked up by the existing pytest run in
`make check`. CPU-only, no network, consistent with the rest of the suite.

Four deterministic checks over the authoritative set — `AGENTS.md`,
`CLAUDE.md`, `README.md`, `PROJECT_DOCUMENTATION.md`, `FOLLOWUPS.md`,
`data/README.md`, `docs/datasets/*.md`:

1. **Path refs resolve.** Any backticked ref containing `/` must exist, minus
   an allowlist for paths that are correctly absent from a clean checkout.
2. **Markdown anchors resolve.** `PROJECT_DOCUMENTATION.md#ann-difficulty--the-gate`
   must match a real heading, slugged by GitHub's algorithm.
3. **Python symbol refs resolve.** `src/eval/ann_difficulty.py::lid_median`
   must exist, checked by parsing the module's AST — no import, so no torch
   load and no side effects.
4. **No new `file:NNN` citations** in the authoritative set.

### What the allowlist is for, and why check 1 finds nothing today

A naive scan reports three broken paths in the authoritative docs. All three
are false positives, and understanding why sets the allowlist:

- `PROJECT_DOCUMENTATION.md:598` `eval/metrics.json` and `:600`
  `eval_file_to_file/metrics.json` sit in a list titled "Typical run directory
  contents". They are run-dir-relative, not repo-relative.
- `FOLLOWUPS.md:70` names an absolute path on `tig-gpu`. It describes another
  machine.

So the allowlist covers generated artifacts (`runs/**`, `data/*.npy`), run-dir
prefixes (`eval/`, `eval_file_to_file/`, `eval_embeddings/`), and absolute
paths. With it, check 1 passes on the current tree.

That is the honest scoping: **check 1 is a regression guard, not a fixer.** It
finds nothing today and exists so that the first genuinely broken path fails
CI. The value in this design is concentrated in checks 2–4, which only become
possible once the sweep changes the citation format.

### Why anchors instead of correcting the line numbers

Correcting `274` to `303` fixes today and rots on the next insertion above line
303. An anchor names what is being cited rather than where it currently sits,
so it survives every edit that does not rename the heading — and when someone
does rename the heading, the lint fails loudly instead of silently pointing at
a hyperparameter. The same argument makes `::symbol` the right form for code
refs.

The one risk is slug-algorithm mismatch: if the test's slugger disagrees with
GitHub's, the test passes while the rendered link 404s. Mitigation is to
implement GitHub's documented rules (lowercase; strip punctuation other than
hyphens and underscores; spaces to hyphens) and assert them against the real
headings in `PROJECT_DOCUMENTATION.md`, including the awkward one:
`## ANN difficulty — the gate` slugs to `ann-difficulty--the-gate`, the em dash
vanishing and leaving a double hyphen.

**Semantic drift is out of scope for the lint.** A doc sentence that is simply
untrue about the code is not mechanically detectable; that is the bot's job.
The two are complementary, and the split is what keeps the bot's prompt short.

## Unit 3 — the sweep

Ordered by value.

**3a. Convert all 15 line-number citations.** Nine in `AGENTS.md`, four in
`FOLLOWUPS.md`, two in `AGENTIC-REVIEW.md` (which 3d deletes, so those resolve
themselves). The `AGENTS.md` targets:

| Invariant / row | Cited | Reads today | Correct anchor |
|---|---|---|---|
| 1, the gate (×2) | `:274` | `- lambda_gp: 5.0` | `#ann-difficulty--the-gate` |
| 2, variant ladder (×2) | `:172` | blank | `#model-variants-the-per-dataset-ladder` |
| 4, checkpoint + run config | `:216` | `\|---\|---\|---\|---\|` | `#generator_type` |
| architectures row | `:146` | mid-sentence, wrong section | `#model-architecture` |
| evaluation row | `:323` | blank | `#metric-definitions` |

The two correct ones — `README.md:10-26` and `docs/superpowers/README.md:6-13`
— convert to anchors as well, for uniformity and so check 4 can be absolute.

**3b. Qualify the bare code citations in `FOLLOWUPS.md`.** The four
`l2_normalize` copies are cited as `evaluate_file_to_file.py:43`,
`plot_distance_cdf.py:30`, `plot_distance_cdf_pillow.py:46` and
`plot_embedding_clusters.py:33` — no directory, and each line number off by one
or two. Verified still open: four copies remain, byte-identical, against
`eda_report.maybe_l2_normalize`. They become `src/eval/<file>.py::l2_normalize`.

**3c. Correct the GitHub Issues claim.** `AGENTS.md:34` and `FOLLOWUPS.md:3`
both say Issues are disabled. They are enabled on `origin`, the public mirror,
and disabled on `upstream`. The sentence needs to say which remote it means.

**3d. Delete `AGENTIC-REVIEW.md` and `docs/ai-first-development-workflow.md`,**
with the `README.md` block that introduces them. The review's five surviving
conclusions were: no `AGENTS.md`, no CI, a gate documented but not executable,
an untested sampling path, and hard-coded run directories. All five have since
been fixed — `AGENTS.md` exists, `.github/workflows/ci.yml` exists, `34c4fbf`
made the gate executable, the generate-coverage PR covered sampling, and
`f1d13ca` replaced the hard-coded run dirs with a manifest. The file already
carries a note that its counts describe a stale base. What remains is 13KB that
every agent reads and then discounts. The vendored workflow guide describes no
part of this project and is kept, by its own header, only so the review's
citations resolve; it goes with the review. Git history keeps both.

**3e. Audit `FOLLOWUPS.md` entry by entry** against the code, flagging entries
that look resolved rather than deleting them — closing a follow-up is a
judgement about whether the underlying question is settled, not a doc edit.
`deep_ladder_summary.json` is confirmed present at
`docs/datasets/deep_ladder_summary.json`, as its entry claims.

**3f. Verify the command invocations** in `README.md` and
`PROJECT_DOCUMENTATION.md` against the actual `argparse` definitions — flag
names, defaults, required arguments. Checkable from the repo, and the quick
start is the most-run prose in the project.

## Known conflict

The unmerged branch `docs/followups-to-issues` (`0f3ff85`, no open PR) deletes
`FOLLOWUPS.md` and repoints its references in `AGENTS.md`, `README.md` and
`PROJECT_DOCUMENTATION.md` at issues #15–#22. This design branches from `main`
and treats `FOLLOWUPS.md` as live, so 3b, 3c and 3e conflict with it textually.

Accepted deliberately. The conflicts are prose-level in three files, and the
sweep improves the surrounding text either way. If the migration lands first,
3b's citations move into the issue bodies, 3c disappears, and 3e becomes a pass
over the issues instead — but the anchors, the lint and the bot are unaffected,
which is the bulk of the work.

## Testing

- The lint tests itself: checks 2–4 run against the swept docs, and the slug
  function gets direct unit tests including the em-dash heading.
- The sweep is verified by `make check` going green with the lint active, which
  is only possible if every converted citation resolves.
- The workflow cannot be tested before merge. It is validated by inspection
  against `claude-review.yml`, which is known to run, and by its first PR.

## Success criteria

`make check` fails if any doc cites a path, anchor or symbol that does not
exist. `AGENTS.md` points an agent at the gate rather than at `lambda_gp`. The
repo carries no document whose findings are entirely resolved. A PR that
changes behaviour without updating the docs gets a comment saying so.
