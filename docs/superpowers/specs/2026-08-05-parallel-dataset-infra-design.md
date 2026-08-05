> **AI-generated working note.** Written by Claude during development and kept
> for provenance. It is not the source of truth: where this file disagrees with
> `PROJECT_DOCUMENTATION.md`, the latter wins.

# Parallel per-dataset agents against one shared GPU

Date: 2026-08-05
Branch: `infra/parallel-dataset-runner`

## Problem

Phase (a) of the multi-dataset design left six benchmark families configured
and documented but untrained. The intended next step is one agent per family,
each researching its dataset and training its ladder, working in parallel.

The hardware does not allow that reading of "parallel". `tig-gpu` is a single
RTX 4060 with 8GB of VRAM. Six agents cannot train six models at once; they
contend for one card. Meanwhile the work that *is* parallelizable — fetching,
EDA, LID and hubness profiling, config authoring, writing each dataset page —
is CPU-bound and would sit idle behind a naive one-at-a-time discipline.

There is a second problem the box already demonstrates. `/workspace` holds four
checkouts of this project at four different commits, totalling 10.8GB, three on
stale branches and one with no git history at all. Two carry uncommitted
changes. That is what unmanaged parallel access produced last time, and nothing
currently prevents it recurring.

## Constraints

- **One RTX 4060, 8GB VRAM**, 64 cores, 125GB RAM, ~96GB free disk.
- **The box is an ephemeral vast.ai container.** Unprivileged: no
  Docker-in-Docker, no kernel modules, no sysctls. Long-running processes are
  managed by supervisor. If the instance dies, `/workspace` dies with it.
- **No local data or CUDA.** Every measurement, not just training, executes on
  the box. "Research in parallel" is still remote work.
- **Agents must not git-operate on the box.** That is the mechanism that
  produced the current sprawl.
- **The box must be rebuildable.** Host identity is one variable; nothing on
  the box may be hand-made.

## Design

### Two lanes, not one queue

Training needs the card. Fetching and profiling do not — `eda_report` and
`ann_difficulty` at the locked canonical N=20000, k=100 are CPU work. Putting
both behind one serial queue would waste 64 cores; putting both in a parallel
pool would thrash 8GB of VRAM.

```
agent (per dataset, local worktree)
      │  writes job spec, gets id, moves on
      ▼
/workspace/queue/pending/<id>.json
      │
      ├──► CPU lane  ── N concurrent ── fetch, eda_report, ann_difficulty
      └──► GPU lane  ── 1 slot        ── train_wgan_gp
                          │
                          ▼
            runner (supervisor, single instance)
              reap → claim → run → artifacts → commit
```

### Queue

A directory tree under `/workspace/queue`, with states as subdirectories —
`pending/`, `running/`, `done/`, `failed/` — and transitions by atomic
`rename(2)`. Job specs are JSON files.

No database and no daemon dependency. The whole system state is legible to
`ls`, survives every agent session ending, and can be repaired with `mv`. A
queue that needs a running service to inspect is a queue that becomes opaque
exactly when something has gone wrong.

A job spec:

```json
{ "id": "glove-v0-train-01", "dataset": "glove", "lane": "gpu",
  "branch": "ds/glove", "commit": "a1b2c3d",
  "cmd": ["python", "-m", "src.train.train_wgan_gp",
          "--config", "configs/glove/v0.yaml"],
  "timeout_s": 21600, "attempts": 0 }
```

`commit` is pinned, not just `branch`. The runner checks out that exact tree,
so a returned `summary.json` is attributable to a config that can be read back.
A branch reference alone would let the tree move under a queued job and produce
a number nobody can reproduce.

Jobs are deduplicated on `dataset` + variant + `commit`. Resubmitting an
identical job is a no-op rather than a second run.

### Runner

One supervisor-managed process, the sole launcher of work on the box. It polls
`pending/`, admits CPU jobs up to a concurrency cap and GPU jobs strictly one
at a time, and moves specs between state directories.

The CPU cap defaults to **4**, not to the core count. `eda_report` and
`ann_difficulty` are memory- and BLAS-bound rather than core-bound, and a k-NN
pass over 20000 rows at 1536 dimensions already threads internally; admitting
six of those at once oversubscribes BLAS and slows all six. Four is a starting
point to be measured in phase 3, not a tuned value.

**Workers never touch git.** Six concurrent CPU jobs committing to six branches
in one checkout would corrupt the index. Workers write artifacts to disk only;
the runner's single main loop performs every git operation between polls. All
repository mutation is serialized through one thread by construction rather
than by discipline.

### Reaper

Runs on runner start and between jobs:

- Read the claim file; if its pid is dead, release and log.
- Kill orphaned processes holding CUDA that no live job owns.
- Remove `.part` files and partial run directories.

Plus a per-job wall-clock watchdog. This is the deterministic half of cleanup.
It belongs in the runner rather than in an agent because it must run when no
agent is alive — which is exactly when a leaked job needs reaping.

The judgment half — disk pressure, stale checkouts, orphaned run directories,
drifted branches — goes to a periodically scheduled housekeeping agent that
*reports and proposes*, and does not delete.

### GPU lock: its own repository

`gpu_lock.py` moves out of this repo into a small standalone project, installed
onto any box by bootstrap and used by every project that touches the card.

The lock is already host-global in effect: it keys on the GPU UUID and writes
to a fixed directory, so two checkouts running that code contend correctly
today. Extracting it does not fix a live bug. It removes a latent one. The lock
is only correct if every participant derives the same key and the same path,
and the box already holds four checkouts at four commits. The moment one
changes the key format or the lock directory, they hold *different* locks for
the *same* card — the failure the module's own docstring warns about: isolation
that looks correct and is not. One shared installation makes that impossible
instead of merely unlikely.

The runner design sharpens this. Once the runner is the sole launcher of GPU
work in this project, it enforces the single slot itself and the lock becomes
nearly vestigial internally. What remains for the lock is defending against
what the runner does not control: a stray manual invocation, another project, a
direct ssh run. That residual responsibility is entirely cross-project, so it
belongs above any one repo.

What must be shared is the protocol, not merely the file:

| | |
|---|---|
| Lock path | fixed directory, file named by GPU UUID |
| Key derivation | `torch.cuda.get_device_properties(...).uuid`, index-keyed fallback |
| Claim file | JSON: pid, owner, command, started_at |

Any implementation pinning those three interoperates.

Enforcement stays **advisory**, as `flock` is, with one addition: a preflight
that refuses to start when the card already carries foreign CUDA processes,
naming the pid and command. This cannot stop a determined direct run. It turns
accidental contention into a fast, readable failure instead of an OOM
thirty minutes in.

*Open at implementation time:* the new repository's owner and name. This repo
lives under `Daniel-T-S-Adams`; creating a repository is an outward-facing act
and needs confirmation before it happens.

### Isolation

Each agent owns `.claude/worktrees/<dataset>` locally and never touches
another's. The box gets a **single** checkout, owned by the runner, which is
the only thing that git-operates there. Agents submit jobs and read artifacts.

This removes the contamination vector rather than asking six agents to respect
a convention. Per-dataset configs and documentation barely overlap, so the
convention would probably hold — but it is the arrangement that already failed
once, and structure is cheaper than vigilance.

### Artifacts

Runs land in `/workspace/runs/<dataset>/<variant>/`. The runner commits
`summary.json`, `run_config.yaml` and `run_metadata.json` to the dataset's
branch. Checkpoints and report HTML stay on the box and are fetched on demand.

The split follows what each artifact is for. The gate is four numbers per
dataset; versioning them means a ladder decision shows up in a PR diff and can
be compared against an older one. Weights are large, reproducible from a pinned
commit plus a seed, and belong in git only by accident.

### Failure handling

| Failure | Response |
|---|---|
| Job exits non-zero | → `failed/`, stderr tail captured into the spec so the agent reads it without ssh |
| Runner dies mid-job | supervisor restarts; reaper finds `running/<id>.json` with a dead pid, requeues **once** via `attempts`, then fails it |
| Wall-clock exceeded | watchdog kills, marks failed, no retry — a hung job is a bug, not a transient |
| CUDA OOM | detected distinctly, never retried blindly; at 1536 dimensions on 8GB this is a config error to surface |
| Duplicate submission | deduplicated on `dataset` + variant + `commit` |

The requeue-once rule is the load-bearing one: without an attempt counter, a
crash-looping job occupies the only card indefinitely.

**Accepted risk.** The instance is ephemeral. Committing metrics as jobs
complete means findings survive an instance loss; checkpoints do not. The box
has an rclone configuration already provisioned, so pushing passing checkpoints
to durable storage is available as an opt-in. Not built now.

### Bootstrap

`tools/box/bootstrap.sh` takes a bare instance to a working runner,
idempotently: clone, create the venv, install requirements, write the
supervisor program file, create the queue tree, start the runner. The
supervisor configuration ships in the repo rather than being added by hand.

Host identity lives in exactly one place, `tools/box/host.env`, holding
`BOX_SSH_HOST=tig-gpu`. Rebuilding is that one edit plus a bootstrap run.

Rebuild costs a re-fetch of roughly 15GB of HDF5, about half an hour. vast.ai
volumes could persist `/workspace/data-cache` across instances; slow is
acceptable before that complexity is worth it.

### Disk budget

Six HDF5 sources ≈ 15GB. Six 250k `.npy` subsets ≈ 3GB. Against ~96GB free
once the 10.8GB of stale checkouts is reclaimed. Comfortable at 250k. All six
at 1m would add ~12GB and is worth doing on demand rather than upfront.

## Sequencing

Two dependencies set the order. The queue work depends on nothing. The dataset
agents are hard-blocked on PR #7, which is what puts `configs/<dataset>/` and
`src/data/fetch.py` on `main`.

| Phase | Work | Blocked on |
|---|---|---|
| 0 | Cleanup, local and box | — |
| 1 | Extract the GPU lock to its own repo; cherry-pick device resolver, resume and EMA-checkpoint commits onto `main` | — |
| 2 | Queue, runner, reaper, bootstrap | phase 1 |
| 3 | Rebuild box from bootstrap; fetch and profile all six | #7 merged, phase 2 |
| 4 | Breadth-first `v0` across all six families | phase 3 |
| 5 | Gate bands per dataset, then deepen ladders | phase 4 |

Phase 3 is where the parallelism pays: six fetches and six `eda_report` runs
occupy the CPU lane at once. Phase 4 is serial on the card — at roughly 1–3
hours per 30k-step run, one rung across six families is most of a day.

This spec covers more than one implementation plan. **Phases 0–2 are the first
plan** — cleanup, the lock extraction, and the queue/runner/reaper/bootstrap
build. Phases 3–5 are execution against that infrastructure and get their own
plan once it exists, because what they should contain depends on what phase 3
measures.

**Breadth-first is deliberate.** A `v0` everywhere before any `v1` sets gate
bands against cross-family evidence rather than one family's idiosyncrasies,
and surfaces whether openai at 1536 dimensions fits in 8GB on the first day
rather than after three days spent deepening SIFT.

## Cleanup

Local, mechanical:

- Prune the dead `/tmp/claude-1000/deep-pr` worktree.
- Fast-forward `main`, currently behind 14.
- Drop `ann/difficulty-panels`, superseded by PR #7.
- Push `worktree-gan+infra-exec` and `worktree-gan+next-iteration`, which exist
  only on one disk.

The box is **not** mechanical, and nothing there is deleted before it is
inspected:

- `/workspace/wgan-v3` — 3.3GB, **not a git repository**. Its contents exist
  nowhere else.
- `/workspace/wgan-synthetic` — 11 uncommitted files; also holds the eight run
  directories carrying the trained SIFT ladder the open PRs reference.
- `/workspace/wgan-sparse-v2` — 7 uncommitted files.
- `/workspace/wgan-sparse` — clean, on `worktree-sparse-generator`.

After bootstrap there is one runner-owned checkout. Rescue first, reclaim
second.

The `pentest-scorer` supervisor process, unrelated to this project and running
23 hours, is to be killed.

## Not in this design

- **Multi-GPU or multi-host scheduling.** One card, one box. The queue's lane
  abstraction would extend, but nothing here anticipates it.
- **Durable checkpoint storage.** rclone is available; opt-in later.
- **Gate band values.** Set in phase 5 from measured ladders, per dataset, and
  recorded in `docs/datasets/<name>.md`.
- **The v3/v4 model track** on `worktree-gan+infra-exec` — `structured_gated`,
  the log-ratio regularizer and seven probe tools. It stays on its branch until
  someone owns it. Only the infrastructure commits are extracted in phase 1.
