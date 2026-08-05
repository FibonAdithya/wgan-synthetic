# Building an AI-First Development Workflow

> **Vendored external document — not project documentation.** Copied verbatim
> on 2026-08-05 from `ai-first-development-workflow.md` at the root of the
> sibling `tig-cpu` repository. Nothing below describes this repository; it is
> a general guide about a different codebase, kept here only so that the
> references in `AGENTIC-REVIEW.md` resolve inside a fresh clone.
>
> Like the AI working notes in `docs/superpowers/`, this file is **not
> authoritative**. Where it disagrees with `PROJECT_DOCUMENTATION.md`,
> `README.md`, `data/README.md`, or the code, those win. It is a snapshot and
> is not updated as either repository changes; edits belong upstream in
> `tig-cpu`, not here.

This guide describes a practical way to structure a software project so AI
coding agents can make useful changes without becoming an uncontrolled source
of risk. It is distilled from a working, AI-heavy monorepo, but it is written
to stand alone: no knowledge of that repository or product is required.

The central lesson is simple:

> AI development works best when the repository supplies context, executable
> feedback, durable memory, and explicit authority boundaries.

A longer prompt is not a substitute for those things. The highest-value work
is turning unwritten team knowledge into versioned files and turning subjective
expectations into commands and release gates.

## 1. The operating model

An effective AI-first workflow is a closed loop:

```text
Issue or goal
    |
    v
Load repository rules and relevant domain context
    |
    v
Research current behavior -> write a bounded plan
    |
    v
Implement on a feature branch -> run local checks
    |
    v
Open a pull request
    |
    +--> independent, specialized AI reviews
    |          |
    |          v
    |     one bounded fix pass
    |          |
    +----------+
    |
    v
Deterministic CI -> human approval -> merge queue
    |
    v
Deploy exact candidate to staging -> end-to-end tests -> user acceptance test
    |
    v
Production promotion
    |
    v
Update the repository's durable context
```

The loop has five supporting systems:

| System | Purpose | Typical repository artifacts |
| --- | --- | --- |
| Context | Tell an agent what matters for this task | Root instructions, module skills, architecture docs |
| Memory | Preserve decisions and discoveries between sessions | ADRs, plans, progress logs, runbooks |
| Feedback | Detect whether the change works | Unit tests, integration sandbox, E2E, UAT |
| Authority | Define what an agent may decide or mutate | Branch protection, permissions, human gates |
| Coordination | Prevent agents and automation racing each other | Pipeline states, bot identities, concurrency rules |

If one of these is missing, the human becomes the missing system. They must
re-explain architecture, remember earlier decisions, manually test changes, or
untangle concurrent edits.

## 2. Put a short operating contract at the repository root

Every agent session should begin with one canonical instruction file. Use the
filename supported by your tools, such as `AGENTS.md`; generate tool-specific
files such as `CLAUDE.md` from it if required.

The root contract should be a router and a set of non-negotiable rules, not an
encyclopedia. Include:

- The repository's purpose in one paragraph.
- The source-of-truth hierarchy.
- The branch, review, and deployment model.
- Commands that define a valid change.
- Security and secrets rules.
- Actions reserved for humans.
- A map from code areas to deeper context files.
- The definition of done.

Example:

```markdown
# Repository operating contract

## Source of truth

Runtime behavior and tests outrank prose. Accepted ADRs explain why a design
exists. Plans describe intended work and may be stale; verify them against code.

## Mandatory workflow

1. Work on a feature branch; never push to `main`.
2. Read the module guide for every area you change.
3. Run `make check` and the relevant integration scenario.
4. Open a PR linked to its issue.
5. Do not approve, merge, or deploy. Those actions require a human.
6. After staging deployment, run a user-level acceptance scenario.

## Safety

- Use the secrets manager; never create or commit `.env` files.
- Do not run migrations, production writes, or destructive cleanup without
  explicit authorization.
- Do not weaken tests or CI to make a check pass.

## Context routing

- API or database work: `docs/agent/api.md`
- Web work: `docs/agent/web.md`
- Deployment work: `docs/agent/deployment.md`
```

Why this works:

- Rules travel with the code and are reviewed like code.
- The agent does not depend on a human remembering to repeat constraints.
- Concrete commands are less ambiguous than instructions such as “test it
  thoroughly.”
- Explicit human-only actions prevent completion pressure from being mistaken
  for authorization.

Keep the file internally consistent. If it says both “enable auto-merge” and
“only a human may merge,” the agent cannot reliably infer the real policy.

## 3. Use progressive context, not one giant prompt

A large project cannot fit all useful knowledge into every task. Instead, use a
small root router plus task-specific context packages. These are often called
skills, playbooks, or agent guides.

A useful structure is:

```text
AGENTS.md                         # Always-loaded operating contract
docs/agent/
  repository.md                  # System map and cross-cutting conventions
  api.md                         # API contracts, state machines, commands
  web.md                         # UI architecture and test conventions
  data.md                        # Schemas, migrations, invariants
  deployment.md                  # Environments, release and rollback
  add-provider.md                # A repeatable specialist workflow
```

Each context package should answer five questions:

1. When should an agent load this?
2. What does this subsystem own, and what does it not own?
3. What are its public interfaces and invariants?
4. Where are the relevant files and integration points?
5. How can the agent verify a change?

Example module guide:

````markdown
---
name: payments
description: Load for checkout, invoices, credit balances, pricing, or payment webhooks.
---

# Payments

## Responsibilities

- Converts provider events into idempotent ledger transactions.
- Never treats a browser redirect as proof of payment.

## Key files

| Path | Purpose |
| --- | --- |
| `src/payments/webhooks.ts` | Verifies and dispatches provider events |
| `src/payments/ledger.ts` | Owns balance mutation |

## Invariants

- A provider event ID is applied at most once.
- Money is stored as integer minor units.
- Every balance mutation creates an immutable ledger row.

## Verification

```bash
npm test -- payments
./scripts/e2e-payment-sandbox.sh
```
````

Prefer interfaces, ownership, state transitions, failure modes, and executable
runbooks over exhaustive lists of private functions. Agents can read private
implementation from the code. The guide should supply the relationships that
are expensive to rediscover.

For cross-module work, explicitly require multiple guides. A new API field, for
example, may require the API, CLI, web, data, and deployment contexts.

### Make context portable across AI tools

Do not maintain hand-edited copies under several vendor directories. They will
drift. Keep one canonical source and either:

- Point every tool at it.
- Generate vendor-specific files and fail CI when regeneration changes them.
- Use symlinks where all supported environments preserve them.

Add a check such as:

```bash
./scripts/generate-agent-context.sh
git diff --exit-code -- AGENTS.md .claude/ .codex/
```

The goal is tool diversity without multiple conflicting truths.

## 4. Treat documentation as the agent's long-term memory

An AI session is temporary. Repository memory is durable. Separate different
types of knowledge so an agent can reason about their authority and freshness.

| Document type | Question answered | Mutation policy |
| --- | --- | --- |
| Root contract | How must work be performed? | Current and tightly reviewed |
| Module guide | How does this area work now? | Updated with interface changes |
| ADR | Why was a durable decision made? | Immutable; supersede with a new ADR |
| Plan | How might or did a substantial change proceed? | Status-marked; not automatically authoritative |
| Runbook | How is an operational task executed safely? | Tested and periodically re-verified |
| Progress log | What did a long-running agent discover? | Append-oriented working memory |
| Archive | What did the system used to do? | Frozen and clearly non-authoritative |

State the precedence explicitly. A practical hierarchy is:

```text
tests and current code
    > current contracts and module docs
    > accepted ADRs
    > active plans
    > historical plans and archives
```

This does not mean code is always correct. It means an agent must not change
current behavior merely because an old plan describes something different.

### Give operational docs a schema

Use frontmatter or a standard header:

```yaml
title: Restore a failed search index
owner: search-platform
status: active
last_verified: 2026-07-15
source_paths:
  - services/indexer
```

Use required runbook sections:

- Purpose and scope
- Preconditions
- Steps
- Validation
- Rollback
- Escalation
- References

This improves both machine navigation and human trust. Structural validation
can require the fields and headings, while a scheduled exercise or owner review
is still needed to establish factual freshness.

### Update context from interface changes

After implementation, inspect the diff and map changed paths to context:

```text
API route changed       -> API guide + user API docs
Database schema changed -> data model + migration runbook
Public CLI changed      -> CLI guide + user docs
Workflow changed        -> deployment guide
Architecture changed    -> new or superseding ADR
```

Automate the mapping and review, but do not let an unconstrained model rewrite
all prose after every commit. Prefer narrow, evidence-based edits. A docs-sync
reviewer should compare the changed code with only the affected documents.

## 5. Write plans that an agent can execute and audit

For non-trivial work, planning is context construction. A good plan prevents
the implementation agent from repeatedly rediscovering the same facts.

Include:

- Goal and user-visible outcome.
- Non-goals.
- Current behavior, backed by file paths and code inspection.
- Constraints and invariants.
- Decisions already made and genuinely open questions.
- A phased file-change outline.
- Compatibility and migration strategy.
- Test matrix and acceptance criteria.
- Rollout, observability, and rollback.
- Status: proposed, active, implemented, superseded, or abandoned.

Avoid plans that merely restate the ticket as a list of implementation verbs.
An agent benefits more from “the same response is assembled in two handlers;
both must carry the field” than from “update the response.”

### Use acceptance criteria as an executable contract

Weak:

```text
- Add pagination.
- Test it.
```

Strong:

```text
- `GET /items?limit=2` returns two items and a non-empty cursor.
- Passing that cursor returns the next two items without duplicates.
- An invalid cursor returns 400 with code `INVALID_CURSOR`.
- The query has a supporting index and does not perform one query per item.
- `npm test -- items-pagination` and `make integration` pass.
```

The strong version gives implementation, review, and UAT agents the same
observable target.

### For long initiatives, keep a task ledger and progress memory

A useful autonomous-work format is a machine-readable task list:

```json
{
  "id": "DATA-004",
  "title": "Add resumable uploads",
  "priority": 4,
  "acceptanceCriteria": [
    "Interrupted upload resumes from the last confirmed part",
    "Existing single-request upload remains compatible",
    "Integration test passes"
  ],
  "passes": false,
  "notes": ""
}
```

Pair it with an append-only progress log containing:

- Commands that really work in this repository.
- Patterns and integration points discovered in code.
- Root causes, not only symptoms.
- Files changed and checks run.
- New constraints for later tasks.
- Release-gate results.

Do not rely on a `passes` Boolean set by the same agent. Derive completion from
test results, commits, and deployment evidence wherever possible.

## 6. Design the repository for cheap, deterministic feedback

Agents perform much better when they can validate assumptions without waiting
for a human or touching shared infrastructure.

Provide one discoverable command for each feedback layer:

```makefile
check: format-check lint typecheck test
integration: check
	./sandbox/run.sh test
```

The layers should have different jobs:

1. Static checks catch syntax, type, formatting, lint, and policy violations.
2. Unit tests isolate business rules and edge cases.
3. Integration tests exercise component boundaries with local dependencies.
4. A local sandbox proves a minimal user journey without cloud credentials.
5. Staging E2E proves the exact release candidate in realistic infrastructure.
6. UAT tests the change as a user would, including meaningful bad input.

### A sandbox is especially valuable for AI development

A good sandbox:

- Starts with one command.
- Uses isolated ports and state so runs can coexist.
- Avoids production credentials.
- Supports seed, submit, inspect, and stop operations.
- Has a canonical smoke scenario.
- Always collects logs and cleans up.
- Returns non-zero on failure.

Example:

```bash
./sandbox/run.sh start --id "$RUN_ID"
./sandbox/run.sh submit examples/hello-world.yaml --follow --id "$RUN_ID"
./sandbox/run.sh assert-state SUCCEEDED --id "$RUN_ID"
./sandbox/run.sh stop --id "$RUN_ID"
```

This turns “verify the workflow” into an executable instruction and lets the
same scenario run locally and in CI.

### Never hide a required failure

Commands such as these make an AI workflow appear healthier than it is:

```bash
npm test || true
npm run lint || true
```

If a suite is temporarily advisory, name it as advisory, emit a visible
annotation, assign an owner and expiry, and keep it out of claims such as “all
tests pass.” Required checks must propagate their exit status.

Likewise, validating that docs contain headings does not validate their facts.
Combine structural checks with path-based freshness checks and periodic manual
verification.

## 7. Make the command line the automation interface

AI agents are most reliable when important product and operational actions have
stable, composable commands. A UI-only workflow forces the agent into fragile
browser automation and gives CI little to assert.

For every important action:

- Expose a CLI command or API operation.
- Support non-interactive authentication.
- Provide `--json` output with a versionable schema.
- Use meaningful exit codes.
- Make destructive actions explicit and non-zero on cancellation or refusal.
- Have the CLI and UI call the same backend operation.

Example:

```bash
project jobs list --json
project subscription change pro --json
project data delete /datasets/demo --confirm
```

If the product has both CLI and UI, treat them as two clients of one contract.
When a bug is found in one client, review the equivalent flow in the other. A
strict one-to-one mapping may be excessive for some products, but parity at the
API and behavior level is highly transferable.

This also makes UAT precise: the agent can invoke a real user workflow, inspect
structured output, and save evidence.

## 8. Put AI review inside the pull-request boundary

AI review is most useful as additional coverage before human approval, not as a
replacement for deterministic CI or ownership.

### Use specialized independent reviewers

One giant “review this code” prompt produces inconsistent coverage. Run focused
reviewers in parallel, for example:

| Reviewer | Focus |
| --- | --- |
| General correctness | Logic, state transitions, error handling, API compatibility |
| Security | Auth boundaries, injection, secrets, tenant isolation, traversal |
| Performance | Query count, indexes, pagination, hot loops, platform limits |
| Tests and docs | Missing behavior tests and stale public/developer documentation |
| Cleanup | Dead paths, duplication, stale references, unnecessary scope |
| Domain specialist | Project-specific invariants such as billing or lifecycle safety |

Using more than one model family can reduce correlated blind spots, but model
diversity is not a substitute for prompt diversity and tests.

Each reviewer should receive:

- The PR diff and base/head SHAs.
- The relevant repository context files.
- A focused checklist with domain-specific risks.
- Permission to read the repository, but not to change code.
- A strict output schema.

Example output:

```markdown
## MUST FIX

- `src/auth.ts:81` — The new query does not constrain `organization_id`, so a
  valid user can retrieve another tenant's record. Include the authenticated
  organization in the key and add a cross-tenant test.

## SHOULD FIX

- `src/auth.ts:95` — Extract the duplicated mapping after the authorization fix.

## LOOKS GOOD

- Error responses preserve the existing public contract.
```

Require evidence: file and line, consequence, and a concrete fix or test. Tell
reviewers not to manufacture findings to fill every section.

### Use machine-readable verdicts

Do not infer a gate by searching free-form prose for the string `MUST FIX`.
Phrases such as “no MUST FIX issues” cause false positives, and a slightly
different heading causes false negatives.

Prefer a checked JSON artifact:

```json
{
  "reviewer": "security",
  "headSha": "abc123",
  "verdict": "changes_required",
  "findings": [
    {
      "severity": "must_fix",
      "path": "src/auth.ts",
      "line": 81,
      "summary": "Missing tenant constraint"
    }
  ]
}
```

Validate it against a schema and ensure every verdict refers to the current
head SHA.

### Re-review changed code, not merely the pull request

Skipping review after the first pass saves cost, but it can leave later code
unreviewed. Cache reviews by patch hash or reviewed head SHA:

- If the patch is identical after a rebase, reuse the verdict.
- If code changes, automatically review the new diff.
- Provide a human-controlled label to force a fresh full review.

Use an explicit fast path for low-risk changes, but define eligibility in
policy. “Fast” should skip expensive AI commentary, never type checks, required
tests, or human approval.

## 9. If an AI agent fixes reviews, make the pass narrow and bounded

An automated fix agent can remove mechanical review cycles, but it has more
authority than a reviewer and therefore needs tighter constraints.

A sound pattern is:

1. Wait for every reviewer to finish.
2. Aggregate all structured `must_fix` findings.
3. Give one fix agent the original goal, full diff, and all findings.
4. Permit edits only on the feature branch.
5. Require minimal fixes plus tests.
6. Commit under a distinct bot identity.
7. Re-run deterministic CI and review the new code.
8. Stop after one attempt and return unresolved findings to the author.

Explicitly prohibit:

- Changing the task's product or operational scope.
- Adding migrations, infrastructure changes, or destructive operations unless
  the original task authorized them.
- Editing workflow and branch-protection files to make itself pass.
- Deleting or weakening tests.
- Approving or merging the pull request.

The fix agent must receive the original constraints, not only reviewer
comments. A reviewer may propose a technically sensible change that the task
explicitly excluded.

### Give every state one writer

Do not let a local coding agent and a CI fix agent edit the same branch at the
same time. Define ownership:

```text
IMPLEMENTING       -> local agent owns the branch
REVIEWING          -> reviewers are read-only
AUTO_FIXING        -> CI fix agent owns the branch
CI                 -> all agents are read-only
HUMAN_DECISION     -> human approves, rejects, or requests work
```

This avoids merge conflicts and duplicated fixes. Concurrency groups should
cancel superseded review runs while preserving deployment operations that must
finish atomically.

## 10. Keep consequential authority human-owned

AI can propose and validate changes without owning irreversible decisions.
Reserve these for humans unless a much narrower policy explicitly delegates
them:

- PR approval and merge.
- Production migrations and destructive data repair.
- Enabling paid infrastructure or raising spend limits.
- Weakening security, test, or release gates.
- Rotating secrets or changing identity policy.
- Emergency bypasses.

Enforce this technically. A sentence in a prompt is not branch protection.

Practical controls include:

- Protected `main` with required checks and human review.
- A merge queue that tests the candidate rebased on current `main`.
- Environment protection for production.
- Separate review, fix, deploy, and production credentials.
- Bot identities that cannot approve their own changes.
- An audit trail linking issue, PR, review, build, deployment, and UAT.

The human gate is most valuable when the automation presents a compact evidence
bundle: goal, risk, changed contracts, test results, AI findings and resolutions,
staging result, and rollback plan.

## 11. Promote the exact candidate through environments

A robust AI workflow uses one main branch and environment promotion rather than
long-lived environment branches:

```text
feature branch
    -> PR checks
    -> human approval
    -> merge queue tests the integrated SHA
    -> staging deploy of that SHA
    -> staging E2E
    -> merge/promotion
    -> production deploy of the same built artifact or SHA
```

Important details:

- Test the merge candidate, not only the feature branch.
- Serialize writes to a shared staging environment.
- Record the commit SHA in health and deployment metadata.
- Reuse immutable artifacts instead of rebuilding different bits per environment.
- Block production when staging E2E fails.
- Keep rollback documented and executable.
- Upload logs and artifacts even when tests fail.

Distinguish E2E from UAT. E2E checks a stable scripted journey. UAT asks whether
the feature works as a real user expects. A good UAT report says what command was
run, which environment and build it used, which result was observed, and which
edge case was tried.

Risk-based escape hatches can be useful, but they should be rare, auditable, and
human-controlled. A “skip E2E” label should not silently skip unrelated local
integration tests or production safety checks.

## 12. Handle secrets and AI runners as a security boundary

An AI process running in CI is code execution. Treat prompts, diffs, repository
files, issue text, and review comments as untrusted inputs.

Minimum controls:

- Do not expose secrets to read-only reviewers unless strictly necessary.
- Give review jobs read-only repository permissions and a narrowly scoped way
  to publish their result.
- Give the fix agent write access only to the PR branch.
- Never run privileged AI jobs for pull requests from forks.
- Avoid executing PR-controlled scripts in a job that holds write tokens or
  secrets.
- Use ephemeral OIDC credentials where possible.
- Separate staging and production accounts.
- Pin third-party actions and AI CLI versions to reviewed immutable versions.
- Disable model training and response storage for proprietary source where the
  provider supports it.
- Sanitize logs and artifacts.
- Add time, token, and cost limits.

Avoid combining a full host sandbox, a repository write token, a model API key,
and staging credentials in the same general-purpose review process. If a job
must be powerful, isolate it, constrain its inputs and allowed paths, and put a
human gate after it.

Central secret management is preferable to repository-local `.env` files, but
“stored in a secret manager” does not itself provide least privilege. Each job
should fetch only the named values it needs.

## 13. Build maintenance into the workflow

Agent context decays as code changes. Maintenance needs both continuous and
periodic loops.

### On every relevant pull request

- Map changed paths to affected guides and user docs.
- Have a reviewer compare those docs with the code.
- Regenerate derived indexes and tool-specific context.
- Require an ADR for a new durable architectural choice.
- Verify public CLI/API examples still execute.

### Periodically

- Audit every guide against its source module.
- Search for stale branch names, commands, endpoints, and version numbers.
- Exercise runbooks and update `last_verified` only after verification.
- Remove or archive completed plans.
- Compare duplicate context trees byte-for-byte.
- Review workflow permissions, pinned dependencies, ignored failures, and costs.
- Sample AI review findings for false-positive and escape rates.

Use a separate model or human reviewer for context maintenance when practical,
but do not assume repeated model passes guarantee truth. Source inspection and
executable examples remain the evidence.

## 14. Measure the workflow

Without measurements, adding agents can increase cost and ceremony without
improving outcomes. Track:

- PR lead time and time spent waiting for AI review.
- AI review cost per PR.
- Percentage of findings accepted, rejected, or already covered by tests.
- False-negative escapes found after merge.
- Fix-agent success rate and revert rate.
- CI flake rate.
- Re-review frequency after new commits.
- Staging and UAT failure rates.
- Documentation freshness and broken-command rate.
- Number and age of advisory or ignored checks.

Optimize for escaped defects and feedback latency, not number of AI comments.
Seven reviewers that repeat each other are worse than three reviewers with
clear, non-overlapping contracts.

## 15. A practical adoption sequence

Do not begin with a fleet of AI reviewers. Build the feedback and authority
foundation first.

### Day 1: make one agent safe and useful

1. Add the root operating contract.
2. Add `make check` and ensure it fails correctly.
3. Protect `main`; require a PR and human approval.
4. Document the secrets and destructive-action policy.
5. Add a PR definition-of-done checklist.

### Week 1: improve context and feedback

1. Add a system map and guides for the two busiest modules.
2. Create a one-command local integration environment.
3. Add ADR and runbook templates.
4. Require plans with acceptance criteria for large changes.
5. Add machine-readable CLI output for important workflows.

### When PR volume justifies it

1. Add one general AI reviewer and measure it.
2. Add security or domain specialists where defects actually occur.
3. Produce structured, SHA-bound review results.
4. Add a bounded fix agent only after branch ownership is defined.
5. Add a merge queue and staging promotion.

### When autonomous multi-step work justifies it

1. Add a task ledger with executable acceptance criteria.
2. Add an append-only progress log.
3. Insert human and deployment gates between phases.
4. Derive completion from evidence rather than agent self-report.

## 16. Recommended project skeleton

```text
AGENTS.md
Makefile
.github/
  workflows/
    pr-checks.yml
    ai-review.yml
    merge-queue.yml
    deploy-staging.yml
    deploy-production.yml
docs/
  agent/
    repository.md
    api.md
    web.md
    deployment.md
  architecture/
  adr/
    README.md
    template.md
  plans/
  runbooks/
    template.md
  archive/
scripts/
  check-agent-context.sh
  generate-agent-context.sh
  sandbox/
    run.sh
```

The names are unimportant. The separation of concerns is the point.

## 17. Definition of done for AI-authored changes

Use a checklist that requires evidence:

- [ ] The current behavior and relevant module context were inspected.
- [ ] The change stayed within the issue's stated scope and non-goals.
- [ ] Public contracts, migrations, and compatibility effects are identified.
- [ ] Required format, lint, type, unit, and integration checks pass without
      ignored failures.
- [ ] Tests cover the changed behavior and important failure path.
- [ ] User docs, module guides, ADRs, and runbooks are updated where needed.
- [ ] No secrets, generated credentials, or local environment files were added.
- [ ] AI review findings refer to the current head SHA and are resolved or
      explicitly rejected with a reason.
- [ ] A human approved the consequential change.
- [ ] The exact candidate passed staging E2E.
- [ ] A real-user UAT scenario passed and its evidence is recorded.
- [ ] Deployment and rollback are observable.

## 18. Practices to adopt, adapt, and avoid

### Adopt

- Versioned root rules with explicit human-only authority.
- Small, routed domain guides with invariants and verification commands.
- Plans with non-goals and executable acceptance criteria.
- A hermetic local sandbox used again in CI.
- Specialized parallel review with structured severity.
- A bounded fix pass and single-writer branch ownership.
- Human approval, merge-queue validation, staged promotion, and UAT.
- Documentation maintenance tied to changed interfaces.

### Adapt to your risk and scale

- Number and choice of AI reviewers.
- Whether UI-to-CLI parity is strict or only shares API behavior.
- Fast and no-E2E modes.
- Automatic production promotion.
- Autonomous task ledgers for long initiatives.

### Avoid

- Multiple hand-maintained copies of agent context.
- Free-form prose as a machine gate.
- Skipping review merely because a PR was reviewed once.
- `|| true` on checks described as required.
- Latest-version AI CLIs installed in release-critical CI.
- Privileged AI runners with broad credentials and untrusted inputs.
- Allowing a fix agent to expand product or operational scope.
- Treating document structure or a `last_verified` date as proof of accuracy.
- Letting an agent approve, merge, deploy, and verify its own change.

## Appendix A: What was observed in the source repository

This guide was derived by inspecting the repository at commit `06406a3`
(2026-08-01). The most important source artifacts were:

- `CLAUDE.md`: root workflow, authority rules, context routing, and definition
  of done.
- `.claude/skills/`: 16 repository and domain context packages, including
  context-maintenance workflows.
- `.codex/skills/`: a second assistant-specific context tree.
- `.github/workflows/claude-review.yml`: seven parallel reviewers, a Codex fix
  agent, deterministic checks, and mode labels.
- `.github/workflows/merge-queue.yml`: retesting of the integrated candidate,
  staging deployment, and staging E2E.
- `.github/workflows/deploy-staging.yml` and `deploy-prod.yml`: environment
  promotion and production E2E.
- `sandbox/`: a local end-to-end environment used by agents and CI.
- `docs/adr/`, `docs/plans/`, `docs/runbooks/`, and module READMEs: layered
  repository memory.
- `scripts/ralph/prd.json` and `scripts/ralph/progress.txt`: an example task
  ledger and cross-session progress memory for a multi-phase initiative.
- `scripts/docs/`: context generation, structural validation, and an older
  AI-based documentation updater.
- Git history: repeated changes prompted by real workflow failures, including
  agent edit races, bot approvals, excessive review loops, fix-agent timeouts,
  merge-queue behavior, and an unauthorized migration created by a fix agent.

### Strengths found

- The repository encodes workflow policy instead of relying on oral knowledge.
- It routes agents to detailed domain context and verification commands.
- It combines two model families and several review specialisms.
- Review agents are implemented to comment rather than approve; policy reserves
  merge authority for a human.
- The fix agent is bounded to one attempt and has a distinct identity.
- A merge queue tests the candidate with current `main` before staging E2E.
- The local sandbox and real-environment UAT create multiple feedback layers.
- Plans, ADRs, runbooks, and progress logs preserve unusually rich reasoning.

### Gaps that informed the cautions in this guide

- The root docs say there are eight parallel reviews, while the workflow
  currently defines seven.
- One root summary says to enable auto-merge, while its hard rules and the
  current workflow reserve merging for a human.
- A code-review runbook still describes obsolete `develop`/`stage`/`prod`
  branches, approvals, auto-merge, and an older review architecture.
- A branch-protection setup script also configures obsolete branches and check
  names.
- The Claude context tree contains 16 skills and the Codex tree 12; several
  shared files differ, and the mandatory workflow skill is not mirrored.
- Some TypeScript tests and lints are made non-blocking with `|| true`, both in
  local commands and CI.
- The AI-review workflow grants broad top-level permissions and runs Codex with
  a full host sandbox while model and GitHub credentials are present.
- AI CLIs are installed without a pinned version during each run.
- Review verdicts are detected by searching comments for `MUST FIX` rather than
  consuming a validated structured result.
- After an initial review, later pushes skip AI review unless a label requests
  it, even if the patch changed materially.
- The `noe2e` label skips more than the root workflow summary implies.
- Manual staging UAT is mandatory in the written workflow, but automated
  production promotion has already started by the time it is performed, so UAT
  is verification rather than a pre-production release gate.
- Documentation validation checks shape, not factual freshness, and is not a
  required part of the main CI path.
- An automated docs updater remains in the tree but is not wired into current
  workflows and still contains assumptions from an older branch model.

These gaps do not erase the value of the design. They demonstrate a final
principle: an AI workflow is production software. It needs tests, security
review, observability, ownership, migration, and maintenance just like the
product it changes.
