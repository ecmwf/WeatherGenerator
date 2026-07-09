# Agentic setup

How LLM coding tools get their instructions in this repo.

## Single source of truth, loaded by opt-in

`AGENT-README.md` at the repo root is the canonical instruction file, and it is
**opt-in**: the repo tracks no `CLAUDE.md`/`AGENTS.md`/`GEMINI.md`, so no LLM tool
loads context from this repo by default. Each user points their personal, untracked
instruction file at it — for Claude Code, an `@AGENT-README.md` import line in
`CLAUDE.local.md` (repo-specific, gitignored) or in `~/.claude/CLAUDE.md`; other
tools use their equivalent personal-context mechanism. This keeps one tracked file
to edit (no per-provider copies to drift) while letting users combine it with
personal instructions or opt out entirely.

Consequence: nothing here is auto-loaded for every agent. Write `AGENT-README.md`
assuming it is the only instruction file in context, and keep `agent_docs/`
discoverable from it by explicit root-relative pointers.

Planned, not yet implemented: scoped instruction files inside `src/`, `packages/`,
`config/`, `tests/` for directory-local rules. Until they exist, scope-local rules
live in `AGENT-README.md` or the relevant `agent_docs/` doc.

## What goes where

The common agent task — assess or change a code section — has three phases that fail
differently: routing (find the right files; failure = wasted tokens), understanding
(grasp enough to change it correctly; failure = a plausible wrong edit), and
verification (run the right checks). Effectiveness is decided in the understanding
phase, by knowledge the code itself cannot state. Documentation priority: effectiveness
first; token efficiency second, and only where it doesn't compromise effectiveness.

**AGENT-README.md — in context every session (once opted in). Route, rule, verify.
Nothing else.**

- Orientation map: enough for the first search to land right.
- Hard rules and invariants that apply to most edits.
- Verification commands.
- Pointers to `agent_docs/` with a trigger phrased in task vocabulary, bidirectional
  where the doc tracks code: "read before touching X; update after changing X". The
  update half is what keeps central docs from going stale — the agent editing code
  has the reminder in context. Always reference docs by root-relative path
  (`agent_docs/<name>.md`).
- Inclusion test per line: would an agent lacking it take a wrong turn or make a wrong
  edit? No → cut it, or demote it to `agent_docs/`.
- State each fact in exactly one place — `AGENT-README.md` and the docs load
  together, so duplication is paid twice. Pointers are the one exception: redundant
  pointers are cheap and aid discovery; facts are not exempt.

**agent_docs/ — loaded on demand. Everything a correct edit of a given kind needs.**

- Three genres, split by the question they answer; no fact belongs to two:
  - `agent_docs/*.md` — systems: what happens at runtime, one doc per dataflow (not per
    directory). Template: ~5-line summary, H2 per stage with `file:symbol` anchors
    and data shapes at boundaries, then a "Coupling & invariants" section ("change
    X → check Y") — that section is the payload.
  - `agent_docs/recipes/` — procedures: how to make a change of kind X. Trigger line,
    numbered steps with file anchors, verification command, pitfalls.
  - `agent_docs/decisions/` — rationale: why something is deliberately this way
    (context → decision → consequences). Prevents "fixing" intentional choices.
  - Mechanism goes in systems, procedure in recipes, rationale in decisions; the
    latter two link into systems docs instead of restating them.

- Document the delta between what the code says and what is true: dataflow across
  files, coupling ("if you change X, also update Y"), invariants, which of two
  similar mechanisms is canonical, why something is deliberately unusual. A file
  inventory is grep-replaceable — low value; coupling and rationale are not — high value.
- Structure for partial reads: conclusion/summary first, clear H2 sections, `file:symbol`
  anchors. Agents often read only the top of a file.
- Every doc must be listed in the `AGENT-README.md` documentation index; an unlisted
  doc isn't discovered reliably. Procedure: `agent_docs/recipes/add-documentation.md`.
- All agent docs live in top-level `agent_docs/` — the highest-value content is
  cross-cutting and has no home directory, and one tree keeps grep-fallback discovery
  reliable. Scoping comes from the pointer's trigger phrase, not from the doc's
  location (one file, N pointers — this is also what avoids repeating shared content).
  Central files never move when coupling grows, so pointers stay valid. Flat until it
  hurts; then group by subsystem or task, never by directory. Name files in task
  vocabulary (`config-system.md`, `masking.md`) — filenames are matched by searches.
  Exceptions: human-facing reference stays in `docs/` (e.g.
  `docs/evaluate_config_reference.md`), directory-scoped reference stays next to the
  code as `DOCS-*.md` (next block), package `README.md`s stay in their package, and a
  quasi-independent package (own lockfile, deployable on its own) may keep real docs
  in-package so they travel if it is extracted.
- Scope-local rules and coupling one-liners are not docs: they go in
  `AGENT-README.md` (or, once scoped instruction files exist, the narrowest one that
  covers all files involved).

**DOCS-*.md — directory-scoped reference, living next to the code it describes.**

- One per subsystem directory (`config/`, `config/streams/`,
  `src/weathergen/{model,datasets,train}/`, `packages/`): file-by-file inventories,
  class/function detail with line anchors, schemas, option lists — what is in that
  directory and how its scripts function. This is deliberately the grep-replaceable
  layer: faster than reading the code, but derivable from it.
- The split against `agent_docs/` is scope-by-task vs scope-by-directory, not level
  of detail (both are detailed): a fact that spans directories — a runtime dataflow,
  coupling ("change X → also update Y"), a workflow, a design choice — goes in
  `agent_docs/`; detail local to one directory's files goes in its `DOCS-*.md`. When
  both need a fact, one states it and the other links to it.

**The pointer/content asymmetry:** a pointer costs ~15 tokens, a missed invariant costs
a wrong edit. When in doubt, the pointer goes in `AGENT-README.md` and the content
goes in `agent_docs/`. Be generous with pointers, strict with content.

## Editing rules

- Context maintenance is part of every change, not a documentation task: if an edit
  alters behavior described in a systems doc, a `DOCS-*.md` reference file, a coupling
  line, or an `AGENT-README.md` rule/inventory, update that description in the same
  commit.
  Agents take statements literally, so stale context turns into wrong edits later.
  The bidirectional pointer triggers ("update after changing X") mark which docs a
  change touches; keeping descriptions true is on the author.
- Facts only — document what the setup is, not what it should be; agents take
  statements literally. Mark unverified claims as such or leave them out.
- Plain imperatives over formatting: bold and callouts don't raise compliance. Keep
  the *why* on non-obvious rules ("never X — it breaks Y") so agents know when a rule
  generalizes.
- Personal instructions never go in the tracked files. Use `CLAUDE.local.md`
  (repo-specific, gitignored) or `~/.claude/CLAUDE.md` (user-global) — the same files
  that carry the opt-in pointer to `AGENT-README.md`.

## Enforcement

Not yet implemented: a `check-agent-docs` CI check asserting that every
`agent_docs/...`/`docs/...` path referenced in `AGENT-README.md` exists and that
every `agent_docs/` file is listed in its documentation index. Until it exists,
verify links manually (grep) when adding, renaming, or removing docs.

When adding or changing docs, follow `agent_docs/recipes/add-documentation.md`.
