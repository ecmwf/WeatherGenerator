# Agentic setup

How LLM coding tools get their instructions in this repo.

## Single source of truth

`AGENTS.md` is the canonical instruction file — at the repo root and, for scoped rules,
inside `src/`, `packages/`, `config/`, `tests/`. All providers resolve to these files,
so there is one file per scope to edit and the copies can't drift.

| Provider | Mechanism |
|---|---|
| OpenAI Codex | reads `AGENTS.md` natively, root + nested |
| Claude Code | `CLAUDE.md → AGENTS.md` symlink in each scope |
| Gemini CLI | `.gemini/settings.json` sets the context file to `AGENTS.md` (root + nested); legacy clients fall back to the root `GEMINI.md` symlink |
| GitHub Copilot | reads root `AGENTS.md` natively (coding agent also nested); `.github/copilot-instructions.md → AGENTS.md` symlink as fallback |
| Cursor | `.cursor/rules/main.mdc` stub pointing at `/AGENTS.md` (needs its own `.mdc` frontmatter format, so no symlink) |

## What goes where

The common agent task — assess or change a code section — has three phases that fail
differently: routing (find the right files; failure = wasted tokens), understanding
(grasp enough to change it correctly; failure = a plausible wrong edit), and
verification (run the right checks). Effectiveness is decided in the understanding
phase, by knowledge the code itself cannot state. Documentation priority: effectiveness
first; token efficiency second, and only where it doesn't compromise effectiveness.

**AGENTS.md — auto-loaded. Route, rule, verify. Nothing else.**

- Orientation map: enough for the first search to land right.
- Hard rules and invariants that apply to most edits in the scope.
- Verification commands.
- Pointers to `agent_docs/` with a trigger phrased in task vocabulary, bidirectional
  where the doc tracks code: "read before touching X; update after changing X". The
  update half is what keeps central docs from going stale — the agent editing code in
  that scope has the reminder in context. Always reference docs by root-relative path
  (`agent_docs/<name>.md`); CI checks these links.
- Inclusion test per line: would an agent lacking it take a wrong turn or make a wrong
  edit? No → cut it, or demote it to `agent_docs/`.
- Put each rule in the narrowest `AGENTS.md` that covers it (nested files load only
  when an agent works in that subtree), and state each fact at exactly one level —
  ancestor files load together, so duplication is paid twice.

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
- Every doc must be linked from the nearest `AGENTS.md`; an unlinked doc isn't
  discovered reliably.
- All agent docs live in top-level `agent_docs/` — the highest-value content is
  cross-cutting and has no home directory, and one tree keeps grep-fallback discovery
  reliable. Scoping comes from which `AGENTS.md` points to a doc, not from its location
  (one file, N pointers — this is also what avoids repeating shared content per scope).
  Central files never move when coupling grows, so pointers stay valid. Flat until it
  hurts; then group by subsystem or task, never by directory. Name files in task
  vocabulary (`config-system.md`, `masking.md`) — filenames are matched by searches.
  Exceptions: human-facing reference stays in `docs/` (e.g.
  `docs/evaluate_config_reference.md`), package `README.md`s stay in their package, and
  a quasi-independent package (own lockfile, deployable on its own) may keep real docs
  in-package so they travel if it is extracted.
- Scope-local rules and coupling one-liners are not docs: they go in the narrowest
  `AGENTS.md` that covers all files involved.

**The pointer/content asymmetry:** a pointer costs ~15 tokens, a missed invariant costs
a wrong edit. When in doubt, the pointer goes in AGENTS.md and the content goes in
`agent_docs/`. Be generous with pointers, strict with content.

## Editing rules

- Edit `AGENTS.md` files only; symlinks and settings propagate automatically.
- Facts only — document what the setup is, not what it should be; agents take
  statements literally. Mark unverified claims as such or leave them out.
- Plain imperatives over formatting: bold and callouts don't raise compliance. Keep
  the *why* on non-obvious rules ("never X — it breaks Y") so agents know when a rule
  generalizes.
- Personal instructions never go in the tracked provider files. Use `CLAUDE.local.md`
  (repo-specific, gitignored, loaded in addition to the project file) or
  `~/.claude/CLAUDE.md` (user-global) — both compose with the repo instructions
  instead of replacing them.

## Adding a new scope

From the repo root:

```bash
$EDITOR <dir>/AGENTS.md          # scoped rules, keep terse
ln -s AGENTS.md <dir>/CLAUDE.md  # Claude Code
```

Codex, Copilot's coding agent, and Gemini (via the setting) pick up the nested
`AGENTS.md` without further plumbing. Root symlinks, if ever lost:

```bash
ln -s AGENTS.md CLAUDE.md
ln -s AGENTS.md GEMINI.md
ln -s ../AGENTS.md .github/copilot-instructions.md
```

## Enforcement

`scripts/actions.sh check-agent-docs` (run in CI, linting job) asserts:

- root provider files are symlinks resolving to `AGENTS.md`;
- the Cursor stub and `.gemini/settings.json` reference `AGENTS.md`;
- every nested `AGENTS.md` has a sibling `CLAUDE.md` symlink resolving to it;
- every `agent_docs/...` or `docs/...` path referenced in any `AGENTS.md` exists (no
  dangling pointers);
- every file in `agent_docs/` and `docs/` is referenced by at least one `AGENTS.md`
  (no orphan docs).

Needed because checkouts without symlink support turn symlinks into plain text files,
and tools sometimes replace a symlink with a regular file on write — either silently
breaks single-source-of-truth.
