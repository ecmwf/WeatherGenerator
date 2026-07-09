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

## Editing rules

- Edit `AGENTS.md` files only; symlinks and settings propagate automatically.
- Root `AGENTS.md` is auto-loaded into every session for every tool — every token costs
  context budget. Keep it to rules, a layout map, and the doc index.
- Scoped rules go in the narrowest `AGENTS.md` that covers them; they load only when an
  agent works in that subtree.
- Reference docs go in `docs/`, linked from the nearest AGENTS.md with a one-line
  "read when..." hook. An unlinked doc isn't discovered reliably.

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
- every nested `AGENTS.md` has a sibling `CLAUDE.md` symlink resolving to it.

Needed because checkouts without symlink support turn symlinks into plain text files,
and tools sometimes replace a symlink with a regular file on write — either silently
breaks single-source-of-truth.
