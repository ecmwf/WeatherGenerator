# Recipe: add or change agent documentation

Use when: creating a doc under `agent_docs/`, or renaming/moving/removing one.

1. Pick the genre (taxonomy in `agent_docs/agentic-setup.md`): systems doc (a runtime
   dataflow) → `agent_docs/<topic>.md`; procedure → `agent_docs/recipes/`; rationale →
   `agent_docs/decisions/`.
2. Name the file in task vocabulary (`config-system.md`, not `common-package.md`) —
   filenames are what searches match.
3. Follow the genre template. Systems: ~5-line summary first, H2 per stage with
   `file:symbol` anchors, "Coupling & invariants" section last. Recipes: trigger line,
   numbered steps, verification, pitfalls. Decisions: context → decision → consequences.
4. Wire the pointers — both are required, a doc without them is invisible to agents:
   - the narrowest `AGENTS.md` covering everything the doc spans, phrased
     bidirectionally: "read before touching X; update after changing X";
   - the documentation index in the root `AGENTS.md`.
5. Reference docs by root-relative path (`agent_docs/<name>.md`).
6. Verify: `./scripts/actions.sh check-agent-docs` (also runs in CI) — catches
   dangling references, orphan docs, and agent docs missing from the root index.
   On rename/move/delete, additionally grep the old path across `agent_docs/` itself:
   cross-references between docs are not machine-checked.

Pitfalls:

- Updating a doc's content without updating the AGENTS.md trigger line (or vice versa).
- Restating systems content in a recipe or decision — link to the systems doc instead;
  each fact lives in exactly one place.
- Adding a doc only to a nested AGENTS.md: tools that don't auto-load nested files
  will never find it — the root index is the universal fallback.
