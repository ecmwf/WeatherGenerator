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
4. Wire the pointer — a doc without it is invisible to agents: add a line to the
   documentation index in `AGENT-README.md`, phrased bidirectionally: "read before
   touching X; update after changing X".
5. Reference docs by root-relative path (`agent_docs/<name>.md`).
6. Verify manually (no CI check yet — see `agent_docs/agentic-setup.md`,
   Enforcement): every path the doc references exists, and on rename/move/delete
   grep the old path across `AGENT-README.md` and `agent_docs/` — cross-references
   are not machine-checked.

Pitfalls:

- Updating a doc's content without updating its `AGENT-README.md` trigger line (or
  vice versa).
- Restating systems content in a recipe or decision — link to the systems doc instead;
  each fact lives in exactly one place.
