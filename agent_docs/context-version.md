# Context version — verified-against stamp

The `agent_docs/` tree and all AGENTS.md files describe the code as it existed at the
state below. Code merged after this stamp may have drifted. Where docs and code
disagree, trust the code — then update the docs and this stamp (procedure:
`agent_docs/recipes/add-documentation.md`).

Because development happens on forks, a commit alone is ambiguous: the identification
is always (repository, commit).

Verified against:

- repository: `github.com/florianscheidl/WeatherGenerator` (fork of
  `github.com/ecmwf/WeatherGenerator`)
- branch: `dev/ai-context-and-documentation`
- commit: `6af5a9f4f724be4895ba359c89147d114acd1070`
- date: 2026-07-09

## Checking for drift

Before relying on a doc's coupling/invariant claims for a non-trivial edit — or whenever
doc and code seem to disagree — diff the code against the stamp, scoped to the paths the
doc describes (every doc anchors its claims to files):

```bash
git cat-file -e <commit> 2>/dev/null || git fetch <repository> <commit>  # stamp commit present in this clone?
git diff --stat <commit>..HEAD -- src/weathergen/train/    # did the described area change?
git log --oneline <commit>..HEAD -- <anchored-file>        # what changed, per file
```

- Empty diff for the doc's paths → the doc is current for your purpose, regardless of
  the stamp's age.
- Non-empty diff → read the changed code; it wins over the doc. Update the doc and
  this stamp with what you learn.
- Commit unavailable even after fetching (different fork, shallow clone) → skip the
  diff and verify claims directly against the code.

## Bumping

Bump this stamp (repository, branch, commit, date) whenever the docs are re-verified
against the code — at minimum after every doc-updating change, per the context-sync
rule in the root AGENTS.md.
