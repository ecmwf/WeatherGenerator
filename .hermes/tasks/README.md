# Task Management Guidelines

## Structure

Each task gets its own folder: `tasks/<YYYY-MM-DD-task-name>/`

### Folder Contents

```
tasks/2026-04-17-timing-metrics/
├── README.md          # Task overview and current status
├── step-01-analysis.md    # Initial codebase exploration
├── step-02-design.md      # Design decisions
├── step-03-implementation.md  # Code changes
├── step-04-testing.md     # Verification steps
└── step-05-completion.md  # Final summary and docs
```

## README.md Template

```markdown
# [Task Name]

**Status:** [Active | In Progress | Completed | Blocked]

**Created:** YYYY-MM-DD

**Related Skills:** [skill1, skill2]

## Goal

[One sentence describing what we're building]

## Progress

- [x] Step 1: Analysis
- [ ] Step 2: Design
- [ ] Step 3: Implementation
- [ ] Step 4: Testing
- [ ] Step 5: Documentation

## Current Blockers

[Any issues or decisions needed]

## Links

- [Implementation PR](link)
- [Design Doc](link)
- [Related Issues](link)
```

## Step Files

Each step file documents:
1. **What was done** (1-2 sentence summary)
2. **Key decisions** (why we chose this approach)
3. **Code changes** (file paths, line numbers)
4. **Next steps** (what comes next)

Keep step files concise. Update iteratively as work progresses.

## Completion

When task is complete:
1. Mark README status as "Completed"
2. Add final step file with summary
3. Move user-facing docs to `docs/` if applicable
4. Update relevant skills if new patterns discovered

## Active Tasks

- `2026-04-17-timing-metrics` - Add timing metrics to training pipeline
