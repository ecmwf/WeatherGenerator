# WeatherGenerator Agent Structure

Created: 2026-04-17

## Overview

This document describes the `.hermes/` directory structure for managing agent workflows, skills, and task tracking in the WeatherGenerator project.

## Structure

```
.herms/
├── README.md              # Skills overview - start here
├── skills/                # Reusable task-type procedures
│   ├── planning/
│   │   └── SKILL.md       # How to break down features into tasks
│   └── metrics/
│       └── SKILL.md       # How to add metrics and logging
├── tasks/                 # Active task documentation
│   ├── README.md          # Task management guidelines
│   └── 2026-04-17-timing-metrics/
│       └── README.md      # Timing metrics implementation
└── docs/                  # User-facing documentation
    └── README.md          # Documentation template
```

## Philosophy

### Skills = Task Types
Skills describe **how** to do a type of task, not specific features:
- ✅ `planning` - How to plan any feature
- ✅ `metrics` - How to add metrics
- ❌ `timing-metrics` - Too specific (this is a task, not a skill type)

### Tasks = Specific Work
Tasks track **what** we're building right now:
- ✅ `2026-04-17-timing-metrics` - Add timing metrics to training
- ✅ `2026-04-18-auth-system` - Implement authentication

### Docs = User-Facing
Docs explain **how users** use completed features:
- ✅ "How to configure timing metrics"
- ❌ "How we implemented timing metrics" (this goes in task docs)

## Workflow

### Starting a New Feature

1. **Check skills overview** (`.hermes/README.md`)
2. **Load relevant skill** (e.g., `planning`)
3. **Create task folder** (`.hermes/tasks/YYYY-MM-DD-feature/`)
4. **Write plan** (step-by-step tasks)
5. **Implement task-by-task** (document each step)
6. **Commit frequently** (after each task)
7. **Move to docs** (if user-facing feature)

### Example: Adding Timing Metrics

```bash
# 1. Check skills overview
cat .hermes/README.md

# 2. Load metrics skill
# (Hermes agent auto-detects or manually load)

# 3. Create task folder
mkdir -p .hermes/tasks/2026-04-17-timing-metrics

# 4. Write plan in README.md
# (see .hermes/tasks/README.md for template)

# 5. Implement and document
# - step-01-analysis.md
# - step-02-design.md
# - step-03-implementation.md
# - step-04-testing.md
# - step-05-completion.md

# 6. Commit
git add .hermes/tasks/2026-04-17-timing-metrics/
git commit -m "docs: add timing metrics task documentation"
```

## Best Practices

### Skills
- Focus on **patterns**, not specific features
- Include: when to use, steps, examples, pitfalls
- Update when discovering better approaches
- Keep concise (2-4 pages max)

### Tasks
- One folder per feature/task
- Document iteratively as work progresses
- Link to relevant skills
- Keep step files focused (one action per step)

### Docs
- Only for stable, user-facing features
- Explain **how to use**, not **how we built**
- Include examples and common use cases
- Keep updated as features evolve

## Git Integration

`.gitignore` includes commented entry for `.hermes/`:
```
# Agent-specific files (optional - uncomment if you want to ignore)
# .hermes/
```

**Keep tracked if:**
- Team collaboration on procedures
- Skills evolve over time
- Task history is valuable

**Ignore if:**
- Agent-specific temporary files
- Personal workflow notes
- Not needed for project reproducibility

## Current Skills

| Skill | Purpose | When to Use |
|-------|---------|-------------|
| `planning` | Break down features into tasks | Before any multi-step work |
| `metrics` | Add metrics and logging | When tracking performance |

## Current Tasks

| Task | Status | Description |
|------|--------|-------------|
| `2026-04-17-timing-metrics` | ✅ Completed | Add timing metrics to training pipeline |

## Future Enhancements

Potential new skills:
- `implementation` - Code patterns and guidelines
- `testing` - Test writing and verification
- `hpc-deployment` - HPC cluster workflows
- `debugging` - Systematic debugging approaches

Potential new docs:
- Timing metrics user guide
- Configuration reference
- HPC deployment guide

## Maintenance

### Update Skills When:
- Discover better approaches
- Fix missing steps
- Add new pitfalls
- Update examples

### Archive Completed Tasks:
- Move to `tasks/archive/` if not needed
- Keep recent tasks for reference
- Delete old temp files

### Keep Docs Current:
- Update when features change
- Remove deprecated sections
- Add new use cases

---

**Created by:** WeatherGenerator Agent  
**Last Updated:** 2026-04-17
