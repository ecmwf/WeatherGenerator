# .hermes Directory

This directory contains agent-specific files for the WeatherGenerator project.

## Structure

```
.herms/
├── README.md              # Skills overview and usage guide
├── skills/                # Reusable procedures and workflows
│   ├── planning/
│   │   └── SKILL.md       # Task planning and breakdown
│   └── metrics/
│       └── SKILL.md       # Metrics and logging patterns
├── tasks/                 # Active task tracking
│   ├── README.md          # Task management guidelines
│   └── 2026-04-17-timing-metrics/
│       └── README.md      # Timing metrics task documentation
└── docs/                  # User-facing documentation (when features complete)
    └── README.md          # Documentation template
```

## Purpose

- **Skills**: Task-type procedures (planning, implementation, metrics, etc.)
- **Tasks**: Specific work items with step-by-step documentation
- **Docs**: User-facing feature documentation

## Usage

### For Hermes Agent

1. Check `README.md` for skills overview
2. Load relevant skill before starting task
3. Create task folder for active work
4. Document progress in step files
5. Move completed work to `docs/` if user-facing

### For Humans

1. Read `README.md` to understand project workflows
2. Check `tasks/` for active work status
3. Review `docs/` for completed feature documentation
4. Use skills as reference for best practices

## Git Ignore

Add to `.gitignore`:
```
# Agent-specific files
.herms/
```

Or keep tracked if team collaboration on procedures is desired:
```
# Keep skills and tasks, ignore temporary agent state
.herms/tasks/*/temp/
```

## Best Practices

- **Skills**: Update when discovering better approaches
- **Tasks**: Keep step files concise and iterative
- **Docs**: Only create for stable, user-facing features
- **Naming**: Use `YYYY-MM-DD-description` for task folders
