# WeatherGenerator Skills Overview

This directory contains reusable procedures and workflows for the WeatherGenerator project.

## How to Use Skills

When working on a task, check this overview first to find the relevant skill:
- **Planning & Design** → Use `planning` skill
- **Code Implementation** → Use `implementation` skill  
- **Testing** → Use `testing` skill
- **HPC Deployment** → Use `hpc-deployment` skill
- **Metrics & Logging** → Use `metrics` skill

Each skill contains:
- When to use it
- Step-by-step procedures
- Code examples and templates
- Common pitfalls

## Available Skills

| Skill | Description | When to Use |
|-------|-------------|-------------|
| [`planning`](skills/planning/SKILL.md) | Create implementation plans with bite-sized tasks | Before any multi-step feature |
| [`implementation`](skills/implementation/SKILL.md) | Code implementation guidelines and patterns | During feature development |
| [`testing`](skills/testing/SKILL.md) | Test writing and verification procedures | When adding tests or debugging |
| [`hpc-deployment`](skills/hpc-deployment/SKILL.md) | HPC cluster deployment workflows | When deploying to HPC systems |
| [`metrics`](skills/metrics/SKILL.md) | Metrics logging and MLflow integration | When adding new metrics or logging |

## Task Tracking

Active tasks are tracked in the `tasks/` directory:
- Each task has its own folder
- Contains step-by-step documentation as work progresses
- Links to relevant skills
- Final implementation notes

See `tasks/README.md` for task management guidelines.

## User Documentation

User-facing documentation is in `docs/`:
- How to use implemented features
- API references
- Configuration guides

Only create docs when a feature is complete and stable.

## Adding New Skills

1. Create `skills/<skill-name>/SKILL.md`
2. Add entry to this overview table
3. Include: when to use, steps, examples, pitfalls
4. Keep skills focused on task types, not specific features

## Best Practices

- **Skills = task types** (e.g., "planning", not "timing-metrics")
- **Tasks = specific work** (e.g., "add timing metrics to training")
- **Docs = user-facing** (e.g., "how to use timing metrics")
- Update skills when you discover better approaches
- Keep task docs iterative and concise
