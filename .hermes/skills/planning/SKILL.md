# Planning Skill

Use this skill when breaking down requirements into implementable tasks.

## When to Use

- Before implementing any multi-step feature
- When requirements are unclear or complex
- Before delegating to subagents
- When you need to document your approach

## Process

### 1. Understand Requirements

Read and understand:
- Feature requirements
- Design documents
- Acceptance criteria
- Constraints

### 2. Explore Codebase

Use Hermes tools:
```python
# Understand structure
search_files("*.py", target="files", path="src/")

# Find similar patterns
search_files("similar_pattern", path="src/", file_glob="*.py")

# Read key files
read_file("src/main.py")
```

### 3. Design Approach

Decide:
- Architecture pattern
- File organization
- Dependencies needed
- Testing strategy

### 4. Create Bite-Sized Tasks

**Each task = 2-5 minutes of focused work.**

Every step is one action:
- "Write the failing test"
- "Run it to make sure it fails"
- "Implement minimal code"
- "Run tests and verify pass"
- "Commit"

**Too big:**
```markdown
### Task 1: Build authentication system
[50 lines across 5 files]
```

**Right size:**
```markdown
### Task 1: Create User model with email field
[10 lines, 1 file]

### Task 2: Add password hash field
[8 lines, 1 file]
```

### 5. Document Task Structure

Each task should include:

```markdown
### Task N: [Descriptive Name]

**Objective:** What this accomplishes (one sentence)

**Files:**
- Create: `exact/path/to/new_file.py`
- Modify: `exact/path/to/existing.py:45-67`
- Test: `tests/path/to/test_file.py`

**Step 1: Write failing test**
```python
def test_specific_behavior():
    result = function(input)
    assert result == expected
```

**Step 2: Run test to verify failure**
Run: `pytest tests/path/test.py::test_specific_behavior -v`
Expected: FAIL

**Step 3: Write minimal implementation**
```python
def function(input):
    return expected
```

**Step 4: Run test to verify pass**
Run: `pytest tests/path/test.py::test_specific_behavior -v`
Expected: PASS

**Step 5: Commit**
```bash
git add tests/path/test.py src/path/file.py
git commit -m "feat: add specific feature"
```
```

### 6. Save Task Plan

Create task folder: `tasks/YYYY-MM-DD-feature-name/`
- Save plan as `README.md`
- Create step files as work progresses

## Principles

### DRY (Don't Repeat Yourself)
Extract common patterns, don't copy-paste.

### YAGNI (You Aren't Gonna Need It)
Implement only what's needed now, not "future flexibility."

### TDD (Test-Driven Development)
Every code task should include:
1. Write failing test
2. Run to verify failure
3. Write minimal code
4. Run to verify pass

### Frequent Commits
Commit after every task with clear messages.

## Common Mistakes

| Bad | Good |
|-----|------|
| "Add authentication" | "Create User model with email and password_hash" |
| "Add validation function" | "Add validation function" + complete code |
| "Test it works" | "Run `pytest tests/test_auth.py -v`, expected: 3 passed" |
| "Create the model file" | "Create: `src/models/user.py`" |

## Execution

After planning, offer:
> "Plan complete and saved to `tasks/YYYY-MM-DD-feature-name/`. Ready to implement task-by-task. Shall I proceed?"

When implementing:
- Follow tasks sequentially
- Create step files documenting progress
- Update README with status
- Commit after each task

## Related Skills

- `implementation` - For coding tasks
- `testing` - For test writing
- `hpc-deployment` - For deployment workflows
