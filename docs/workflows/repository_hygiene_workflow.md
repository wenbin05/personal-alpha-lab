# Repository Hygiene Workflow

Use this workflow before committing accepted project phases. It is intentionally conservative: commit durable source, tests, docs, templates, and scripts; keep local data, secrets, and generated run artifacts out of git unless the user explicitly requests otherwise.

## When To Commit

Commit when:

- a phase has been accepted or is ready for handoff,
- the next phase is about to start and the current work should be checkpointed,
- the staged files are limited to source, tests, docs, templates, and scripts unless the user explicitly asks to commit other artifacts.

Do not commit exploratory local data just because it was generated during validation. Validation artifacts may be useful locally, but they should stay ignored unless the phase explicitly designates them as tracked examples or templates.

## Never Commit

Never stage or commit:

- `.env`
- `.venv/`
- SQLite database files such as `data/*.db`
- database backups such as `data/*backup*.db`
- `data/processed/` run artifacts unless explicitly requested
- `__pycache__/`
- `.pytest_cache/`
- API keys, tokens, secrets, or credential-bearing logs

If any forbidden item appears staged, unstage it before committing.

## Pre-Commit Checks

Run and inspect:

```bash
git status --short --ignored
git diff --cached --name-only
```

Confirm staged files are expected and do not include forbidden paths. A useful staged-file check is:

```bash
git diff --cached --name-only | grep -E '(^\\.env$|^\\.venv/|\\.db$|backup.*\\.db$|^data/processed/|__pycache__|\\.pytest_cache)'
```

The command should print nothing. If it prints a path, unstage that path.

Run the standard validation checks:

```bash
.venv/bin/pytest -q
python3 -m compileall -q personal-alpha-lab
```

Run relevant quality harness checks for the phase. Common examples:

```bash
.venv/bin/python scripts/quality_harness.py health-check
.venv/bin/python scripts/quality_harness.py dataset-check --dataset-id 49
.venv/bin/python scripts/quality_harness.py holdout-status --dataset-id 50
.venv/bin/python scripts/quality_harness.py provider-readiness
```

If scanner invariance matters for the phase, compare stored before/after scanner snapshots:

```bash
.venv/bin/python scripts/quality_harness.py scanner-compare \
  --before data/processed/<before>.json \
  --after data/processed/<after>.json \
  --fail-on-change
```

If Streamlit is already running, verify the health endpoint:

```bash
.venv/bin/python scripts/quality_harness.py health-check
```

Keep the local server running if the phase or user asked for it.

## Commit Message Format

Use a short imperative message. Mention the phase or durable capability when useful.

Good examples:

- `Add compliant research event provider readiness`
- `Add holdout maturity workflow`
- `Add research annotation quality filters`

Avoid vague messages such as `updates`, `fix stuff`, or `final changes`.

## Commit

Stage only approved files explicitly, for example:

```bash
git add src tests docs scripts
git status --short
git diff --cached --name-only
git commit -m "Add compliant research event provider readiness"
```

Prefer explicit paths when there are many ignored artifacts in the repository.

## Post-Commit Checks

After committing, run:

```bash
git status --short
git rev-parse --short HEAD
```

Report:

- commit hash,
- files committed,
- tests and runtime status,
- intentionally untracked or ignored files such as `.env`, `.venv/`, DB files, DB backups, and run artifacts.

Do not push unless the user explicitly asks.

## Push Guidance

Push only when requested by the user. Confirm the branch, then use:

```bash
git branch --show-current
git push origin <branch>
```

Do not push secrets, local databases, generated run artifacts, or backups.
