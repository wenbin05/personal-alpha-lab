# Token-Efficient Execution Workflow

Use this workflow to keep project phases focused, safe, and concise.

## Context-First Rule

1. Read `AGENTS.md` and `docs/research/current_state_handoff.md` first.
2. Read only files explicitly named in the task.
3. Inspect an additional file only when a concrete import, schema, or runtime dependency requires it.
4. Do not perform broad repository exploration by default.

## Scope Budget

Before implementation, state:

- target outcome,
- expected files to change,
- expected validation tier,
- explicit out-of-scope items.

If the task materially exceeds the expected file set or crosses another subsystem, stop and report the dependency instead of silently expanding scope.

## Validation Tiers

### Tier 0 - Documentation Only

- Run `git diff --check`.
- Confirm only intended documentation files changed.
- Do not run pytest, compileall, browser smoke tests, or scanner snapshots.

### Tier 1 - Isolated Code Change

- Run targeted tests for the changed module.
- Compile affected modules.
- Run the relevant lightweight harness command.
- Do not run a full browser smoke test unless UI code changed.

### Tier 2 - Cross-Layer Feature

- Run targeted tests first, then full pytest.
- Run compileall.
- Run relevant quality harness checks.
- Browser smoke-test only changed UI pages.
- Run scanner invariance only when scoring or data paths could be affected.

### Tier 3 - Dataset, Model, Or Scoring Change

- Follow the full acceptance-audit workflow.
- Run dataset manifest and leakage checks.
- Capture scanner before/after snapshots.
- Run model or holdout governance checks as applicable.
- Back up the database before mutation.

## Token-Saving Implementation Rules

- Prefer existing repository functions and harness commands.
- Do not reopen files already summarized unless exact implementation details are required.
- Do not repeatedly inspect generated artifacts.
- Do not rerun successful expensive checks after documentation-only changes.
- Run targeted tests before full tests.
- Avoid restating the entire project history in the final response.
- Do not generate large reports unless requested.

## Standard Compact Final Report

Use this structure:

- Outcome
- Files changed
- Validation performed
- Safety/invariance result
- Commit and git status
- Blockers or next decision

## Prompt Shorthand

Future prompts may say:

> Follow the token-efficient execution workflow using Tier N validation.

Use the named tier unless a higher tier is required by a concrete safety dependency. Report that dependency before expanding validation.

## Escalation Rule

Stop and ask or report when:

- the task requires an unmentioned schema migration,
- the expected file count materially expands,
- scanner scoring could change,
- Dataset 50 would be evaluated or promoted,
- external network or API access becomes necessary,
- destructive database operations are required.
