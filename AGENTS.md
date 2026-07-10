# Personal Alpha Lab Agent Guide

Use this file as durable project context for future Codex work. Keep changes small, auditable, and aligned with the existing modular Python/Streamlit architecture.

Before beginning a new phase, read `docs/research/current_state_handoff.md` and inspect only the additional files needed for that phase.

## Project Guardrails

- This project is for personal U.S. equities research and paper trading only.
- Do not add buy, sell, or hold recommendations.
- Do not add real-money broker execution.
- Do not commit secrets, `.env`, API keys, local SQLite databases, generated datasets, cache files, virtual environments, or `__pycache__`.
- Do not change scanner scoring unless the user explicitly asks for a scoring phase.
- Preserve scanner invariance unless the phase explicitly changes scoring.
- LLM outputs must remain review-gated before they can affect catalysts or scoring.
- Pending, rejected, proposal-only, audit-link-only, and reverted LLM records must contribute zero to scanner scores.
- Dataset and model work must be point-in-time correct.
- Outcome labels must never appear in model features.
- Use chronological or walk-forward splits only; do not use random train/test splits for time-series equity research.
- Back up the active SQLite database before risky migrations, data-ingestion pilots, publication/reversal audits, or large backfills.
- Do not fabricate missing market data, SEC filings, earnings events, news, catalysts, labels, or provider responses.

## Development Rules

- Keep business logic in `src/`; keep `app.py` focused on Streamlit setup/routing.
- Prefer existing modules and repository patterns over new abstractions.
- Keep UI changes practical and transparent; avoid language that implies guaranteed profit.
- When changing data or dataset code, preserve deterministic hashes unless the phase explicitly changes feature definitions.
- When adding provider integrations, make them optional, cache-aware, failure-tolerant, and safe without paid API keys.
- Keep raw/provider data separate from curated/model-facing features.
- Preserve audit trails; prefer reversible status changes over destructive deletion.

## Verification Defaults

- Run `.venv/bin/pytest -q` before completion when code or tests change.
- Run `python3 -m compileall -q personal-alpha-lab` from the parent directory before completion when Python code changes.
- Browser smoke-test relevant Streamlit pages when UI code changes.
- For dataset/scoring changes, compare scanner outputs before and after unless scoring changes are explicitly in scope.
- For database migrations or ingestion pilots, report the backup path, table counts or dataset IDs, warnings, and any failed items.

## Useful Workflow References

- Point-in-time dataset/model evaluation: `docs/workflows/point_in_time_dataset_model_workflow.md`
- Acceptance audits: `docs/workflows/acceptance_audit_workflow.md`
