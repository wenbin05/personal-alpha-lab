# Quality Harness Workflow

Use this workflow before and after annotation/model phases. The harness is read-only unless you explicitly pass an output path for JSON or Markdown artifacts.

## Commands

Run commands from the repository root.

```bash
python scripts/quality_harness.py scanner-snapshot --output data/processed/scanner_before.json
python scripts/quality_harness.py scanner-compare --before data/processed/scanner_before.json --after data/processed/scanner_after.json --fail-on-change
python scripts/quality_harness.py dataset-check --dataset-id 49
python scripts/quality_harness.py model-compare --model-run-id 247 --baseline-run-id 145
python scripts/quality_harness.py annotation-coverage --dataset-id 49
python scripts/quality_harness.py holdout-status --dataset-id 50
python scripts/quality_harness.py portfolio-shadow-status
python scripts/quality_harness.py provider-readiness
python scripts/quality_harness.py document-coverage --provider company_ir_press_release --output data/processed/company_ir_document_coverage.json --queue-output data/processed/company_ir_enrichment_queue.csv
python scripts/quality_harness.py health-check
python scripts/quality_harness.py annotation-template-path
```

## Guardrails

- Scanner snapshot reads stored `scan_results`; it does not refresh market data.
- Scanner comparison reports score, label, risk, and catalyst-score changes.
- Dataset checks confirm labels, audit columns, identifiers, and metadata are excluded from model features.
- Model comparison uses persisted final-test metrics only.
- Annotation coverage uses point-in-time Dataset 49 rows and detects zero-coverage features, source-quality/informativeness distributions, low-specificity neutral rows, routine SEC-heavy rows, material non-SEC rows, and coarse future-availability violations.
- Holdout status reports label coverage, promotion gates, and cache-only extension availability; an immature holdout candidate is expected to return a successful command status when manifest/leakage checks pass.
- Portfolio shadow status verifies the frozen prospective policy boundary, immutable cohort/constituent integrity, fixed weights, outcome maturity, and forward-sample language without creating a cohort or evaluating Dataset 50.
- Provider readiness reports configured research-event providers, disabled/blocked providers, compliance notes, API key requirements, and whether any network calls would occur.
- Company IR source-document linkage is local and opt-in; harness validation should confirm candidate/annotation/document links without creating LLM extractions or active catalysts.
- Document coverage uses read-only SQLite access, makes no provider calls, and exports a workflow-priority queue only; its priority is not an alpha or scanner score.
- The annotation template at `docs/templates/research_annotations_template.csv` contains synthetic demo rows only. Do not import it into production data unless explicitly requested.
- The Streamlit health check verifies `http://localhost:8501/_stcore/health` and does not stop or restart the server.
