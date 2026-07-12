# Company IR Document Enrichment Workflow

Use this workflow to attach manually supplied company investor-relations or press-release text to existing research candidates and annotations. It never retrieves a URL and never runs LLM extraction.

## Complete The Queue

Start from the latest document-enrichment queue, such as:

```text
data/processed/phase2e5a_company_ir_enrichment_queue.csv
```

Keep `candidate_id` unchanged. Add manually obtained source text to `raw_text`, `cleaned_text`, or `text`. Optional fields include `annotation_id`, `source_document_id`, `document_type`, `title`, `source_url`, `published_at`, `available_at`, and `review_note`.

Set `text_completeness` to one of:

- `complete`: the supplied text is the complete press release or IR document.
- `partial`: the supplied text is a meaningful but incomplete excerpt.
- `evidence_only`: the supplied text is only quoted evidence.

Missing completeness defaults conservatively to `partial`. Never describe an excerpt as a complete document.

## Dry Run

Dry-run is the default and does not modify SQLite:

```bash
.venv/bin/python scripts/import_company_ir_document_enrichment.py \
  --input data/processed/completed_company_ir_enrichment.csv \
  --dry-run \
  --report-output data/processed/company_ir_enrichment_dry_run.json
```

Review row-level errors, create/reuse decisions, candidate and annotation link plans, and projected document coverage. The importer rejects missing candidates, wrong providers, rejected candidates, annotation mismatches, unsupported document types, and missing or unusable text.

## Explicit Apply

Apply only after reviewing a clean dry-run report:

```bash
.venv/bin/python scripts/import_company_ir_document_enrichment.py \
  --input data/processed/completed_company_ir_enrichment.csv \
  --apply \
  --report-output data/processed/company_ir_enrichment_apply.json
```

`--apply` creates a timestamped SQLite backup before writing. It creates or reuses a SourceDocument using ticker plus normalized source URL, stable text hash, then title/date fallback. Existing candidates and annotations are linked; they are not recreated. Reapplying the same CSV is idempotent.

Keep the backup until coverage and linkage checks pass. If an apply must be rolled back, stop Streamlit and restore the complete backup deliberately; do not copy individual rows between databases.

## Guardrails

- Input must be manually supplied. There is no URL fetching, crawling, scraping, or automatic discovery.
- Candidate and annotation research content is preserved; only SourceDocument linkage and audit timestamps change.
- Ingestion timestamps use the current application time and are never backdated.
- `available_at` remains event-availability provenance and is not treated as ingestion time.
- Documents become available for manual Documents / Text and LLM Review workflows only.
- No fallback or OpenAI extraction runs automatically.
- No active catalyst or scanner score changes occur.
