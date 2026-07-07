# News/Event Candidate Ingestion Workflow

Use this workflow for compliant, research-only news or event imports.

## Guardrails

- Do not scrape Reddit, X/Twitter, forums, or websites.
- Do not crawl websites or bypass robots/rate limits.
- Use manually provided CSV rows or manually provided URLs only.
- Do not make LLM calls in this workflow.
- Do not create active catalysts from candidates.
- Do not change scanner scoring.
- Imported candidates must become `research_event_annotations` only after explicit review.
- All imported annotations must remain `research_only = true` and `scanner_scoring_effect = 0`.

## Candidate Lifecycle

1. Stage candidates from a CSV/manual provider.
2. Review staged candidates in Model Lab.
3. Mark candidates as `accepted` or `rejected`.
4. Import accepted candidates as research-only annotations.
5. Run scanner invariance, Dataset 49 manifest check, annotation coverage, and tests.

Candidate statuses:

- `staged`: parsed and waiting for review.
- `accepted`: approved for research-only annotation import.
- `rejected`: reviewed and excluded.
- `duplicate`: matched an existing annotation or candidate.
- `imported`: converted into a research-only annotation.

## CSV Template

Use:

`docs/templates/news_event_candidates_template.csv`

Required columns:

- `ticker`
- `event_date`
- `title`

Recommended columns:

- `available_at`
- `event_type`
- `summary`
- `source`
- `source_url`
- `evidence_text`
- `sentiment_label`
- `strength`
- `confidence`
- `tags`
- `source_quality`
- `informativeness`
- `review_note`
- `provider_name`
- `provider_event_id`
- `duplicate_theme_key`
- `provider_metadata_json`

Source-quality categories:

- `official_company`
- `regulator`
- `exchange_or_index_provider`
- `sec_archive`
- `credible_news`
- `manual_note`
- `unknown`

Event-informativeness labels:

- `material_high`
- `material_medium`
- `routine_low`
- `duplicate_theme`
- `low_specificity`

If these fields are omitted, the app infers them conservatively from source, source URL, event type, sentiment, strength, confidence, tags, and text specificity. Inferred labels are audit metadata only unless a future phase explicitly promotes them into model-facing features.

## Deduplication

Candidates are deduplicated using:

- ticker
- event date
- normalized title
- source URL
- evidence text hash, when evidence is present

Duplicates are staged as `duplicate`; they are not imported automatically.
