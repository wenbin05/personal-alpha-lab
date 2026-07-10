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

For the strict company IR / press-release provider, use:

`docs/templates/company_ir_press_release_candidates_template.csv`

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
- `document_type` (`company_ir_press_release` for the strict IR provider)
- `published_at`
- `raw_text`
- `cleaned_text` or `text`
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

## Strict Company IR / Press-Release Provider

The `company_ir_press_release` provider is local-only and accepts only user-supplied CSV rows or user-supplied URLs embedded in CSV rows. It does not fetch URLs, crawl websites, discover company pages, or import rows automatically.

Additional rules:

- `source_url` is required and must be an `http` or `https` URL.
- `provider_name` must be omitted or set to `company_ir_press_release`.
- `source_quality` must be omitted or set to `official_company`; omitted values are defaulted to `official_company` and surfaced as a warning for reviewer verification.
- rows are staged as candidates and must be accepted before they can be imported as research-only annotations.
- accepted rows import only into `research_event_annotations` with `research_only = true` and `scanner_scoring_effect = 0`.

## Optional Company IR Source-Document Bridge

An accepted `company_ir_press_release` candidate can create or reuse a local `SourceDocument` only when the reviewer explicitly enables document linkage during import or deliberately links an already imported candidate.

- Source text is taken from `raw_text`, then `cleaned_text`/`text`, then `evidence_text` as a limited fallback.
- Evidence-only documents are marked partial and warn that full source text was not provided.
- Documents use `document_type=company_ir_press_release` and `source=company_ir_press_release`.
- Candidate and research-annotation rows both retain the linked `source_document_id`.
- Documents are deduplicated by ticker plus source URL, ticker plus stable text hash, then ticker/title/published-date fallback.
- Linked documents are visible in Documents / Text and become selectable in LLM Review.
- No fallback or OpenAI extraction runs automatically. No active catalyst is created, and scanner scoring remains unchanged.

## Deduplication

Candidates are deduplicated using:

- ticker
- event date
- normalized title
- source URL
- evidence text hash, when evidence is present

Duplicates are staged as `duplicate`; they are not imported automatically.
