# Personal Alpha Lab

Personal Alpha Lab is a local Streamlit research assistant for U.S. equities. It helps scan, rank, research, backtest, and monitor a small editable universe of stocks and ETFs using transparent, rule-based logic.

This first draft is intentionally lean. It uses free OHLCV data through `yfinance`, local SQLite storage, explainable scores, and simple educational backtests. It does not place trades and does not require paid API keys.

## Disclaimer

This project is for personal research and paper trading only. It is not financial advice, investment advice, or a production trading system. Backtests are simplified and may not reflect live performance. There is no broker execution in v1.

## Features

- Market regime dashboard for SPY, QQQ, IWM, QQQ/SPY, IWM/SPY, and VIX when available.
- Daily ranked watchlist with explainable 0-100 alpha scores, including auditable catalyst contribution.
- Catalyst Center for manual notes, SEC filing metadata, and future provider adapters.
- Historical earnings-event ingestion for point-in-time dataset features, using yfinance best-effort data or CSV import.
- Documents / Text ingestion page for local source documents, manual pasted text, CSV imports, and optional SEC filing text fetches.
- LLM Review page for fallback and optional OpenAI extraction, with manual approve/reject/supersede workflow.
- Review-only catalyst proposals and extraction-to-catalyst audit links that remain separate from active catalyst scoring.
- Dataset Lab for point-in-time feature snapshots, future outcome labels, leakage checks, and versioned CSV exports.
- Ticker research page with price/volume charts, moving averages, feature breakdowns, and rule-based summaries.
- Simple backtests for momentum breakout, mean reversion, moving average trend, and top-score weekly rebalance.
- SQLite-backed paper trade journal.
- Alert preview text for top-ranked tickers.
- Validation / Debug page for auditing OHLCV metadata, feature values, score components, penalties, and warnings.
- Clean placeholders for future news, LLM, options, and broker-paper adapters.

## Setup

```bash
cd personal-alpha-lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional environment setup:

```bash
cp .env.example .env
```

The app runs without paid API keys. Historical SEC EDGAR metadata/text requests require a descriptive `SEC_USER_AGENT` in `.env` so requests identify this local research app and a real contact email. If it is missing or left as the example placeholder, SEC backfills fail gracefully and no filings are downloaded.

Optional OpenAI extraction configuration:

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=...
LLM_MODEL=...
LLM_MAX_INPUT_CHARS=12000
LLM_TIMEOUT_SECONDS=60
```

OpenAI API usage is billed separately from ChatGPT Plus. The app never displays or stores the API key, and no OpenAI call is made unless you explicitly run one from the `LLM Review` page.

## Run The App

```bash
streamlit run app.py
```

The app creates `data/alpha_lab.db` automatically. Historical OHLCV data is cached in SQLite so repeated scans do not redownload everything unnecessarily.

## Run Tests

```bash
pytest
```

## Edit The Universe

The default universe is stored in:

```text
config/universe.csv
```

Add or remove rows with columns:

```text
ticker,name,sector
```

V1 is designed for a small personal research universe. It is not intended to scan thousands of tickers unless you explicitly expand it and accept slower free-data downloads.

## Scoring

The score engine returns a transparent dictionary:

```python
{
    "ticker": "XYZ",
    "score": 0-100,
    "label": "Strong Watch | Watch | Neutral | Weak | Avoid",
    "breakdown": {...},
    "penalties": [...],
    "reasons": [...]
}
```

Default score weights:

- Market regime compatibility: 15
- Momentum/trend: 20
- Relative strength vs SPY: 15
- Volume anomaly: 10
- Liquidity quality: 15
- Catalyst events/manual notes: 10
- Options placeholder: 10
- Risk/reward setup: 5

Penalties are applied after the weighted score:

- Low liquidity: -20
- Price below $5: -10
- Below 200D moving average: -10
- Extreme overextension: -5 to -15
- Missing or limited data: -5 to -20

Labels:

- 80 to 100: Strong Watch
- 65 to 79: Watch
- 50 to 64: Neutral
- 35 to 49: Weak
- Below 35: Avoid

## Market Regime

The Market Regime page uses SPY, QQQ, IWM, and VIX when available:

- Risk-On: SPY above 50D MA, 50D MA above 200D MA, QQQ/SPY relative strength positive, and VIX not elevated.
- Risk-Off: SPY below 200D MA or VIX elevated.
- Small-Cap Rotation: IWM is above its 50D MA and improving versus SPY.
- Neutral or Choppy: fallback labels when the tape is mixed.

If SPY data is missing or too short for a reliable 200D moving average, the app defaults conservatively to `Neutral` and shows a warning/confidence note rather than presenting a confident regime label.

## Data Storage

SQLite tables are created automatically:

- `ohlcv_cache`: downloaded daily OHLCV cache
- `scan_results`: latest scanner runs
- `trade_journal`: local paper trade journal
- `manual_catalysts`: user-entered catalyst notes
- `catalysts`: provider-agnostic catalyst/event records for manual notes, SEC metadata, news placeholders, and future adapters
- `earnings_events`: provider-agnostic historical earnings events with EPS/revenue fields, announcement/availability timestamps, provider payload hashes, quality warnings, and deterministic dedupe keys
- `earnings_event_revisions`: immutable change history when a provider revises a previously stored earnings event
- `earnings_response_cache`: local cache for best-effort earnings-provider responses
- `source_documents`: provider-agnostic source text records linked to tickers and optionally to catalyst events
- `llm_extractions`: provider-agnostic extraction records, review statuses, evidence snippets, and reviewer notes
- `catalyst_proposals`: reviewed proposal drafts mapped deterministically from approved extractions
- `extraction_catalyst_links`: reversible audit links between approved extractions and existing catalysts
- `catalyst_publications`: immutable publication/reversal audit records with before/after snapshots and catalyst-component deltas
- `dataset_builds`: point-in-time dataset metadata, requested ranges, feature columns, label definitions, row counts, hashes, warnings, and export paths
- `feature_snapshots`: per-ticker, per-date point-in-time feature snapshots with grouped JSON feature families
- `outcome_labels`: future return labels linked to feature snapshots
- `catalyst_revisions`: immutable future change log for catalyst create/update/delete actions
- `backfill_runs`: resumable historical dataset backfill run metadata, progress, warnings, and status
- `backfill_items`: per-ticker backfill status, generated rows, coverage, label counts, and errors

The scanner page includes controls to refresh market data or clear the OHLCV cache.

## Catalyst Center

The `Catalyst Center` page is Phase 2A data enrichment. It adds catalyst/event infrastructure without LLMs, options data, machine learning, or broker trading.

You can manually add catalyst notes with:

- Ticker
- Event date
- Event type
- Title and summary/thesis
- Sentiment label
- Catalyst strength from 0 to 10
- Confidence from 0.0 to 1.0
- Optional source URL

Manual notes persist in SQLite and appear in:

- Catalyst Center
- Daily Scanner expandable ticker details
- Ticker Research
- Validation / Debug

The SEC adapter uses the free EDGAR submissions metadata API when available. It stores filing metadata for supported forms such as `10-K`, `10-Q`, `8-K`, `S-1`, `S-3`, `424B*`, and Form `4`, including amended variants. Important filing types like `8-K`, `S-1`, `S-3`, `424B*`, and Form `4` are labeled as needing manual review.

SEC timing uses three distinct timestamps:

- `event_date`: the filing date or event date shown to the user.
- `available_at`: when the information became public. SEC filings use EDGAR acceptance datetime.
- `created_at`: when this local app ingested or stored the record.

Historical datasets filter catalysts and filings with `available_at`, not the later ingestion timestamp and not the report-period date. Backfilling a 2023 filing in 2026 can therefore make it available to snapshots after its 2023 EDGAR acceptance time without leaking it into earlier rows.

SEC filing metadata can optionally fetch and store source text through the Documents / Text layer. The app stores cleaned text for preview and future extraction, but it does not interpret filing text with NLP or LLMs yet.

The historical earnings adapter uses best-effort yfinance metadata. Earnings dates, EPS fields, revenue fields, and announcement times may be absent, stale, or empty. If yfinance cannot return data, the app records a warning and continues to support manual CSV import. Earnings events are stored separately from active catalysts and do not alter scanner scoring.

The news adapter is currently a placeholder interface. No paid news provider is required, and the app does not scrape aggressively.

Active catalyst records and review-only proposals are deliberately separate:

- Active catalyst records in the `catalysts` table are the only catalyst records used by scanner scoring.
- Catalyst proposals are draft/review artifacts in `catalyst_proposals`; they do not alter active catalysts.
- Extraction-to-catalyst links are audit records only. Linking an approved extraction to a catalyst does not change that catalyst's date, title, sentiment, strength, confidence, or summary.
- Proposal statuses are `draft`, `reviewed_ready`, `rejected`, and `superseded`. `reviewed_ready` means ready for a future controlled publication workflow, not published.
- Controlled publication is explicit: only `reviewed_ready` proposals can be published, and publication requires a preview, confirmation checkbox, and publisher note.

Catalyst scoring is conservative:

- No catalyst events: 0 catalyst contribution, no penalty.
- Positive catalyst: adds up to 10 points in the catalyst component, scaled by strength and confidence.
- Negative catalyst: applies an explicit score penalty capped at -15.
- Neutral/unknown catalyst: no automatic boost or penalty.
- SEC filing metadata flagged `Needs Review`: warning only unless manually assigned sentiment later.

No catalyst event is an automatic buy/sell instruction. It is an input for research only.

## Documents / Text

The `Documents / Text` page is Phase 2A.5 infrastructure for future extraction. It collects and manages local source text without adding LLMs, options data, machine learning, new alpha strategies, or broker trading.

Stored source documents include:

- SEC filing text
- News articles or excerpts you paste manually
- Earnings call excerpts or transcripts you paste manually
- Imported CSV text/news rows
- Other manually supplied research text

Each document stores raw text, cleaned text, a stable text hash, parsing status, warnings, source metadata, optional SEC accession/form metadata, and an optional linked catalyst ID. Raw text and cleaned text are stored separately so cleaning remains auditable.

Manual text entry supports:

- Ticker
- Document type
- Source
- Title
- Published date
- Optional source URL
- Optional linked catalyst
- Raw pasted text

CSV import supports columns such as:

```text
ticker,document_type,title,published_at,source,source_url,text,sentiment_label,catalyst_strength,confidence
```

Rows with `sentiment_label`, `catalyst_strength`, or `confidence` can create a linked catalyst event. Rows with only text create source documents only. Import validation checks for missing ticker, missing text, invalid dates, unsupported document types, and duplicate source text/source URLs.

SEC filing text fetches use the filing URL stored in SEC metadata. Large filings are guarded by a maximum download size and may be stored as `partial`. Failed fetches store a warning/status instead of crashing the app. SEC text fetching can be affected by EDGAR availability, timeouts, file size, rate limits, malformed filings, or missing primary document URLs.

Source document availability improves auditability only. It does not materially boost scanner scores. Catalyst scoring still comes from catalyst events, not document count.

Privacy/security note: user-pasted text and imported CSV text are stored locally in SQLite at `data/alpha_lab.db`. Do not paste confidential third-party content unless you are comfortable storing it on this machine.

## LLM Extraction Foundation

Phase 2B adds the storage, validation, review workflow, fallback extractor, and optional OpenAI provider for source-document extraction.

The app now has a provider-agnostic `LLMExtraction` schema and a local `llm_extractions` SQLite table for future document reviews. Extraction records include detected event type, sentiment, catalyst strength, risk severity, confidence, document relevance, evidence sufficiency, time horizon, evidence snippets, summaries, proposed score effect, review readiness, reviewer notes, and sanitized response metadata.

Important guardrails:

- Every new extraction defaults to `pending_review`.
- Pending, rejected, and superseded extractions do not affect scanner scoring.
- The fallback extractor is deterministic keyword logic for pipeline testing only.
- The OpenAI provider is disabled unless `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, and `LLM_MODEL` are configured.
- OpenAI extraction requires an explicit confirmation checkbox and button click; it never runs during page load, filtering, or reruns.
- Document text leaves the machine only for explicitly confirmed OpenAI extractions.
- Fallback and OpenAI output never generate buy/sell/hold recommendations.
- Approval only changes review status; scoring integration is intentionally not active yet.
- Evidence snippets must be exact contiguous quotations from the submitted source text after whitespace normalization. Unsupported/paraphrased snippets are removed and not replaced.
- Exact evidence means the extraction is traceable to the stored source text; it does not prove that the source itself is factually true.

Confidence is defined narrowly: confidence that the extracted financial interpretation is directly supported by the supplied document text. It is not confidence that JSON parsed, that the model produced a label, or that the source is true.

Calibration bands:

- `0.75-1.0`: explicit, directly supported statements.
- `0.40-0.74`: partial context or ambiguous language.
- `0.0-0.39`: short, noisy, speculative, irrelevant, or insufficient text.

Quality fields:

- `document_relevance`: `relevant`, `uncertain`, `irrelevant`, or `unknown`.
- `evidence_sufficiency`: `sufficient`, `limited`, `insufficient`, or `unknown`.
- `review_readiness`: `ready_for_review`, `needs_evidence`, or `insufficient_document`.

## Review-Only Catalyst Proposals

Phase 2B-4A adds a proposal layer for approved extractions. It is designed to make future catalyst publication auditable without allowing LLM output to mutate active catalyst records automatically.

How the workflow works:

- Approve an extraction on `LLM Review`.
- In the reviewed extraction details, inspect the deterministic catalyst proposal preview.
- Optionally select an existing same-ticker catalyst as the target for an `update_existing` proposal.
- Create a non-scoring proposal. Weak-readiness extractions (`needs_evidence` or `insufficient_document`) require an explicit override and reviewer note.
- Edit proposed fields manually if needed.
- Mark the proposal as `reviewed_ready`, `rejected`, or `superseded`.
- Optionally link or unlink an approved extraction to an existing catalyst as a reversible audit record.

Proposal mapping is deterministic:

- Ticker comes from the approved extraction.
- Event type, sentiment, strength, confidence, risk severity, relevance, and sufficiency come from the validated extraction.
- Event date comes from the source document published date when available.
- Title is generated from ticker, detected event type, and document title.
- Summary comes from the extraction short summary.
- Source URL comes from the source document.
- Evidence uses only validated exact evidence snippets already stored on the extraction.

Important limits:

- Proposals are not active catalysts.
- Proposals have `LLM proposal score contribution: 0`.
- Links do not mutate active catalyst fields.
- `reviewed_ready` is still review-only; future publication/scoring remains a separate phase.
- Grounded evidence only proves traceability to source text, not that the source is factually true.

## Controlled Catalyst Publication

Phase 2B-4B adds controlled publication from a manually reviewed proposal into the active catalyst table. Nothing publishes automatically.

Publication eligibility:

- Extraction must be `approved`.
- Proposal must be `reviewed_ready`.
- Proposal must not already have an active publication.
- Document relevance must be `relevant`.
- Evidence sufficiency must be `sufficient` or `limited`.
- Non-neutral proposals require at least one validated exact evidence snippet.
- Ticker must match across extraction, source document, proposal, and target catalyst.
- `update_existing` proposals require an existing same-ticker target catalyst.

Publication is blocked for pending/rejected/superseded extractions, draft/rejected/superseded proposals, irrelevant/uncertain/unknown documents, insufficient/unknown evidence, missing evidence for non-neutral signals, and duplicate publication attempts. There is no silent override that turns insufficient LLM evidence into an active LLM-supported signal.

The publication preview shows:

- Source document metadata.
- Approved extraction metadata.
- Exact evidence snippets.
- Proposed catalyst fields.
- Target catalyst for updates.
- Field-level before/after diff.
- Catalyst component before and after using the production catalyst scoring function.
- Catalyst-only delta.
- Provenance and warnings.

Scoring behavior:

- Published catalysts become normal active catalyst records.
- Published catalysts affect scores only through the existing catalyst sentiment, strength, and confidence formula.
- `proposed_score_effect` is never added directly to scanner score.
- There is no separate LLM score component.
- Existing caps remain in force: positive catalyst contribution up to +10 and negative catalyst penalty down to -15.
- Pending extractions, proposal-only rows, audit links, and reverted publication rows contribute 0.
- Limited-evidence publications cap active catalyst confidence at `0.60`.
- Published confidence cannot exceed the approved extraction confidence.
- Only one active publication may exist for a proposal.

Published catalysts are tagged as `llm_supported` and `manually_reviewed` in active catalyst provenance metadata. Provenance links publication ID, proposal ID, extraction ID, source document ID, provider/model metadata, evidence snippets, and source document URL. The original document text is not duplicated into publication metadata.

Reversal:

- Created catalysts can be reverted only if the catalyst has not changed since publication.
- Updated catalysts can be restored to the exact pre-publication snapshot only if the catalyst has not changed since publication.
- If a catalyst was manually edited after publication, automatic reversal is blocked and the user must resolve it manually.
- Reversal never erases the publication audit record; it marks the publication `reverted`.

## LLM Review

The `LLM Review` page lets you run either the deterministic fallback extractor or an explicitly configured OpenAI extraction on stored `SourceDocument` records, then manually review the extraction result next to the source text.

How to use it:

- Add or import source text on `Documents / Text`.
- Open `LLM Review`.
- Filter documents by ticker and select a stored document.
- Inspect document metadata, warnings, parsing status, and source-text preview.
- Choose `Fallback test mode` or `OpenAI`.
- Choose an extraction type.
- For OpenAI, confirm that the selected text will be sent to OpenAI and press `Run OpenAI Extraction`.
- Review the pending extraction alongside relevance, evidence sufficiency, readiness, positive points, risks, exact evidence snippets, summaries, warnings, provider metadata, and the cleaned source-text preview.
- Add a reviewer note and approve, reject, or supersede the extraction.
- For approved extractions, optionally create a review-only catalyst proposal or link the extraction to an existing catalyst for auditability.
- For `reviewed_ready` proposals, inspect the publication preview and explicitly publish into active catalysts only after confirming and adding a publisher note.

LLM review limitations:

- Fallback results are keyword-based, low-confidence, and intended only to test the review pipeline.
- OpenAI results are schema-validated and evidence snippets are checked against submitted text, but they still require manual review.
- If review readiness is `needs_evidence` or `insufficient_document`, approval requires an explicit override checkbox and a non-empty reviewer note.
- Provider failures do not create trusted extraction records.
- The app stores sanitized response metadata, validated model output, response ID, and token usage when available; it does not store API keys or duplicated full request payloads.
- Empty or unusable documents are blocked before extraction.
- If a document already has a pending extraction, rerunning requires an explicit supersede checkbox.
- Approved/rejected/superseded records remain readable in review history.
- Approval only changes `review_status` to `approved`; it does not change catalysts, alerts, scanner scoring, or recommendations.
- Proposal creation/linking from approved extractions still does not change active catalysts, alerts, scanner scoring, or recommendations.
- Publishing a reviewed-ready proposal is the only workflow here that changes active catalysts, and it remains reversible when no later catalyst edits conflict.

## Trading Calendar

Cache freshness and stale-data warnings use `src/utils/trading_calendar.py`, which follows U.S. equities trading days in the America/New_York timezone. If `pandas_market_calendars` or `exchange_calendars` is installed, the helper uses a real NYSE calendar. Otherwise it falls back to weekday logic plus common NYSE holidays such as New Year's Day, MLK Day, Presidents Day, Good Friday, Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving, and Christmas.

The app assumes daily bars are not reliably available until after 18:00 ET. If you run the app from Asia/Singapore during the U.S. trading day, the latest expected bar is usually the previous U.S. trading session.

## Dataset Lab

The `Dataset Lab` page is the Phase 2C-1 point-in-time research dataset foundation. It does not train models. It builds auditable historical rows that later baseline ML or Deep Learning experiments can consume.

What it builds:

- `FeatureSnapshot` rows: ticker, trading date, as-of timestamp, feature version, market-regime features, technical/momentum features, relative strength, volume/liquidity, active catalyst features, published LLM-supported catalyst features, and data-quality flags.
- `OutcomeLabel` rows: forward returns for 1, 5, and 20 sessions plus matching SPY forward returns and excess returns.
- `DatasetBuild` rows: requested date range, ticker universe, explicit model-feature column list, audit/label/identifier/metadata column lists, feature manifest, label definitions, row count, deterministic data hash, warnings, and export path.

Timing convention:

- A snapshot dated `T` is treated as available after the close of `T`.
- Features use cached market data only through `T`.
- Entry uses the next cached trading session close after `T`.
- Exit uses the close `N` sessions after entry.
- The signal-date-to-entry return is not included in the label.
- `label_available_at` is after the exit-date close, so labels are never feature inputs.

Leakage controls:

- Market data is sliced through the snapshot date before feature functions run.
- Relative strength uses ticker and SPY returns over aligned windows ending at the snapshot date.
- Manual/system catalysts are included only if their `created_at` timestamp is no later than the snapshot as-of timestamp.
- LLM-supported catalyst data is included only after controlled publication time, using publication time rather than source document date.
- Published LLM-supported catalysts are historically available when `published_at <= snapshot_time < reverted_at`. If a publication is later reverted, that future reversal does not erase earlier snapshots where the publication was active.
- Superseded publications are treated as active until their supersession timestamp.
- Pending, rejected, proposal-only, audit-link-only, and reverted publication rows contribute zero.
- Reverted publications are excluded after their reversal time; publication snapshots preserve the earlier active window for historical reconstruction.
- Every exported dataset column has an explicit role: `identifier`, `metadata`, `model_feature`, `audit`, or `label`.
- `feature_columns_json` contains model-ready features only. Audit fields, identifiers, timestamps, policy versions, workflow/status fields, and labels are excluded from default training inputs.
- The training loader returns `X`, a selected `y`, metadata, and audit columns separately; outcome labels never appear in `X`.
- Dataset builds are append-only. A new build creates a new `dataset_id` and a versioned CSV instead of silently overwriting an earlier build.

Phase 2C-2 adds resumable backfill runs:

- `Start Backfill Run` creates a dataset build and one backfill item per ticker.
- `Resume Backfill` processes pending tickers and can stop safely between tickers.
- `Retry Failed Tickers` resets failed items to pending.
- Snapshot and label inserts are idempotent per dataset/ticker/trading date, so rerunning a run does not create duplicate snapshots or labels.
- Existing cached OHLCV is used first. A provider fetch is attempted only when the requested range is not already covered.
- Each item records provider metadata, fetch timestamp, price-adjustment convention, generated snapshots, completed labels by horizon, warnings, and errors.

Phase 2C-3B.2 hardens the build/write path for larger pilots:

- Feature snapshots are written with bounded batch inserts instead of one transaction per snapshot.
- Outcome labels remain batched and idempotent by snapshot/horizon.
- Market-regime values are precomputed once per trading date and reused across tickers.
- Neutral non-SEC catalyst state is precomputed when no manual/system/LLM publication state can affect the requested window.
- Dataset finalization streams flattened rows for deterministic hashing and CSV export instead of holding duplicate full flattened frames.
- The stream hash preserves the existing canonical contract: rows are ordered by ticker/date, storage IDs are excluded from the hash, and CSV bytes match the previous full-DataFrame export.
- Dataset Lab previews are bounded; full exports happen only through explicit build/backfill actions.
- Backfill metadata records timing and peak-RSS measurements to help identify future bottlenecks.

Phase 2C-4C adds the final dataset-build memory hardening:

- Full-universe cached builds can run in cache-only mode for validation so missing OHLCV ranges are reported without provider downloads.
- Per-ticker technical, SEC, and earnings features are precomputed with bounded/vectorized paths before snapshot rows are assembled.
- Dataset-build caches are cleared between tickers so ticker-local frames do not stay retained for the whole build.
- Zero-duration publish/revert audit rows are treated as inactive windows and do not trigger unnecessary historical catalyst reconstruction.
- SEC EDGAR revision history is skipped in non-SEC catalyst reconstruction paths, preserving scanner/catalyst semantics while avoiding large audit-only allocations.

Phase 2C-3A adds historical SEC filing ingestion for small corporate-equity pilots:

- `Historical SEC Filing Backfill` in Dataset Lab fetches SEC metadata only for selected corporate tickers and date ranges.
- Requests use cached SEC responses first, bounded request pacing, conservative retry behavior, and per-ticker failure isolation.
- Filings are stored as neutral, needs-review `sec_filing` catalyst events with EDGAR acceptance time as `available_at`.
- SEC metadata does not infer sentiment, does not change catalyst scoring by filing type, and does not fetch every full filing document automatically.
- Deterministic point-in-time SEC features are available in Dataset Lab snapshots. Model-facing SEC fields use category-specific unique event-day counts, presence flags, days-since-latest category events, metadata availability, and separate structured-note, ownership, core-periodic, and current-event activity.
- Raw filing counts, generic eligible filing counts, Form 4 raw counts, unknown-classification counts, feature-excluded counts, needs-review workflow flags, and policy-version fields are retained as audit columns, not model features.
- 424B activity is not treated as a single equity-financing signal. The dataset separates `sec_recent_equity_financing_flag`, `sec_recent_structured_note_flag`, and `sec_recent_registration_or_prospectus_other_flag`.
- These SEC features are dataset features only. They do not directly alter the live scanner score.

Phase 2C-4A adds the historical earnings-event foundation:

- `Historical Earnings Event Backfill` in Dataset Lab fetches best-effort yfinance earnings metadata for selected tickers and date ranges.
- Manual CSV import is available for earnings rows when yfinance coverage is missing or incomplete.
- Earnings storage distinguishes `announced_at`, `available_at`, provider `fetched_at`, and local `created_at`/`updated_at`.
- Point-in-time snapshots include an earnings event only when `available_at <= snapshot after-close timestamp`.
- Before-market events may be available on that session; after-market events are available after the session close. Unknown timing uses a conservative provider-layer availability timestamp and warning.
- Model-facing earnings fields include event-present windows, sessions since latest earnings, latest EPS surprise percent/direction, optional revenue surprise percent, timing-known, and data-available controls.
- Provider metadata, raw values, warnings, missing-field counts, and event IDs are audit columns, not default model features.
- Earnings events do not automatically create positive/negative catalysts and do not affect live scanner scoring.

The page shows row count, feature/label column counts, ticker/date coverage, missingness, dataset preview, individual snapshot inspection, catalyst IDs available to a snapshot, label availability timestamps, per-ticker backfill progress, and a chronological split preview. Splits are deterministic training/validation/test periods with an optional session gap; random train/test splitting is intentionally not used for time-series research.

Data sufficiency report:

- Total rows, tickers, and years covered.
- Missingness by column.
- Label counts by horizon.
- Per-ticker coverage, generated snapshots, catalyst row counts, positive/negative catalyst counts, and LLM-supported catalyst rows.
- Return distribution for forward and excess-return labels.
- Honest warnings for unavailable catalyst revision history or empty datasets.

Current dataset limitations:

- It uses the local OHLCV cache first and can fetch missing OHLCV ranges through the configured free provider when coverage is incomplete.
- It is not a survivorship-bias-free institutional feature store.
- Historical reconstruction of catalysts is conservative. New catalyst create/update/delete actions are recorded in `catalyst_revisions`, but old records do not have invented revision history. Historical periods affected by missing revision history are marked with a data-quality warning.
- Catalyst history is backfilled only from local records. The app does not fabricate old news, filings, missing earnings events, or LLM-supported catalyst history.
- No text embeddings, model training, walk-forward optimization, Spark, or cloud infrastructure are included in Phase 2C-2.

Planned progression:

```text
Point-in-time dataset
→ automated historical backfill
→ simple baseline models
→ walk-forward comparison
→ Deep Learning only if it beats baselines out of sample
```

## Validation / Debug Page

Use the `Validation / Debug` page before trusting a scanner result or before adding new data providers. Select a ticker from `config/universe.csv` and choose a requested period; the page builds the audit report from the current controls.

The page shows:

- Raw OHLCV metadata: requested period, actual date range, row count, latest trading date, cache age, and cache/download source.
- Calendar metadata: latest expected U.S. trading day and whether the app is using a real exchange calendar or fallback holiday rules.
- Latest five OHLCV rows.
- Missing-value counts for Open, High, Low, Close, Adj Close, and Volume.
- Current engineered feature values, including returns, moving averages, distances from moving averages, volume ratio, average daily dollar volume, and relative strength versus SPY.
- Score component table for market regime, momentum, relative strength, volume, liquidity, catalyst placeholder, options placeholder, and risk/reward.
- Catalyst inputs: event rows, contribution, penalty, confidence, manual/system source, and warnings.
- Source document inputs: document count, recent document rows, linked catalyst state, parsing statuses, and warnings.
- LLM proposal inputs: proposal rows, extraction-catalyst audit links, and explicit `LLM proposal score contribution: 0`.
- Publication inputs: active/reverted publication rows, active publication count, and proof that proposal-only rows remain zero until publication.
- Applied penalties and the final score label.
- Data-quality warnings for insufficient history, stale latest date, missing or zero volume, suspicious prices, failed SPY comparison, missing OHLCV columns, and cache period mismatch. Stale-data warnings are calendar-aware and may indicate delayed free data, holiday/weekend timing, failed downloads, or stale cache.

This page is intended as the first stop when a result looks odd. It does not add LLM, options, broker, or machine-learning features.

## Backtesting Notes

The backtester is educational and simple:

- Signals are generated from historical close data.
- Positions enter and exit at the next trading day's close after the signal date in v1.
- The backtester does not earn the signal-date-to-entry return.
- Slippage defaults to 0.10% per trade.
- Benchmark comparison uses SPY buy-and-hold.
- Strategies are not optimized or validated for live use.

Included strategies:

- Momentum breakout: close above 50D MA, positive 20D return, and volume ratio above threshold.
- Mean reversion: price below 20D MA by a selected percentage while above 200D MA.
- Moving average trend: close above 50D MA and 50D MA above 200D MA.
- Scanner score strategy: ranks tickers every N trading days and equal-weights the top N.

For the top-score portfolio strategy, the UI labels win rate, average win/loss, and count metrics as rebalance-period approximations. The rows summarize portfolio holding periods and selected legs, not individual broker-style fills or precise order-level execution. This is useful for sanity checking the portfolio rule, but it is not a professional trade blotter.

## What Not To Trust Yet

- Catalyst data is metadata, manual notes, and optional local source text only; approved LLM extractions are not scoring inputs yet.
- Catalyst proposals and extraction-catalyst links are non-scoring review artifacts.
- Source documents and extractions improve review/auditability but are not automatically scored as alpha signals.
- Dataset Lab rows are research inputs only. They are not trained models and do not produce recommendations.
- News provider support is a placeholder unless you add/import events later.
- Options scores are still placeholders.
- SEC filing events marked `Needs Review` are not interpreted as bullish or bearish without manual review.
- Earnings data from free providers may be unavailable, partial, revised, or uncertain; revenue fields are often missing and are not inferred.
- yfinance data can be stale, revised, incomplete, rate-limited, or unavailable.
- Backtests do not model taxes, borrow costs, liquidity impact, partial fills, delistings, survivorship bias, or realistic portfolio operations.
- Portfolio backtest trade statistics are rebalance-period approximations, not order-level execution records.
- Market-cap filters are placeholders unless a future data provider supplies reliable metadata.
- Optional OpenAI extraction can be wrong or incomplete even with structured output and exact-evidence checks.

## Current Limitations

- Free yfinance data can be delayed, incomplete, rate-limited, or unavailable.
- No real news API in Phase 2A.
- No real options chain, unusual options activity, IV rank, or open-interest analysis in v1.
- SEC support includes metadata and optional raw/cleaned filing text fetches. No financial statement parser or automated scoring interpretation yet.
- Earnings support is best-effort yfinance metadata plus CSV import. It is dataset-only and does not create catalyst sentiment or scanner-score effects.
- LLM extraction is review-only. Fallback extraction is deterministic; OpenAI extraction is optional and never affects scores automatically.
- Review-only catalyst proposals are not published into active catalysts and never affect scores automatically.
- Controlled publication can make a reviewed proposal an active catalyst, but only through the existing catalyst formula and explicit user confirmation.
- Dataset Lab creates point-in-time snapshots, labels, and resumable automated backfills from local/cache-first data, but it does not solve survivorship bias or missing historical catalyst coverage.
- No live trading or broker execution.
- Market cap filters are placeholders unless your chosen data provider supplies reliable company metadata.
- Backtests do not model taxes, borrow costs, liquidity impact, partial fills, corporate action edge cases, or survivorship bias.

## Roadmap Hooks

Phase 2:

- Real news API
- Earnings calendar
- SEC filing parser
- LLM feature extraction for news, transcripts, and filings
- Telegram or email alerts

Phase 2B:

- Optional OpenAI extraction over stored source documents
- Structured extraction review with exact evidence snippets checked against submitted text
- Review-only catalyst proposal mapping and reversible extraction-to-catalyst audit links
- Controlled publication with before/after scoring preview and optimistic-concurrency reversal

Phase 2C:

- Point-in-time research dataset
- Automated historical backfill
- Simple baseline models before any Deep Learning
- Walk-forward comparison before trusting out-of-sample results

Phase 3:

- Options chain provider
- Unusual options activity scanner
- IV rank / IV percentile
- Call/put volume ratio
- Open-interest change
- Gamma exposure approximation where data is available

Phase 4:

- LightGBM/XGBoost ranking model
- Historical feature store
- Walk-forward validation
- Feature importance dashboard

Phase 5:

- Broker integration for paper trading only
- No live trading until explicitly approved by the user
