# Prospective Options Snapshot Workflow

## Purpose

Collect immutable, research-only daily option-chain observations for a bounded liquid-equity universe. Options records remain separate from Dataset 49, Dataset 50, frozen shadow artifacts, shadow predictions, and scanner scoring.

The initial universe is `AAPL`, `AMD`, `AMZN`, `COIN`, `META`, `MSFT`, `NVDA`, and `TSLA`. The default collection includes the nearest two valid expirations.

## Safety Contract

- Collection is prospective only; historical options chains are not reconstructed.
- `--dry-run` is the default behavior and makes no network calls or database changes.
- `--apply` explicitly authorizes yfinance option-chain calls and creates one timestamped database backup before mutation.
- No paid provider, scraping, LLM, broker, news, or social call is used.
- Each ticker is isolated so one provider failure does not discard successful tickers.
- Missing provider values remain missing and are never fabricated.
- A provider/date run and its contracts are immutable after insertion.
- Repeating the same provider/date collection is blocked rather than overwritten.
- A weekday collection before the existing 18:00 ET completed-session boundary is refused, preventing an intraday chain from being labeled as the prior completed session.
- Options observations do not affect scanner scores, shadow predictions, datasets, catalysts, or recommendations.

## Commands

Plan the default pilot without network or database access:

```bash
.venv/bin/python scripts/collect_options_snapshots.py --dry-run
```

Collect one prospective snapshot:

```bash
.venv/bin/python scripts/collect_options_snapshots.py --apply
```

Use an explicitly bounded universe or expiration count:

```bash
.venv/bin/python scripts/collect_options_snapshots.py \
  --apply \
  --tickers AAPL AMD NVDA TSLA \
  --max-expirations 2
```

Audit stored coverage without provider calls:

```bash
.venv/bin/python scripts/quality_harness.py options-status
```

## Snapshot Timing

`snapshot_date` uses the existing latest-completed U.S. session rule. `as_of_timestamp` stores the actual UTC collection time. The underlying reference price is bounded to history dates on or before `snapshot_date`; later or incomplete bars are not used. Provider contract timestamps are retained where yfinance supplies them.

## Deterministic Summaries

The summary service calculates put/call open-interest and volume ratios, nearest and next-expiry ATM IV relationships, open-interest concentration strikes and distances, missingness, contract counts, and median relative bid-ask spread for usable positive quotes. Zero or crossed quotes are excluded from spread calculations.

These are descriptive research observations. They are not true dealer gamma exposure, max-pain signals, or bullish/bearish recommendations.

## Sample Maturity

- Fewer than 20 snapshot dates: `collection_only`
- 20 to 59 dates: `preliminary_research`
- 60 or more dates: `eligible_for_feature_evaluation`

Feature evaluation remains a separate, explicitly approved phase.
