#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.options_research.snapshots import OptionsSnapshotError, collect_options_snapshots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect immutable, research-only options-chain snapshots.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Plan only; makes no network calls or database changes.")
    mode.add_argument("--apply", action="store_true", help="Authorize yfinance collection and immutable persistence.")
    parser.add_argument("--tickers", nargs="+", help="Ticker universe. Defaults to the eight-symbol pilot.")
    parser.add_argument("--max-expirations", type=int, default=2)
    parser.add_argument("--db", help="SQLite path. Defaults to configured DATABASE_PATH.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    db_path = Path(args.db) if args.db else load_settings().database_file
    try:
        report = collect_options_snapshots(
            db_path,
            apply=bool(args.apply),
            tickers=args.tickers,
            max_expirations=args.max_expirations,
        )
    except OptionsSnapshotError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2, sort_keys=True))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"Unexpected options collection failure: {exc}"}, indent=2))
        return 3
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
