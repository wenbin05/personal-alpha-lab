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
from src.modeling.daily_shadow_cycle import (
    DEFAULT_SHADOW_ARTIFACT_ID,
    DailyShadowCycleError,
    run_daily_shadow_cycle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one guarded daily exploratory shadow cycle.")
    parser.add_argument("--artifact-id", default=DEFAULT_SHADOW_ARTIFACT_ID)
    parser.add_argument("--db")
    parser.add_argument("--output")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--refresh-market-data",
        action="store_true",
        help="Authorize bounded yfinance refreshes; effective only with --apply.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    db_path = Path(args.db) if args.db else load_settings().database_file
    try:
        report = run_daily_shadow_cycle(
            db_path,
            artifact_id=args.artifact_id,
            apply=bool(args.apply),
            refresh_market_data=bool(args.refresh_market_data),
        )
    except DailyShadowCycleError as exc:
        report = {"status": "failed", "error": str(exc)}
        exit_code = exc.exit_code
    except Exception as exc:
        report = {"status": "failed", "error": str(exc)}
        exit_code = 5
    else:
        exit_code = 0 if report.get("status") not in {"failed"} else 4
    rendered = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
