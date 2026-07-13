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
from src.modeling.shadow_predictions import (
    ShadowPredictionError,
    apply_shadow_prediction,
    dry_run_shadow_prediction,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cache-only frozen exploratory shadow inference.")
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--as-of", required=True, help="Completed U.S. trading session in YYYY-MM-DD format.")
    parser.add_argument("--db")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    db_path = Path(args.db) if args.db else load_settings().database_file
    try:
        result = (
            dry_run_shadow_prediction(db_path, args.artifact_id, args.as_of)
            if args.dry_run
            else apply_shadow_prediction(db_path, args.artifact_id, args.as_of)
        )
    except ShadowPredictionError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
