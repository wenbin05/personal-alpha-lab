#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.modeling.shadow_portfolios import (
    ShadowPortfolioError,
    apply_outcomes,
    create_cohort,
    dry_run_cohort,
    dry_run_outcomes,
    policy_registration_plan,
    register_policy,
)


def _add_mode(parser: argparse.ArgumentParser) -> None:
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage immutable research-only shadow portfolio cohorts.")
    parser.add_argument("--db", help="SQLite path. Defaults to configured DATABASE_PATH.")
    commands = parser.add_subparsers(dest="command", required=True)

    register = commands.add_parser("register-policy", help="Register the one frozen prospective portfolio policy.")
    _add_mode(register)

    cohort = commands.add_parser("create-cohort", help="Select a prospective cohort from one immutable shadow run.")
    cohort.add_argument("--prediction-run-id", type=int, required=True)
    _add_mode(cohort)

    outcomes = commands.add_parser("update-outcomes", help="Append newly matured cache-only cohort outcomes.")
    outcomes.add_argument("--cohort-id", type=int)
    _add_mode(outcomes)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    db_path = Path(args.db) if args.db else load_settings().database_file
    try:
        if args.command == "register-policy":
            result = policy_registration_plan(db_path) if args.dry_run else register_policy(db_path)
        elif args.command == "create-cohort":
            result = (
                dry_run_cohort(db_path, args.prediction_run_id)
                if args.dry_run
                else create_cohort(db_path, args.prediction_run_id)
            )
        else:
            result = (
                dry_run_outcomes(db_path, cohort_id=args.cohort_id)
                if args.dry_run
                else apply_outcomes(db_path, cohort_id=args.cohort_id)
            )
    except (ShadowPortfolioError, ValueError, sqlite3.IntegrityError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
