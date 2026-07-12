#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotations.document_enrichment import run_company_ir_document_enrichment
from src.config import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run or explicitly apply manual company IR SourceDocument enrichment."
    )
    parser.add_argument("--input", required=True, help="Manually completed enrichment CSV.")
    parser.add_argument("--db", help="SQLite database path; defaults to configured database.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Plan only; this is the default.")
    mode.add_argument("--apply", action="store_true", help="Back up the database and apply valid rows.")
    parser.add_argument("--backup-dir", help="Optional directory for the timestamped apply backup.")
    parser.add_argument("--report-output", help="Optional JSON report path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    db_path = Path(args.db) if args.db else load_settings().database_file
    result = run_company_ir_document_enrichment(
        db_path,
        args.input,
        apply=bool(args.apply),
        backup_dir=args.backup_dir,
    )
    if args.report_output:
        output_path = Path(args.report_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if int(result["summary"]["errors"]) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
