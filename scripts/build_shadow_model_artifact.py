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
from src.modeling.artifacts import (
    ARTIFACT_NAME,
    EXPECTED_DATASET_HASH,
    EXPECTED_DATASET_ID,
    EXPECTED_DATASET_ROWS,
    ArtifactContractError,
    apply_artifact_build,
    dry_run_artifact_build,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the frozen exploratory technical Ridge artifact.")
    parser.add_argument("--dataset-id", type=int, default=EXPECTED_DATASET_ID)
    parser.add_argument("--artifact-name", default=ARTIFACT_NAME)
    parser.add_argument("--db")
    parser.add_argument("--artifact-root", default=str(PROJECT_ROOT / "data" / "model_artifacts"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.artifact_name != ARTIFACT_NAME:
        print(json.dumps({"status": "blocked", "error": f"Artifact name must be {ARTIFACT_NAME}."}, indent=2))
        return 2
    db_path = Path(args.db) if args.db else load_settings().database_file
    try:
        if args.dry_run:
            result = dry_run_artifact_build(
                db_path,
                args.artifact_root,
                dataset_id=args.dataset_id,
                expected_hash=EXPECTED_DATASET_HASH,
                expected_row_count=EXPECTED_DATASET_ROWS,
            )
        else:
            result = apply_artifact_build(
                db_path,
                args.artifact_root,
                project_root=PROJECT_ROOT,
                dataset_id=args.dataset_id,
                expected_hash=EXPECTED_DATASET_HASH,
                expected_row_count=EXPECTED_DATASET_ROWS,
            )
    except ArtifactContractError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
