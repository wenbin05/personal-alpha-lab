#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.annotations.document_coverage import QUEUE_COLUMNS, write_enrichment_queue_csv
from src.quality.harness import (
    build_final_report,
    check_annotation_coverage,
    check_dataset_manifest,
    check_document_coverage,
    check_holdout_status,
    check_model_artifact,
    check_options_status,
    check_portfolio_shadow_status,
    check_shadow_status,
    check_provider_readiness,
    compare_model_run_to_baseline,
    compare_scanner_snapshots,
    latest_scan_snapshot,
    load_json_artifact,
    report_markdown,
    streamlit_health_check,
    write_json_artifact,
)


def _db_path(value: str | None) -> Path:
    if value:
        return Path(value)
    return load_settings().database_file


def _print_result(payload: dict) -> None:
    print(write_json_string(payload))


def write_json_string(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2, default=str, sort_keys=True)


def scanner_snapshot(args: argparse.Namespace) -> int:
    payload = latest_scan_snapshot(_db_path(args.db), run_id=args.run_id)
    if args.output:
        write_json_artifact(args.output, payload)
    _print_result(payload)
    return 0


def scanner_compare(args: argparse.Namespace) -> int:
    before = load_json_artifact(args.before)
    after = load_json_artifact(args.after)
    result = compare_scanner_snapshots(before, after, score_tolerance=args.score_tolerance)
    payload = {"status": result.status, "summary": result.summary, "details": result.details}
    if args.output:
        write_json_artifact(args.output, payload)
    _print_result(payload)
    if args.fail_on_change and result.summary.get("diff_count", 0):
        return 2
    return 0


def dataset_check(args: argparse.Namespace) -> int:
    result = check_dataset_manifest(
        _db_path(args.db),
        args.dataset_id,
        expected_hash=args.expected_hash,
        expected_row_count=args.expected_row_count,
    )
    payload = {"status": result.status, "summary": result.summary, "details": result.details}
    if args.output:
        write_json_artifact(args.output, payload)
    _print_result(payload)
    return 0 if result.status == "passed" else 2


def model_compare(args: argparse.Namespace) -> int:
    rows = []
    status = "passed"
    for model_run_id in args.model_run_id:
        result = compare_model_run_to_baseline(_db_path(args.db), model_run_id, baseline_run_id=args.baseline_run_id)
        rows.append({"model_run_id": model_run_id, "status": result.status, "summary": result.summary, "details": result.details})
        if result.status != "passed":
            status = "failed"
    payload = {"status": status, "baseline_run_id": args.baseline_run_id, "comparisons": rows}
    if args.output:
        write_json_artifact(args.output, payload)
    _print_result(payload)
    return 0 if status == "passed" else 2


def annotation_coverage(args: argparse.Namespace) -> int:
    result = check_annotation_coverage(_db_path(args.db), dataset_id=args.dataset_id)
    payload = {"status": result.status, "summary": result.summary, "details": result.details}
    if args.output:
        write_json_artifact(args.output, payload)
    _print_result(payload)
    return 0 if result.status == "passed" else 2


def holdout_status(args: argparse.Namespace) -> int:
    result = check_holdout_status(_db_path(args.db), dataset_id=args.dataset_id)
    payload = {"status": result.status, "summary": result.summary, "details": result.details}
    if args.output:
        write_json_artifact(args.output, payload)
    _print_result(payload)
    return 0 if result.status == "passed" else 2


def model_artifact_check(args: argparse.Namespace) -> int:
    result = check_model_artifact(_db_path(args.db), artifact_id=args.artifact_id)
    payload = {"status": result.status, "summary": result.summary, "details": result.details}
    if args.output:
        write_json_artifact(args.output, payload)
    _print_result(payload)
    return 0 if result.status == "passed" else 2


def shadow_status(args: argparse.Namespace) -> int:
    result = check_shadow_status(_db_path(args.db), artifact_id=args.artifact_id)
    payload = {"status": result.status, "summary": result.summary, "details": result.details}
    if args.output:
        write_json_artifact(args.output, payload)
    _print_result(payload)
    return 0 if result.status == "passed" else 2


def options_status(args: argparse.Namespace) -> int:
    result = check_options_status(_db_path(args.db))
    payload = {"status": result.status, "summary": result.summary, "details": result.details}
    if args.output:
        write_json_artifact(args.output, payload)
    _print_result(payload)
    return 0 if result.status == "passed" else 2


def portfolio_shadow_status(args: argparse.Namespace) -> int:
    result = check_portfolio_shadow_status(_db_path(args.db))
    payload = {"status": result.status, "summary": result.summary, "details": result.details}
    if args.output:
        write_json_artifact(args.output, payload)
    _print_result(payload)
    return 0 if result.status == "passed" else 2


def health_check(args: argparse.Namespace) -> int:
    result = streamlit_health_check(args.url, timeout_seconds=args.timeout)
    payload = {"status": result.status, "summary": result.summary, "details": result.details}
    if args.output:
        write_json_artifact(args.output, payload)
    _print_result(payload)
    return 0 if result.status == "passed" else 2


def provider_readiness(args: argparse.Namespace) -> int:
    result = check_provider_readiness()
    payload = {"status": result.status, "summary": result.summary, "details": result.details}
    if args.output:
        write_json_artifact(args.output, payload)
    _print_result(payload)
    return 0 if result.status == "passed" else 2


def document_coverage(args: argparse.Namespace) -> int:
    result = check_document_coverage(_db_path(args.db), provider=args.provider)
    queue_rows = result.details.get("_queue_rows", [])
    details = {key: value for key, value in result.details.items() if key != "_queue_rows"}
    payload = {"status": result.status, "summary": result.summary, "details": details}
    if args.queue_output:
        queue = pd.DataFrame(queue_rows).reindex(columns=QUEUE_COLUMNS)
        queue_path = write_enrichment_queue_csv(queue, args.queue_output)
        payload["queue_output"] = str(queue_path)
    if args.output:
        artifact_path = write_json_artifact(args.output, payload)
        payload["output"] = str(artifact_path)
    _print_result(payload)
    return 0 if result.status == "passed" else 2


def final_report(args: argparse.Namespace) -> int:
    report = build_final_report(
        runtime_status=args.runtime_status,
        tests=args.tests,
        files_changed=args.files_changed or [],
        artifacts=args.artifacts or [],
        scanner_invariance=args.scanner_invariance,
        dataset_model_comparison=args.dataset_model_comparison,
        decision=args.decision,
        next_recommendation=args.next_recommendation,
    )
    if args.output_json:
        write_json_artifact(args.output_json, report)
    markdown = report_markdown(report)
    if args.output_markdown:
        Path(args.output_markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_markdown).write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


def template_path(_args: argparse.Namespace) -> int:
    path = PROJECT_ROOT / "docs" / "templates" / "research_annotations_template.csv"
    print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Personal Alpha Lab quality harness.")
    parser.add_argument("--db", help="SQLite database path. Defaults to configured DATABASE_PATH.")
    sub = parser.add_subparsers(dest="command", required=True)

    cmd = sub.add_parser("scanner-snapshot", help="Read latest stored scanner results without refreshing market data.")
    cmd.add_argument("--run-id")
    cmd.add_argument("--output")
    cmd.set_defaults(func=scanner_snapshot)

    cmd = sub.add_parser("scanner-compare", help="Compare before/after scanner snapshot JSON artifacts.")
    cmd.add_argument("--before", required=True)
    cmd.add_argument("--after", required=True)
    cmd.add_argument("--output")
    cmd.add_argument("--score-tolerance", type=float, default=1e-9)
    cmd.add_argument("--fail-on-change", action="store_true")
    cmd.set_defaults(func=scanner_compare)

    cmd = sub.add_parser("dataset-check", help="Check dataset manifest roles and optional hash/row count.")
    cmd.add_argument("--dataset-id", type=int, required=True)
    cmd.add_argument("--expected-hash")
    cmd.add_argument("--expected-row-count", type=int)
    cmd.add_argument("--output")
    cmd.set_defaults(func=dataset_check)

    cmd = sub.add_parser("model-compare", help="Compare model runs against a baseline run.")
    cmd.add_argument("--model-run-id", type=int, nargs="+", required=True)
    cmd.add_argument("--baseline-run-id", type=int, default=145)
    cmd.add_argument("--output")
    cmd.set_defaults(func=model_compare)

    cmd = sub.add_parser("annotation-coverage", help="Audit research annotation coverage for a dataset.")
    cmd.add_argument("--dataset-id", type=int, default=49)
    cmd.add_argument("--output")
    cmd.set_defaults(func=annotation_coverage)

    cmd = sub.add_parser("holdout-status", help="Report holdout maturity, promotion gates, and cache-only extension availability.")
    cmd.add_argument("--dataset-id", type=int, required=True)
    cmd.add_argument("--output")
    cmd.set_defaults(func=holdout_status)

    cmd = sub.add_parser("model-artifact-check", help="Verify a registered frozen model artifact and deterministic replay fixture.")
    cmd.add_argument("--artifact-id", required=True)
    cmd.add_argument("--output")
    cmd.set_defaults(func=model_artifact_check)

    cmd = sub.add_parser("shadow-status", help="Audit immutable shadow prediction runs and forward-sample maturity.")
    cmd.add_argument("--artifact-id")
    cmd.add_argument("--output")
    cmd.set_defaults(func=shadow_status)

    cmd = sub.add_parser("options-status", help="Audit prospective options snapshot coverage and integrity.")
    cmd.add_argument("--output")
    cmd.set_defaults(func=options_status)

    cmd = sub.add_parser("portfolio-shadow-status", help="Audit frozen-policy shadow portfolio cohorts and maturity.")
    cmd.add_argument("--output")
    cmd.set_defaults(func=portfolio_shadow_status)

    cmd = sub.add_parser("health-check", help="Check Streamlit health endpoint without stopping the server.")
    cmd.add_argument("--url", default="http://localhost:8501/_stcore/health")
    cmd.add_argument("--timeout", type=float, default=5.0)
    cmd.add_argument("--output")
    cmd.set_defaults(func=health_check)

    cmd = sub.add_parser("provider-readiness", help="Report research-event provider readiness without making provider calls.")
    cmd.add_argument("--output")
    cmd.set_defaults(func=provider_readiness)

    cmd = sub.add_parser("document-coverage", help="Audit candidate-to-document coverage using read-only SQLite access.")
    cmd.add_argument("--provider", default="company_ir_press_release")
    cmd.add_argument("--output")
    cmd.add_argument("--queue-output")
    cmd.set_defaults(func=document_coverage)

    cmd = sub.add_parser("final-report", help="Create a consistent phase report artifact.")
    cmd.add_argument("--runtime-status", required=True)
    cmd.add_argument("--tests", required=True)
    cmd.add_argument("--scanner-invariance", required=True)
    cmd.add_argument("--dataset-model-comparison", required=True)
    cmd.add_argument("--decision", required=True)
    cmd.add_argument("--next-recommendation", required=True)
    cmd.add_argument("--files-changed", nargs="*")
    cmd.add_argument("--artifacts", nargs="*")
    cmd.add_argument("--output-json")
    cmd.add_argument("--output-markdown")
    cmd.set_defaults(func=final_report)

    cmd = sub.add_parser("annotation-template-path", help="Print the tracked synthetic annotation CSV template path.")
    cmd.set_defaults(func=template_path)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
