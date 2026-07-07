from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.annotations.provider_registry import build_provider_readiness_report
from src.data import storage
from src.modeling.annotation_features import build_annotation_coverage_audit, derive_annotation_features
from src.modeling.holdout_maturity import assess_holdout_maturity, build_holdout_extension_plan
from src.modeling.repository import list_model_final_metrics


SCANNER_COMPARE_FIELDS = ("score", "label", "risk_label", "catalyst_score")
MODEL_COMPARE_METRICS = ("rmse", "oos_r2_vs_train_mean", "spearman_ic", "directional_accuracy")
REPORT_SCHEMA_VERSION = "quality_harness_v1"


@dataclass(frozen=True)
class HarnessResult:
    status: str
    summary: dict[str, Any]
    details: dict[str, Any]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, indent=2)


def json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def write_json_artifact(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json_dumps(payload), encoding="utf-8")
    return output


def load_json_artifact(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _clean_scalar(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _scanner_record_key(record: dict[str, Any]) -> str:
    ticker = record.get("ticker") or record.get("symbol") or record.get("Ticker")
    if not ticker:
        raise ValueError(f"Scanner record is missing ticker: {record}")
    return str(ticker).upper()


def _first_present(record: dict[str, Any], *fields: str) -> Any:
    for field in fields:
        if field in record and record[field] is not None:
            return record[field]
    return None


def normalize_scanner_snapshot(raw: Any) -> dict[str, dict[str, Any]]:
    """Normalize scanner snapshot JSON into a ticker-keyed score map."""
    if isinstance(raw, dict) and "results" in raw:
        raw = raw["results"]
    if isinstance(raw, dict):
        records = []
        for ticker, payload in raw.items():
            record = dict(payload or {})
            record.setdefault("ticker", ticker)
            records.append(record)
    elif isinstance(raw, list):
        records = [dict(item or {}) for item in raw]
    else:
        raise ValueError("Scanner snapshot must be a list, ticker dict, or {'results': [...]} payload.")
    normalized: dict[str, dict[str, Any]] = {}
    for record in records:
        ticker = _scanner_record_key(record)
        normalized[ticker] = {
            "ticker": ticker,
            "score": _clean_scalar(_first_present(record, "score", "alpha_score")),
            "label": _clean_scalar(_first_present(record, "label", "action_label")),
            "risk_label": _clean_scalar(_first_present(record, "risk_label", "risk")),
            "catalyst_score": _clean_scalar(record.get("catalyst_score")),
            "raw": record,
        }
    return normalized


def latest_scan_snapshot(db_path: str | Path, run_id: str | None = None) -> dict[str, Any]:
    """Read a scanner snapshot from stored scan_results without refreshing market data."""
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        selected_run_id = run_id
        if selected_run_id is None:
            row = conn.execute("SELECT run_id FROM scan_results ORDER BY datetime(run_at) DESC, id DESC LIMIT 1").fetchone()
            if row is None:
                return {"created_at": now_iso(), "run_id": None, "results": [], "warnings": ["No scan_results rows found."]}
            selected_run_id = str(row["run_id"])
        rows = conn.execute(
            """
            SELECT ticker, score, label, payload_json
            FROM scan_results
            WHERE run_id = ?
            ORDER BY ticker
            """,
            (selected_run_id,),
        ).fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        payload = json_loads(row["payload_json"], {}) or {}
        payload.setdefault("ticker", row["ticker"])
        payload.setdefault("score", row["score"])
        payload.setdefault("label", row["label"])
        results.append(payload)
    return {"created_at": now_iso(), "run_id": selected_run_id, "results": results}


def compare_scanner_snapshots(before: Any, after: Any, score_tolerance: float = 1e-9) -> HarnessResult:
    before_map = normalize_scanner_snapshot(before)
    after_map = normalize_scanner_snapshot(after)
    tickers = sorted(set(before_map) | set(after_map))
    diffs: list[dict[str, Any]] = []
    for ticker in tickers:
        left = before_map.get(ticker)
        right = after_map.get(ticker)
        if left is None or right is None:
            diffs.append({"ticker": ticker, "change": "missing_ticker", "before_present": left is not None, "after_present": right is not None})
            continue
        changes: dict[str, Any] = {}
        before_score = left.get("score")
        after_score = right.get("score")
        try:
            score_delta = float(after_score) - float(before_score)
        except Exception:
            score_delta = None
        if score_delta is not None and abs(score_delta) > score_tolerance:
            changes["score"] = {"before": before_score, "after": after_score, "delta": score_delta}
        for field in ("label", "risk_label", "catalyst_score"):
            if left.get(field) != right.get(field):
                changes[field] = {"before": left.get(field), "after": right.get(field)}
        if changes:
            diffs.append({"ticker": ticker, "change": "field_diff", "fields": changes})
    status = "passed" if not diffs else "warn"
    return HarnessResult(
        status=status,
        summary={
            "scanner_invariance": status,
            "ticker_count_before": len(before_map),
            "ticker_count_after": len(after_map),
            "diff_count": len(diffs),
        },
        details={"diffs": diffs, "score_tolerance": score_tolerance},
    )


def check_dataset_manifest(
    db_path: str | Path,
    dataset_id: int,
    expected_hash: str | None = None,
    expected_row_count: int | None = None,
) -> HarnessResult:
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        row = conn.execute("SELECT * FROM dataset_builds WHERE dataset_id = ?", (int(dataset_id),)).fetchone()
    if row is None:
        return HarnessResult("failed", {"dataset_id": int(dataset_id), "error": "Dataset not found."}, {"violations": ["missing_dataset"]})
    build = dict(row)
    feature_columns = json_loads(build.get("feature_columns_json"), []) or []
    audit_columns = json_loads(build.get("audit_columns_json"), []) or []
    label_columns = json_loads(build.get("label_columns_json"), []) or []
    identifier_columns = json_loads(build.get("identifier_columns_json"), []) or []
    metadata_columns = json_loads(build.get("metadata_columns_json"), []) or []
    manifest = json_loads(build.get("feature_manifest_json"), {}) or {}
    forbidden = set(audit_columns) | set(label_columns) | set(identifier_columns) | set(metadata_columns)
    violations: list[str] = []
    leaked = sorted(set(feature_columns) & forbidden)
    if leaked:
        violations.append(f"Feature columns include forbidden columns: {', '.join(leaked)}")
    for column in feature_columns:
        role_payload = manifest.get(column, {})
        role = role_payload.get("role") if isinstance(role_payload, dict) else role_payload
        if role and role != "model_feature":
            violations.append(f"Feature column {column} has manifest role {role!r}.")
    for column in label_columns:
        role_payload = manifest.get(column, {})
        role = role_payload.get("role") if isinstance(role_payload, dict) else role_payload
        if role and role != "label":
            violations.append(f"Label column {column} has manifest role {role!r}.")
    for column in audit_columns:
        role_payload = manifest.get(column, {})
        role = role_payload.get("role") if isinstance(role_payload, dict) else role_payload
        if role and role != "audit":
            violations.append(f"Audit column {column} has manifest role {role!r}.")
    data_hash = str(build.get("data_hash") or "")
    row_count = int(build.get("row_count") or 0)
    if expected_hash is not None and data_hash != expected_hash:
        violations.append(f"Dataset hash mismatch: expected {expected_hash}, got {data_hash}.")
    if expected_row_count is not None and row_count != int(expected_row_count):
        violations.append(f"Dataset row_count mismatch: expected {expected_row_count}, got {row_count}.")
    status = "passed" if not violations else "failed"
    return HarnessResult(
        status=status,
        summary={
            "dataset_id": int(dataset_id),
            "data_hash": data_hash,
            "row_count": row_count,
            "feature_count": len(feature_columns),
            "audit_count": len(audit_columns),
            "label_count": len(label_columns),
            "violation_count": len(violations),
            "manifest_check": status,
        },
        details={
            "violations": violations,
            "feature_columns": feature_columns,
            "audit_columns": audit_columns,
            "label_columns": label_columns,
            "identifier_columns": identifier_columns,
            "metadata_columns": metadata_columns,
        },
    )


def _metric_dict(db_path: str | Path, model_run_id: int) -> dict[str, Any]:
    frame = list_model_final_metrics(db_path, int(model_run_id))
    if frame.empty:
        return {}
    metrics = frame.iloc[-1].get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def compare_model_run_to_baseline(db_path: str | Path, model_run_id: int, baseline_run_id: int = 145) -> HarnessResult:
    baseline = _metric_dict(db_path, baseline_run_id)
    candidate = _metric_dict(db_path, model_run_id)
    if not baseline:
        return HarnessResult("failed", {"baseline_run_id": int(baseline_run_id), "error": "Baseline metrics not found."}, {})
    if not candidate:
        return HarnessResult("failed", {"model_run_id": int(model_run_id), "error": "Candidate metrics not found."}, {})
    rows: list[dict[str, Any]] = []
    improved = 0
    worsened = 0
    for metric in MODEL_COMPARE_METRICS:
        before = baseline.get(metric)
        after = candidate.get(metric)
        delta = None
        direction = "higher_is_better"
        outcome = "n/a"
        try:
            delta = float(after) - float(before)
            if metric == "rmse":
                direction = "lower_is_better"
                outcome = "improved" if delta < 0 else "worsened" if delta > 0 else "unchanged"
            else:
                outcome = "improved" if delta > 0 else "worsened" if delta < 0 else "unchanged"
        except Exception:
            pass
        if outcome == "improved":
            improved += 1
        elif outcome == "worsened":
            worsened += 1
        rows.append({"metric": metric, "baseline": before, "candidate": after, "delta": delta, "direction": direction, "outcome": outcome})
    return HarnessResult(
        status="passed",
        summary={
            "baseline_run_id": int(baseline_run_id),
            "model_run_id": int(model_run_id),
            "improved_metrics": improved,
            "worsened_metrics": worsened,
        },
        details={"metric_comparison": rows},
    )


def check_annotation_coverage(db_path: str | Path, dataset_id: int = 49) -> HarnessResult:
    audit = build_annotation_coverage_audit(db_path, dataset_id)
    zero_features = [
        row["feature"]
        for row in audit.get("active_observation_counts", [])
        if int(row.get("active_count", 0) or 0) == 0
    ]
    violations = _annotation_future_availability_violations(db_path, dataset_id)
    status = "passed" if not violations else "failed"
    summary = dict(audit.get("summary", {}))
    summary.update(
        {
            "dataset_id": int(dataset_id),
            "zero_coverage_feature_count": len(zero_features),
            "future_availability_violation_count": len(violations),
            "annotation_coverage_check": status,
        }
    )
    return HarnessResult(
        status=status,
        summary=summary,
        details={
            "warnings": audit.get("warnings", []),
            "zero_coverage_features": zero_features,
            "future_availability_violations": violations,
            "coverage_by_ticker": audit.get("coverage_by_ticker", []),
            "coverage_by_fold": audit.get("coverage_by_fold", []),
            "source_quality_signal_coverage": audit.get("source_quality_signal_coverage", []),
            "fold_density_status": audit.get("fold_density_status"),
            "annotation_db_summary": audit.get("annotation_db_summary", {}),
        },
    )


def check_holdout_status(db_path: str | Path, dataset_id: int) -> HarnessResult:
    maturity = assess_holdout_maturity(db_path, dataset_id)
    extension_plan = build_holdout_extension_plan(db_path, dataset_id)
    manifest_ok = int(maturity.get("manifest", {}).get("violation_count", 0) or 0) == 0
    status = "passed" if manifest_ok else "failed"
    summary = {
        "dataset_id": int(dataset_id),
        "evaluation_regime": maturity.get("evaluation_regime"),
        "row_count": maturity.get("row_count"),
        "ticker_count": maturity.get("ticker_count"),
        "date_range": maturity.get("date_range"),
        "label_coverage": maturity.get("label_coverage"),
        "readiness": maturity.get("readiness"),
        "promotion": maturity.get("promotion"),
        "extension_available": extension_plan.get("extension_available"),
        "extension_trading_day_count": extension_plan.get("extension_trading_day_count"),
        "holdout_status_check": status,
    }
    return HarnessResult(
        status=status,
        summary=summary,
        details={
            "maturity": maturity,
            "extension_plan": extension_plan,
        },
    )


def check_provider_readiness() -> HarnessResult:
    report = build_provider_readiness_report()
    violations: list[str] = []
    if report.get("network_calls_would_occur"):
        violations.append("One or more configured research-event providers would make network calls.")
    if report.get("guardrails", {}).get("scanner_scoring_effect") != 0:
        violations.append("Provider readiness guardrail reports a nonzero scanner scoring effect.")
    status = "passed" if not violations else "failed"
    summary = {
        "configured_provider_count": report.get("configured_provider_count"),
        "enabled_provider_count": report.get("enabled_provider_count"),
        "blocked_or_disabled_provider_count": report.get("blocked_or_disabled_provider_count"),
        "requires_api_key_count": report.get("requires_api_key_count"),
        "network_calls_would_occur": report.get("network_calls_would_occur"),
        "provider_readiness_check": status,
        "violation_count": len(violations),
    }
    return HarnessResult(status, summary, {"violations": violations, **report})


def _annotation_future_availability_violations(db_path: str | Path, dataset_id: int) -> list[dict[str, Any]]:
    from src.annotations.repository import list_annotations
    from src.datasets.training_loader import load_training_dataset
    from src.modeling.targets import RAW_TARGET_5_SESSION

    annotations = list_annotations(db_path, limit=None)
    if annotations.empty:
        return []
    training = load_training_dataset(db_path, dataset_id, RAW_TARGET_5_SESSION)
    metadata = training.metadata.copy()
    metadata["trading_date_parsed"] = pd.to_datetime(metadata["trading_date"]).dt.date
    metadata["as_of_parsed"] = pd.to_datetime(metadata["as_of_timestamp"], utc=True, errors="coerce")
    annotations = annotations.copy()
    annotations["event_date_parsed"] = pd.to_datetime(annotations["event_date"], errors="coerce").dt.date
    annotations["available_at_parsed"] = pd.to_datetime(annotations["available_at"], utc=True, errors="coerce")
    violations: list[dict[str, Any]] = []
    for ticker, group in annotations.groupby("ticker", dropna=False):
        ticker = str(ticker).upper()
        event_dates = group["event_date_parsed"].dropna()
        available_times = group["available_at_parsed"].dropna()
        if event_dates.empty or available_times.empty:
            continue
        earliest_event_date = min(event_dates)
        earliest_available = min(available_times)
        pre_availability = metadata[
            metadata["ticker"].astype(str).str.upper().eq(ticker)
            & ((metadata["trading_date_parsed"] < earliest_event_date) | (metadata["as_of_parsed"] < earliest_available))
        ]
        if pre_availability.empty:
            continue
        pre_active = group[
            group["event_date_parsed"].le(pre_availability["trading_date_parsed"].max())
            & group["available_at_parsed"].le(pre_availability["as_of_parsed"].max())
        ]
        if not pre_active.empty:
            violations.append(
                {
                    "ticker": ticker,
                    "earliest_event_date": str(earliest_event_date),
                    "earliest_available_at": str(earliest_available),
                    "pre_availability_annotation_rows": int(len(pre_active)),
                }
            )
    return violations


def streamlit_health_check(url: str = "http://localhost:8501/_stcore/health", timeout_seconds: float = 5.0) -> HarnessResult:
    try:
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status_code = int(response.status)
            body = response.read(64).decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return HarnessResult("failed", {"url": url, "error": str(exc)}, {})
    status = "passed" if 200 <= status_code < 300 else "failed"
    return HarnessResult(status, {"url": url, "status_code": status_code, "health_check": status}, {"body_preview": body})


def build_final_report(
    runtime_status: str,
    tests: str,
    files_changed: list[str],
    artifacts: list[str],
    scanner_invariance: str,
    dataset_model_comparison: str,
    decision: str,
    next_recommendation: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": now_iso(),
        "runtime_status": runtime_status,
        "tests": tests,
        "files_changed": files_changed,
        "artifacts": artifacts,
        "scanner_invariance": scanner_invariance,
        "dataset_model_comparison": dataset_model_comparison,
        "decision": decision,
        "next_recommendation": next_recommendation,
        "extra": extra or {},
    }


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase Quality Report",
        "",
        f"- Runtime status: {report.get('runtime_status', 'n/a')}",
        f"- Tests: {report.get('tests', 'n/a')}",
        f"- Scanner invariance: {report.get('scanner_invariance', 'n/a')}",
        f"- Dataset/model comparison: {report.get('dataset_model_comparison', 'n/a')}",
        f"- Decision: {report.get('decision', 'n/a')}",
        f"- Next recommendation: {report.get('next_recommendation', 'n/a')}",
        "",
        "## Files Changed",
    ]
    for item in report.get("files_changed", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Artifacts")
    for item in report.get("artifacts", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"
