from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.data import storage


EVALUATION_REGIMES = ("exploratory_dev", "holdout_candidate", "final_holdout")
DATASET_49_INSPECTED_RUN_IDS = (145, 283, 301, 319, 337, 370)
HOLDOUT_PROTOCOL_VERSION = "holdout_safe_annotation_protocol_v1"


@dataclass(frozen=True)
class EvaluationRegimeLabel:
    model_run_id: int
    evaluation_regime: str
    reason: str


@dataclass(frozen=True)
class HoldoutProtocol:
    protocol_version: str = HOLDOUT_PROTOCOL_VERSION
    dataset_id: int = 49
    dataset_role: str = "exploratory_dev"
    development_scope: str = "Use walk-forward development folds for feature and annotation design."
    confirmation_scope: str = "Use a fresh untouched dataset/window before claiming signal robustness."
    purge_embargo_required: bool = True
    final_holdout_selection_rule: str = (
        "Do not choose annotations, features, targets, models, or thresholds using final-holdout metrics."
    )
    allowed_current_claim: str = "Dataset 49 comparisons are exploratory/dev evidence only."
    forbidden_claim: str = "Do not claim final signal robustness from Dataset 49 after repeated inspection."
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DatasetEvaluationRegime:
    dataset_id: int
    evaluation_regime: str
    strategy: str
    rationale: str
    parent_dataset_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, indent=2)


def validate_evaluation_regime(value: str) -> str:
    regime = str(value or "").strip().lower()
    if regime not in EVALUATION_REGIMES:
        raise ValueError(f"Unknown evaluation regime: {value!r}")
    return regime


def label_model_runs(
    run_ids: list[int] | tuple[int, ...],
    *,
    inspected_run_ids: list[int] | tuple[int, ...] = DATASET_49_INSPECTED_RUN_IDS,
    default_regime: str = "holdout_candidate",
) -> list[EvaluationRegimeLabel]:
    """Return non-mutating evaluation-regime labels for model runs."""
    default = validate_evaluation_regime(default_regime)
    inspected = {int(run_id) for run_id in inspected_run_ids}
    labels: list[EvaluationRegimeLabel] = []
    for run_id in run_ids:
        model_run_id = int(run_id)
        if model_run_id in inspected:
            labels.append(
                EvaluationRegimeLabel(
                    model_run_id=model_run_id,
                    evaluation_regime="exploratory_dev",
                    reason="Run was used during iterative Dataset 49 annotation/model research.",
                )
            )
        else:
            labels.append(
                EvaluationRegimeLabel(
                    model_run_id=model_run_id,
                    evaluation_regime=default,
                    reason="Run is not in the configured inspected-run list; verify before treating as holdout evidence.",
                )
            )
    return labels


def create_evaluation_regime_tables(db_path: str | Path) -> None:
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_evaluation_regimes (
                dataset_id INTEGER PRIMARY KEY,
                evaluation_regime TEXT NOT NULL,
                parent_dataset_id INTEGER,
                strategy TEXT NOT NULL,
                rationale TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_dataset_evaluation_regimes_regime
                ON dataset_evaluation_regimes (evaluation_regime, dataset_id)
            """
        )


def upsert_dataset_evaluation_regime(db_path: str | Path, record: DatasetEvaluationRegime) -> None:
    regime = validate_evaluation_regime(record.evaluation_regime)
    create_evaluation_regime_tables(db_path)
    now = _now_iso()
    with storage.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO dataset_evaluation_regimes (
                dataset_id, evaluation_regime, parent_dataset_id, strategy, rationale,
                metadata_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET
                evaluation_regime = excluded.evaluation_regime,
                parent_dataset_id = excluded.parent_dataset_id,
                strategy = excluded.strategy,
                rationale = excluded.rationale,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                int(record.dataset_id),
                regime,
                None if record.parent_dataset_id is None else int(record.parent_dataset_id),
                record.strategy,
                record.rationale,
                _json_dumps(record.metadata),
                now,
                now,
            ),
        )


def get_dataset_evaluation_regime(db_path: str | Path, dataset_id: int) -> dict[str, Any] | None:
    create_evaluation_regime_tables(db_path)
    with storage.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT dataset_id, evaluation_regime, parent_dataset_id, strategy, rationale,
                   metadata_json, created_at, updated_at
            FROM dataset_evaluation_regimes
            WHERE dataset_id = ?
            """,
            (int(dataset_id),),
        ).fetchone()
    if row is None:
        return None
    output = dict(row)
    output["metadata"] = _json_loads(output.pop("metadata_json"), {}) or {}
    return output


def default_annotation_expansion_plan() -> dict[str, Any]:
    """Pre-registered plan for future annotation imports, with no data mutation."""
    return {
        "plan_version": HOLDOUT_PROTOCOL_VERSION,
        "purpose": "Expand research-only annotation coverage without targeting inspected final-test weaknesses.",
        "dataset_scope": "Dataset 49 remains exploratory/dev only.",
        "ticker_list": "Pre-select before sourcing rows; use broad corporate-equity coverage instead of performance-driven picks.",
        "date_range": "Pre-register before import; do not target windows because Dataset 49 final-test metrics looked weak there.",
        "event_type_target_mix": {
            "legal_regulatory": "20-30%",
            "financing_or_corporate_action": "15-25%",
            "product_customer_partnership": "25-35%",
            "earnings_guidance": "10-20%",
            "neutral_material_updates": "10-20%",
        },
        "sentiment_target_mix": {"positive": "30-45%", "neutral_or_mixed": "25-40%", "negative_or_risk": "20-35%"},
        "source_rules": [
            "Use company IR/newsroom, official press releases, SEC archive links, regulator pages, index-provider announcements, or manually verified credible public news.",
            "Do not scrape Reddit, X/Twitter, forums, or websites.",
            "Do not use paid provider data unless a later phase explicitly approves it.",
            "Do not use future price movement or model results to choose, label, or size annotations.",
        ],
        "max_rows_per_ticker": 8,
        "minimum_fields": ["ticker", "event_date", "available_at", "event_type", "source_url", "evidence_text"],
        "scanner_scoring_effect": 0,
        "active_catalyst_creation": False,
    }


def holdout_protocol_dict(dataset_id: int = 49) -> dict[str, Any]:
    protocol = HoldoutProtocol(
        dataset_id=int(dataset_id),
        notes=[
            "Dataset 49 final test has been repeatedly inspected across Phase 2D model/annotation iterations.",
            "Future Dataset 49 improvements may guide development, but they are not final holdout evidence.",
            "Before claiming robustness, create a fresh untouched holdout or locked walk-forward confirmation protocol.",
        ],
    )
    return {
        "protocol_version": protocol.protocol_version,
        "dataset_id": protocol.dataset_id,
        "dataset_role": protocol.dataset_role,
        "development_scope": protocol.development_scope,
        "confirmation_scope": protocol.confirmation_scope,
        "purge_embargo_required": protocol.purge_embargo_required,
        "final_holdout_selection_rule": protocol.final_holdout_selection_rule,
        "allowed_current_claim": protocol.allowed_current_claim,
        "forbidden_claim": protocol.forbidden_claim,
        "notes": protocol.notes,
    }


def build_holdout_status_artifact(
    db_path: str | Path,
    *,
    dataset_id: int = 49,
    inspected_run_ids: list[int] | tuple[int, ...] = DATASET_49_INSPECTED_RUN_IDS,
) -> dict[str, Any]:
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        dataset = conn.execute(
            "SELECT dataset_id, version, row_count, data_hash, requested_start_date, requested_end_date FROM dataset_builds WHERE dataset_id = ?",
            (int(dataset_id),),
        ).fetchone()
        runs = conn.execute(
            f"""
            SELECT model_run_id, dataset_id, target_column, feature_set_name, model_name, status, created_at, completed_at, config_json
            FROM model_runs
            WHERE model_run_id IN ({','.join('?' for _ in inspected_run_ids)})
            ORDER BY model_run_id
            """,
            tuple(int(run_id) for run_id in inspected_run_ids),
        ).fetchall()

    run_ids = [int(row["model_run_id"]) for row in runs]
    labels = label_model_runs(run_ids, inspected_run_ids=inspected_run_ids)
    label_by_id = {label.model_run_id: label for label in labels}
    run_rows: list[dict[str, Any]] = []
    for row in runs:
        config = _json_loads(row["config_json"], {}) or {}
        label = label_by_id[int(row["model_run_id"])]
        run_rows.append(
            {
                "model_run_id": int(row["model_run_id"]),
                "dataset_id": int(row["dataset_id"]),
                "target_column": row["target_column"],
                "feature_set_name": row["feature_set_name"],
                "model_name": row["model_name"],
                "status": row["status"],
                "phase": config.get("phase"),
                "evaluation_regime": label.evaluation_regime,
                "regime_reason": label.reason,
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
            }
        )

    return {
        "artifact_type": "holdout_status_decision",
        "created_at": _now_iso(),
        "protocol": holdout_protocol_dict(dataset_id),
        "dataset": dict(dataset) if dataset is not None else {"dataset_id": int(dataset_id), "missing": True},
        "inspected_model_runs": run_rows,
        "annotation_expansion_plan": default_annotation_expansion_plan(),
        "decision": {
            "dataset_49_final_test_status": "repeatedly_inspected",
            "dataset_49_future_role": "exploratory_dev",
            "new_dataset_required_for_final_claim": True,
            "candidate_import_this_phase": False,
            "modeling_this_phase": False,
        },
    }


def write_holdout_status_artifact(artifact: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps(artifact), encoding="utf-8")
    return path
