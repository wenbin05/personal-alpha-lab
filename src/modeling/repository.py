from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import storage


MODEL_VERSION = "baseline_v1"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def create_model_tables(db_path: str | Path) -> None:
    storage.init_db(db_path)


def insert_model_run(
    db_path: str | Path,
    dataset_id: int,
    dataset_hash: str,
    target_column: str,
    target_horizon: str,
    task: str,
    feature_set_name: str,
    model_name: str,
    config: dict[str, Any],
    split_config: dict[str, Any],
    feature_columns: list[str],
) -> int:
    create_model_tables(db_path)
    now = _now_iso()
    with storage.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO model_runs (
                dataset_id, dataset_hash, target_column, target_horizon, task,
                feature_set_name, model_name, model_version, config_json,
                split_config_json, feature_columns_json, status, warnings_json,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', '[]', ?, ?)
            """,
            (
                int(dataset_id),
                dataset_hash,
                target_column,
                target_horizon,
                task,
                feature_set_name,
                model_name,
                MODEL_VERSION,
                _json_dumps(config),
                _json_dumps(split_config),
                _json_dumps(feature_columns),
                now,
                now,
            ),
        )
        return int(cursor.lastrowid)


def complete_model_run(db_path: str | Path, model_run_id: int, status: str = "completed", warnings: list[str] | None = None) -> None:
    create_model_tables(db_path)
    now = _now_iso()
    with storage.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE model_runs
            SET status = ?, warnings_json = ?, completed_at = ?, updated_at = ?
            WHERE model_run_id = ?
            """,
            (status, _json_dumps(warnings or []), now, now, int(model_run_id)),
        )


def insert_fold_metric(
    db_path: str | Path,
    model_run_id: int,
    fold_name: str,
    split_name: str,
    train_start_date: Any,
    train_end_date: Any,
    eval_start_date: Any,
    eval_end_date: Any,
    train_rows: int,
    eval_rows: int,
    metrics: dict[str, Any],
) -> int:
    create_model_tables(db_path)
    now = _now_iso()
    with storage.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO model_fold_metrics (
                model_run_id, fold_name, split_name, train_start_date, train_end_date,
                eval_start_date, eval_end_date, train_rows, eval_rows, metrics_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(model_run_id),
                fold_name,
                split_name,
                None if train_start_date is None else str(train_start_date),
                None if train_end_date is None else str(train_end_date),
                None if eval_start_date is None else str(eval_start_date),
                None if eval_end_date is None else str(eval_end_date),
                int(train_rows),
                int(eval_rows),
                _json_dumps(metrics),
                now,
            ),
        )
        return int(cursor.lastrowid)


def insert_final_metric(db_path: str | Path, model_run_id: int, split_name: str, metrics: dict[str, Any]) -> int:
    create_model_tables(db_path)
    now = _now_iso()
    with storage.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO model_final_metrics (model_run_id, split_name, metrics_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (int(model_run_id), split_name, _json_dumps(metrics), now),
        )
        return int(cursor.lastrowid)


def insert_predictions(db_path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    create_model_tables(db_path)
    now = _now_iso()
    payload = [
        (
            int(row["model_run_id"]),
            int(row["snapshot_id"]),
            str(row["ticker"]).upper(),
            str(row["snapshot_date"]),
            str(row["target_horizon"]),
            str(row["split_name"]),
            str(row["fold_name"]),
            row.get("y_true"),
            row.get("y_pred"),
            row.get("y_pred_label"),
            row.get("y_score"),
            str(row["feature_set_name"]),
            str(row["model_name"]),
            now,
        )
        for row in rows
    ]
    with storage.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO model_predictions (
                model_run_id, snapshot_id, ticker, snapshot_date, target_horizon,
                split_name, fold_name, y_true, y_pred, y_pred_label, y_score,
                feature_set_name, model_name, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )


def list_model_runs(db_path: str | Path, limit: int = 100) -> pd.DataFrame:
    create_model_tables(db_path)
    with storage.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT *
            FROM model_runs
            ORDER BY datetime(created_at) DESC, model_run_id DESC
            LIMIT ?
            """,
            conn,
            params=(int(limit),),
        )


def list_model_fold_metrics(db_path: str | Path, model_run_id: int) -> pd.DataFrame:
    create_model_tables(db_path)
    with storage.connect(db_path) as conn:
        frame = pd.read_sql_query(
            """
            SELECT *
            FROM model_fold_metrics
            WHERE model_run_id = ?
            ORDER BY metric_id
            """,
            conn,
            params=(int(model_run_id),),
        )
    if not frame.empty:
        frame["metrics"] = frame["metrics_json"].map(lambda value: _json_loads(value, {}))
    return frame


def list_model_final_metrics(db_path: str | Path, model_run_id: int) -> pd.DataFrame:
    create_model_tables(db_path)
    with storage.connect(db_path) as conn:
        frame = pd.read_sql_query(
            """
            SELECT *
            FROM model_final_metrics
            WHERE model_run_id = ?
            ORDER BY final_metric_id
            """,
            conn,
            params=(int(model_run_id),),
        )
    if not frame.empty:
        frame["metrics"] = frame["metrics_json"].map(lambda value: _json_loads(value, {}))
    return frame


def list_model_predictions(db_path: str | Path, model_run_id: int, limit: int = 500) -> pd.DataFrame:
    create_model_tables(db_path)
    with storage.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT *
            FROM model_predictions
            WHERE model_run_id = ?
            ORDER BY split_name, fold_name, snapshot_date, ticker
            LIMIT ?
            """,
            conn,
            params=(int(model_run_id), int(limit)),
        )

