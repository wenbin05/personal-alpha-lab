from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import storage
from src.datasets.models import DatasetBuild, FeatureSnapshot, OutcomeLabel


BUILD_COLUMNS = [
    "dataset_id",
    "version",
    "build_timestamp",
    "requested_start_date",
    "requested_end_date",
    "ticker_universe_json",
    "feature_columns_json",
    "label_definitions_json",
    "audit_columns_json",
    "label_columns_json",
    "identifier_columns_json",
    "metadata_columns_json",
    "feature_manifest_json",
    "row_count",
    "data_hash",
    "warnings_json",
    "export_path",
    "created_at",
    "updated_at",
]


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


def create_dataset_tables(db_path: str | Path) -> None:
    storage.init_db(db_path)


def insert_dataset_build(db_path: str | Path, build: DatasetBuild) -> int:
    create_dataset_tables(db_path)
    now = _now_iso()
    created_at = build.created_at.isoformat(timespec="seconds") if build.created_at else now
    updated_at = build.updated_at.isoformat(timespec="seconds") if build.updated_at else now
    with storage.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO dataset_builds (
                version, build_timestamp, requested_start_date, requested_end_date,
                ticker_universe_json, feature_columns_json, label_definitions_json,
                audit_columns_json, label_columns_json, identifier_columns_json,
                metadata_columns_json, feature_manifest_json, row_count, data_hash,
                warnings_json, export_path, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                build.version,
                build.build_timestamp.isoformat(timespec="seconds"),
                build.requested_start_date.isoformat(),
                build.requested_end_date.isoformat(),
                _json_dumps(build.ticker_universe),
                _json_dumps(build.feature_columns),
                _json_dumps(build.label_definitions),
                _json_dumps(build.audit_columns),
                _json_dumps(build.label_columns),
                _json_dumps(build.identifier_columns),
                _json_dumps(build.metadata_columns),
                _json_dumps(build.feature_manifest),
                int(build.row_count),
                build.data_hash,
                _json_dumps(build.warnings),
                build.export_path,
                created_at,
                updated_at,
            ),
        )
        return int(cursor.lastrowid)


def update_dataset_export_path(db_path: str | Path, dataset_id: int, export_path: str) -> None:
    create_dataset_tables(db_path)
    with storage.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE dataset_builds
            SET export_path = ?, updated_at = ?
            WHERE dataset_id = ?
            """,
            (export_path, _now_iso(), int(dataset_id)),
        )


def update_dataset_build_summary(
    db_path: str | Path,
    dataset_id: int,
    row_count: int,
    data_hash: str,
    warnings: list[str] | None = None,
    feature_columns: list[str] | None = None,
    audit_columns: list[str] | None = None,
    label_columns: list[str] | None = None,
    identifier_columns: list[str] | None = None,
    metadata_columns: list[str] | None = None,
    feature_manifest: dict[str, Any] | None = None,
    export_path: str | None = None,
) -> None:
    create_dataset_tables(db_path)
    assignments = ["row_count = ?", "data_hash = ?", "updated_at = ?"]
    values: list[Any] = [int(row_count), data_hash, _now_iso()]
    if warnings is not None:
        assignments.append("warnings_json = ?")
        values.append(_json_dumps(warnings))
    if feature_columns is not None:
        assignments.append("feature_columns_json = ?")
        values.append(_json_dumps(feature_columns))
    if audit_columns is not None:
        assignments.append("audit_columns_json = ?")
        values.append(_json_dumps(audit_columns))
    if label_columns is not None:
        assignments.append("label_columns_json = ?")
        values.append(_json_dumps(label_columns))
    if identifier_columns is not None:
        assignments.append("identifier_columns_json = ?")
        values.append(_json_dumps(identifier_columns))
    if metadata_columns is not None:
        assignments.append("metadata_columns_json = ?")
        values.append(_json_dumps(metadata_columns))
    if feature_manifest is not None:
        assignments.append("feature_manifest_json = ?")
        values.append(_json_dumps(feature_manifest))
    if export_path is not None:
        assignments.append("export_path = ?")
        values.append(export_path)
    values.append(int(dataset_id))
    with storage.connect(db_path) as conn:
        conn.execute(
            f"""
            UPDATE dataset_builds
            SET {", ".join(assignments)}
            WHERE dataset_id = ?
            """,
            values,
        )


def insert_feature_snapshot(db_path: str | Path, dataset_id: int, snapshot: FeatureSnapshot) -> int:
    create_dataset_tables(db_path)
    created_at = snapshot.created_at.isoformat(timespec="seconds") if snapshot.created_at else _now_iso()
    with storage.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO feature_snapshots (
                dataset_id, ticker, trading_date, as_of_timestamp, feature_version,
                market_regime_json, technical_json, relative_strength_json,
                volume_liquidity_json, catalyst_json, llm_supported_json,
                data_quality_json, features_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id, ticker, trading_date) DO UPDATE SET
                as_of_timestamp = excluded.as_of_timestamp,
                feature_version = excluded.feature_version,
                market_regime_json = excluded.market_regime_json,
                technical_json = excluded.technical_json,
                relative_strength_json = excluded.relative_strength_json,
                volume_liquidity_json = excluded.volume_liquidity_json,
                catalyst_json = excluded.catalyst_json,
                llm_supported_json = excluded.llm_supported_json,
                data_quality_json = excluded.data_quality_json,
                features_json = excluded.features_json
            """,
            (
                int(dataset_id),
                snapshot.ticker.upper(),
                snapshot.trading_date.isoformat(),
                snapshot.as_of_timestamp.isoformat(timespec="seconds"),
                snapshot.feature_version,
                _json_dumps(snapshot.market_regime),
                _json_dumps(snapshot.technical),
                _json_dumps(snapshot.relative_strength),
                _json_dumps(snapshot.volume_liquidity),
                _json_dumps(snapshot.catalyst),
                _json_dumps(snapshot.llm_supported),
                _json_dumps(snapshot.data_quality),
                _json_dumps(snapshot.features),
                created_at,
            ),
        )
        row = conn.execute(
            """
            SELECT snapshot_id
            FROM feature_snapshots
            WHERE dataset_id = ? AND ticker = ? AND trading_date = ?
            """,
            (int(dataset_id), snapshot.ticker.upper(), snapshot.trading_date.isoformat()),
        ).fetchone()
        return int(row["snapshot_id"])


def _feature_snapshot_row(dataset_id: int, snapshot: FeatureSnapshot, created_at: str) -> tuple[Any, ...]:
    return (
        int(dataset_id),
        snapshot.ticker.upper(),
        snapshot.trading_date.isoformat(),
        snapshot.as_of_timestamp.isoformat(timespec="seconds"),
        snapshot.feature_version,
        _json_dumps(snapshot.market_regime),
        _json_dumps(snapshot.technical),
        _json_dumps(snapshot.relative_strength),
        _json_dumps(snapshot.volume_liquidity),
        _json_dumps(snapshot.catalyst),
        _json_dumps(snapshot.llm_supported),
        _json_dumps(snapshot.data_quality),
        _json_dumps(snapshot.features),
        created_at,
    )


def insert_feature_snapshots(
    db_path: str | Path,
    dataset_id: int,
    snapshots: list[FeatureSnapshot],
) -> dict[tuple[str, date], int]:
    """Insert/update snapshots in one bounded transaction and return IDs by ticker/date."""
    if not snapshots:
        return {}
    create_dataset_tables(db_path)
    created_at = _now_iso()
    keys = {(snapshot.ticker.upper(), snapshot.trading_date) for snapshot in snapshots}
    rows = [_feature_snapshot_row(dataset_id, snapshot, created_at) for snapshot in snapshots]
    with storage.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO feature_snapshots (
                dataset_id, ticker, trading_date, as_of_timestamp, feature_version,
                market_regime_json, technical_json, relative_strength_json,
                volume_liquidity_json, catalyst_json, llm_supported_json,
                data_quality_json, features_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id, ticker, trading_date) DO UPDATE SET
                as_of_timestamp = excluded.as_of_timestamp,
                feature_version = excluded.feature_version,
                market_regime_json = excluded.market_regime_json,
                technical_json = excluded.technical_json,
                relative_strength_json = excluded.relative_strength_json,
                volume_liquidity_json = excluded.volume_liquidity_json,
                catalyst_json = excluded.catalyst_json,
                llm_supported_json = excluded.llm_supported_json,
                data_quality_json = excluded.data_quality_json,
                features_json = excluded.features_json
            """,
            rows,
        )
        fetched = conn.execute(
            """
            SELECT snapshot_id, ticker, trading_date
            FROM feature_snapshots
            WHERE dataset_id = ?
            """,
            (int(dataset_id),),
        ).fetchall()
    output: dict[tuple[str, date], int] = {}
    for row in fetched:
        key = (str(row["ticker"]).upper(), pd.to_datetime(row["trading_date"]).date())
        if key in keys:
            output[key] = int(row["snapshot_id"])
    return output


def insert_outcome_labels(db_path: str | Path, labels: list[OutcomeLabel]) -> None:
    if not labels:
        return
    create_dataset_tables(db_path)
    rows = [
        (
            int(label.snapshot_id or 0),
            label.ticker.upper(),
            label.entry_date.isoformat(),
            label.horizon,
            float(label.entry_price),
            label.exit_date.isoformat(),
            float(label.exit_price),
            float(label.forward_return),
            None if label.spy_forward_return is None else float(label.spy_forward_return),
            None if label.excess_return is None else float(label.excess_return),
            label.label_available_at.isoformat(timespec="seconds"),
        )
        for label in labels
        if label.snapshot_id is not None
    ]
    with storage.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO outcome_labels (
                snapshot_id, ticker, entry_date, horizon, entry_price, exit_date, exit_price,
                forward_return, spy_forward_return, excess_return, label_available_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_id, horizon) DO UPDATE SET
                ticker = excluded.ticker,
                entry_date = excluded.entry_date,
                entry_price = excluded.entry_price,
                exit_date = excluded.exit_date,
                exit_price = excluded.exit_price,
                forward_return = excluded.forward_return,
                spy_forward_return = excluded.spy_forward_return,
                excess_return = excluded.excess_return,
                label_available_at = excluded.label_available_at
            """,
            rows,
        )


def list_dataset_builds(db_path: str | Path, limit: int = 25) -> pd.DataFrame:
    create_dataset_tables(db_path)
    with storage.connect(db_path) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT {", ".join(BUILD_COLUMNS)}
            FROM dataset_builds
            ORDER BY datetime(build_timestamp) DESC, dataset_id DESC
            LIMIT ?
            """,
            conn,
            params=(int(limit),),
        )
    if df.empty:
        return pd.DataFrame(columns=BUILD_COLUMNS)
    display = df.copy()
    display["tickers"] = display["ticker_universe_json"].map(lambda value: ", ".join(_json_loads(value, [])))
    display["feature_count"] = display["feature_columns_json"].map(lambda value: len(_json_loads(value, [])))
    display["warnings"] = display["warnings_json"].map(lambda value: "; ".join(_json_loads(value, [])))
    return display


def load_feature_snapshots(db_path: str | Path, dataset_id: int, limit: int | None = None) -> pd.DataFrame:
    create_dataset_tables(db_path)
    limit_clause = "" if limit is None else "LIMIT ?"
    params: tuple[Any, ...] = (int(dataset_id),) if limit is None else (int(dataset_id), int(limit))
    with storage.connect(db_path) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT *
            FROM feature_snapshots
            WHERE dataset_id = ?
            ORDER BY ticker, trading_date
            {limit_clause}
            """,
            conn,
            params=params,
        )
    if df.empty:
        return df
    for column in [
        "market_regime_json",
        "technical_json",
        "relative_strength_json",
        "volume_liquidity_json",
        "catalyst_json",
        "llm_supported_json",
        "data_quality_json",
        "features_json",
    ]:
        df[column.replace("_json", "")] = df[column].map(lambda value: _json_loads(value, {}))
    return df


def load_outcome_labels(db_path: str | Path, dataset_id: int) -> pd.DataFrame:
    create_dataset_tables(db_path)
    with storage.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT o.*
            FROM outcome_labels o
            JOIN feature_snapshots s ON s.snapshot_id = o.snapshot_id
            WHERE s.dataset_id = ?
            ORDER BY o.ticker, o.snapshot_id, o.horizon
            """,
            conn,
            params=(int(dataset_id),),
        )


def _label_map_for_snapshots(labels: pd.DataFrame) -> dict[int, dict[str, Any]]:
    label_map: dict[int, dict[str, Any]] = {}
    if labels.empty:
        return label_map
    for _, label in labels.iterrows():
        snapshot_id = int(label["snapshot_id"])
        horizon = str(label["horizon"])
        label_map.setdefault(snapshot_id, {})
        label_map[snapshot_id][f"label_{horizon}_entry_date"] = label["entry_date"]
        label_map[snapshot_id][f"label_{horizon}_exit_date"] = label["exit_date"]
        label_map[snapshot_id][f"label_{horizon}_forward_return"] = label["forward_return"]
        label_map[snapshot_id][f"label_{horizon}_spy_forward_return"] = label["spy_forward_return"]
        label_map[snapshot_id][f"label_{horizon}_excess_return"] = label["excess_return"]
        label_map[snapshot_id][f"label_{horizon}_available_at"] = label["label_available_at"]
    return label_map


def _flatten_snapshot_record(snapshot: Any, labels: dict[str, Any]) -> dict[str, Any]:
    features = _json_loads(snapshot["features_json"], {}) or {}
    return {
        "snapshot_id": int(snapshot["snapshot_id"]),
        "dataset_id": int(snapshot["dataset_id"]),
        "ticker": snapshot["ticker"],
        "trading_date": snapshot["trading_date"],
        "as_of_timestamp": snapshot["as_of_timestamp"],
        **features,
        **labels,
    }


def iter_flattened_dataset_rows(
    db_path: str | Path,
    dataset_id: int,
    limit: int | None = None,
) -> tuple[list[str], Any]:
    """Return deterministic flattened columns and a factory for streaming rows.

    The factory avoids forcing callers to keep the whole flattened dataset in
    memory when they only need a hash or export stream.
    """
    create_dataset_tables(db_path)
    labels = load_outcome_labels(db_path, dataset_id)
    label_map = _label_map_for_snapshots(labels)
    limit_clause = "" if limit is None else "LIMIT ?"
    params: tuple[Any, ...] = (int(dataset_id),) if limit is None else (int(dataset_id), int(limit))

    def row_iter():
        with storage.connect(db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT snapshot_id, dataset_id, ticker, trading_date, as_of_timestamp, features_json
                FROM feature_snapshots
                WHERE dataset_id = ?
                ORDER BY ticker, trading_date
                {limit_clause}
                """,
                params,
            )
            for snapshot in rows:
                yield _flatten_snapshot_record(snapshot, label_map.get(int(snapshot["snapshot_id"]), {}))

    seen: set[str] = set()
    columns: list[str] = []
    for row in row_iter():
        for column in row:
            if column not in seen:
                seen.add(column)
                columns.append(column)
    return columns, row_iter


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _infer_stream_dtypes(row_iter_factory: Any, columns: list[str]) -> dict[str, str]:
    stats: dict[str, dict[str, Any]] = {
        column: {"missing": False, "kinds": set()} for column in columns
    }
    for row in row_iter_factory():
        for column in columns:
            value = row.get(column)
            if _is_missing_value(value):
                stats[column]["missing"] = True
                continue
            if isinstance(value, bool):
                stats[column]["kinds"].add("bool")
            elif isinstance(value, int):
                stats[column]["kinds"].add("int")
            elif isinstance(value, float):
                stats[column]["kinds"].add("float")
            else:
                stats[column]["kinds"].add("object")

    dtypes: dict[str, str] = {}
    for column, info in stats.items():
        kinds = set(info["kinds"])
        if not kinds or "object" in kinds:
            continue
        if kinds == {"bool"} and not info["missing"]:
            dtypes[column] = "bool"
        elif kinds.issubset({"int", "float", "bool"}):
            dtypes[column] = "float64" if info["missing"] or "float" in kinds else "int64"
    return dtypes


def _apply_stream_dtypes(frame: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    for column, dtype in dtypes.items():
        if column not in frame.columns:
            continue
        if dtype in {"float64", "int64"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = frame[column].astype(dtype)
        elif dtype == "bool":
            frame[column] = frame[column].astype(bool)
    return frame


def flatten_saved_dataset(db_path: str | Path, dataset_id: int, limit: int | None = None) -> pd.DataFrame:
    snapshots = load_feature_snapshots(db_path, dataset_id, limit=limit)
    if snapshots.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    labels = load_outcome_labels(db_path, dataset_id)
    label_map = _label_map_for_snapshots(labels)
    for _, snapshot in snapshots.iterrows():
        features = snapshot.get("features") or {}
        rows.append(
            {
                "snapshot_id": int(snapshot["snapshot_id"]),
                "dataset_id": int(snapshot["dataset_id"]),
                "ticker": snapshot["ticker"],
                "trading_date": snapshot["trading_date"],
                "as_of_timestamp": snapshot["as_of_timestamp"],
                **features,
                **label_map.get(int(snapshot["snapshot_id"]), {}),
            }
        )
    return pd.DataFrame(rows)


def stream_saved_dataset_export_and_hash(
    db_path: str | Path,
    dataset_id: int,
    export_path: str | Path | None = None,
    chunk_size: int = 1000,
) -> dict[str, Any]:
    columns, row_iter_factory = iter_flattened_dataset_rows(db_path, dataset_id)
    if not columns:
        return {
            "row_count": 0,
            "data_hash": hashlib.sha256(b"empty").hexdigest(),
            "columns": [],
            "export_path": None,
        }
    export_columns = columns
    stable_columns = sorted(column for column in columns if column not in {"snapshot_id", "dataset_id"})
    dtypes = _infer_stream_dtypes(row_iter_factory, columns)
    hasher = hashlib.sha256()
    row_count = 0
    export_target = Path(export_path) if export_path is not None else None
    if export_target is not None:
        export_target.parent.mkdir(parents=True, exist_ok=True)
        if export_target.exists():
            export_target.unlink()

    chunk: list[dict[str, Any]] = []
    wrote_hash_header = False
    wrote_export_header = False

    def flush() -> None:
        nonlocal chunk, wrote_hash_header, wrote_export_header, row_count
        if not chunk:
            return
        frame = _apply_stream_dtypes(pd.DataFrame(chunk), dtypes)
        row_count += int(len(frame))
        stable = frame.reindex(columns=stable_columns)
        payload = stable.to_csv(index=False, na_rep="", header=not wrote_hash_header)
        hasher.update(payload.encode("utf-8"))
        wrote_hash_header = True
        if export_target is not None:
            export_frame = frame.reindex(columns=export_columns)
            export_frame.to_csv(
                export_target,
                mode="a",
                index=False,
                header=not wrote_export_header,
            )
            wrote_export_header = True
        chunk = []

    for row in row_iter_factory():
        chunk.append(row)
        if len(chunk) >= chunk_size:
            flush()
    flush()
    return {
        "row_count": row_count,
        "data_hash": hasher.hexdigest(),
        "columns": export_columns,
        "export_path": str(export_target) if export_target is not None else None,
    }
