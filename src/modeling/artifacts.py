from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.data import storage
from src.datasets.training_loader import TrainingDataset
from src.modeling.feature_sets import TECHNICAL_CORE_CANDIDATES, select_feature_columns
from src.modeling.preprocessing import MatrixPreprocessor


EXPECTED_DATASET_ID = 49
EXPECTED_DATASET_ROWS = 19_495
EXPECTED_DATASET_HASH = "e9523e1134b7eb32b142cb628d51bde76d5a6d139f4be2aba2545f3ca4416184"
TARGET_COLUMN = "label_5_session_excess_return"
ARTIFACT_NAME = "shadow_ridge_technical_v1"
ARTIFACT_VERSION = "1"
MODEL_PARAMETERS = {"alpha": 1.0, "fit_intercept": True, "solver": "auto"}
REPLAY_TOLERANCE = 1e-12
DESIGN_REFERENCE_RUN_IDS = [145, 283]


class ArtifactContractError(ValueError):
    pass


class ArtifactIntegrityError(ValueError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _dependency_versions() -> dict[str, str]:
    names = {
        "pandas": "pandas",
        "numpy": "numpy",
        "scikit_learn": "scikit-learn",
        "joblib": "joblib",
    }
    versions = {"python": sys.version.split()[0]}
    for key, package in names.items():
        try:
            versions[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = "unavailable"
    return versions


def _git_state(project_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=project_root, check=True, capture_output=True, text=True
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True


def _sanitize_frame(frame: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    clean = frame.copy()
    for column in numeric_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return clean


@dataclass
class ExecutableShadowArtifact:
    feature_columns: list[str]
    preprocessor: MatrixPreprocessor
    model: Ridge
    artifact_id: str
    target_column: str
    winsor_lower: float
    winsor_upper: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        actual = list(frame.columns)
        if actual != self.feature_columns:
            missing = [column for column in self.feature_columns if column not in actual]
            extra = [column for column in actual if column not in self.feature_columns]
            if missing:
                raise ArtifactContractError(f"Missing required feature columns: {', '.join(missing)}")
            if extra:
                raise ArtifactContractError(f"Unexpected feature columns: {', '.join(extra)}")
            raise ArtifactContractError("Feature order does not match the frozen artifact contract.")
        clean = _sanitize_frame(frame, self.preprocessor.numeric_columns)
        matrix = self.preprocessor.transform(clean)
        return np.asarray(self.model.predict(matrix), dtype=float)


@dataclass(frozen=True)
class ArtifactContract:
    dataset_id: int
    dataset_hash: str
    dataset_row_count: int
    feature_columns: list[str]
    feature_manifest_hash: str
    target_column: str
    training_row_count: int
    excluded_row_count: int
    training_start: str
    training_cutoff: str
    training_row_identifier_hash: str
    universe: list[str]
    universe_hash: str
    winsor_lower: float
    winsor_upper: float
    numeric_columns: list[str]
    categorical_columns: list[str]
    dependencies: dict[str, str]
    artifact_id: str
    specification_hash: str
    intended_artifact_path: str
    duplicate_artifact_id: str | None = None


def _read_dataset_build(db_path: str | Path, dataset_id: int) -> dict[str, Any]:
    with storage.connect(db_path) as conn:
        row = conn.execute("SELECT * FROM dataset_builds WHERE dataset_id = ?", (int(dataset_id),)).fetchone()
    if row is None:
        raise ArtifactContractError(f"Frozen Dataset {dataset_id} is unavailable.")
    return dict(row)


def _readonly_connect(db_path: str | Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _frozen_rows_readonly(db_path: str | Path, dataset_id: int) -> tuple[list[str], list[dict[str, Any]]]:
    with _readonly_connect(db_path) as conn:
        label_rows = conn.execute(
            """
            SELECT o.* FROM outcome_labels o
            JOIN feature_snapshots s ON s.snapshot_id = o.snapshot_id
            WHERE s.dataset_id = ?
            ORDER BY o.ticker, o.snapshot_id, o.horizon
            """,
            (int(dataset_id),),
        ).fetchall()
        labels: dict[int, dict[str, Any]] = {}
        for label in label_rows:
            snapshot_id = int(label["snapshot_id"])
            horizon = str(label["horizon"])
            values = labels.setdefault(snapshot_id, {})
            values[f"label_{horizon}_entry_date"] = label["entry_date"]
            values[f"label_{horizon}_exit_date"] = label["exit_date"]
            values[f"label_{horizon}_forward_return"] = label["forward_return"]
            values[f"label_{horizon}_spy_forward_return"] = label["spy_forward_return"]
            values[f"label_{horizon}_excess_return"] = label["excess_return"]
            values[f"label_{horizon}_available_at"] = label["label_available_at"]
        snapshots = conn.execute(
            """
            SELECT snapshot_id, dataset_id, ticker, trading_date, as_of_timestamp, features_json
            FROM feature_snapshots
            WHERE dataset_id = ?
            ORDER BY ticker, trading_date
            """,
            (int(dataset_id),),
        ).fetchall()
        rows: list[dict[str, Any]] = []
        columns: list[str] = []
        seen: set[str] = set()
        for snapshot in snapshots:
            features = json.loads(str(snapshot["features_json"] or "{}"))
            row = {
                "snapshot_id": int(snapshot["snapshot_id"]),
                "dataset_id": int(snapshot["dataset_id"]),
                "ticker": snapshot["ticker"],
                "trading_date": snapshot["trading_date"],
                "as_of_timestamp": snapshot["as_of_timestamp"],
                **features,
                **labels.get(int(snapshot["snapshot_id"]), {}),
            }
            rows.append(row)
            for column in row:
                if column not in seen:
                    seen.add(column)
                    columns.append(column)
    return columns, rows


def _frozen_hash_readonly(columns: list[str], rows: list[dict[str, Any]], chunk_size: int = 1000) -> tuple[int, str]:
    stable_columns = sorted(column for column in columns if column not in {"snapshot_id", "dataset_id"})
    stats: dict[str, dict[str, Any]] = {column: {"missing": False, "kinds": set()} for column in columns}
    for row in rows:
        for column in columns:
            value = row.get(column)
            try:
                missing = value is None or bool(pd.isna(value))
            except Exception:
                missing = value is None
            if missing:
                stats[column]["missing"] = True
            elif isinstance(value, bool):
                stats[column]["kinds"].add("bool")
            elif isinstance(value, int):
                stats[column]["kinds"].add("int")
            elif isinstance(value, float):
                stats[column]["kinds"].add("float")
            else:
                stats[column]["kinds"].add("object")
    dtypes: dict[str, str] = {}
    for column, info in stats.items():
        kinds = info["kinds"]
        if not kinds or "object" in kinds:
            continue
        if kinds == {"bool"} and not info["missing"]:
            dtypes[column] = "bool"
        elif kinds.issubset({"int", "float", "bool"}):
            dtypes[column] = "float64" if info["missing"] or "float" in kinds else "int64"
    hasher = hashlib.sha256()
    for start in range(0, len(rows), chunk_size):
        frame = pd.DataFrame(rows[start : start + chunk_size])
        for column, dtype in dtypes.items():
            if dtype in {"float64", "int64"}:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(dtype)
            elif dtype == "bool":
                frame[column] = frame[column].astype(bool)
        payload = frame.reindex(columns=stable_columns).to_csv(
            index=False, na_rep="", header=start == 0
        )
        hasher.update(payload.encode("utf-8"))
    return len(rows), hasher.hexdigest()


def _load_training_readonly(build: dict[str, Any], columns: list[str], rows: list[dict[str, Any]]) -> TrainingDataset:
    frame = pd.DataFrame(rows, columns=columns)
    feature_columns = json.loads(str(build.get("feature_columns_json") or "[]"))
    label_columns = json.loads(str(build.get("label_columns_json") or "[]"))
    identifier_columns = json.loads(str(build.get("identifier_columns_json") or "[]"))
    metadata_columns = json.loads(str(build.get("metadata_columns_json") or "[]"))
    audit_columns = json.loads(str(build.get("audit_columns_json") or "[]"))
    if TARGET_COLUMN not in label_columns or TARGET_COLUMN not in frame.columns:
        raise ArtifactContractError(f"Dataset 49 does not contain the required target {TARGET_COLUMN}.")
    forbidden = set(label_columns) | set(identifier_columns) | set(metadata_columns) | set(audit_columns)
    leaked = sorted(set(feature_columns) & forbidden)
    if leaked:
        raise ArtifactContractError(f"Dataset 49 feature manifest leaks forbidden columns: {', '.join(leaked)}")
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ArtifactContractError(f"Dataset 49 is missing model features: {', '.join(missing)}")
    return TrainingDataset(
        X=frame[feature_columns].copy(),
        y=frame[TARGET_COLUMN].copy(),
        metadata=frame[[column for column in [*identifier_columns, *metadata_columns] if column in frame]].copy(),
        audit=frame[[column for column in audit_columns if column in frame]].copy(),
        feature_columns=feature_columns,
        label_column=TARGET_COLUMN,
    )


def _metadata_dates(training: TrainingDataset, valid: pd.Series) -> tuple[str, str, list[str]]:
    if "trading_date" not in training.metadata.columns:
        raise ArtifactContractError("Dataset metadata does not contain trading_date.")
    dates = pd.to_datetime(training.metadata.loc[valid, "trading_date"], errors="coerce")
    if dates.isna().any():
        raise ArtifactContractError("Eligible training rows contain invalid trading dates.")
    ids_column = "snapshot_id" if "snapshot_id" in training.metadata.columns else None
    if ids_column is None:
        raise ArtifactContractError("Dataset metadata does not contain immutable snapshot identifiers.")
    identifiers = [str(int(value)) for value in training.metadata.loc[valid, ids_column].tolist()]
    return dates.min().date().isoformat(), dates.max().date().isoformat(), identifiers


def _registered_artifact_by_spec(db_path: str | Path, specification_hash: str) -> str | None:
    with storage.connect(db_path) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_artifacts'"
        ).fetchone()
        if exists is None:
            return None
        row = conn.execute(
            "SELECT artifact_id FROM model_artifacts WHERE specification_hash = ?", (specification_hash,)
        ).fetchone()
    return None if row is None else str(row["artifact_id"])


def resolve_artifact_contract(
    db_path: str | Path,
    artifact_root: str | Path,
    *,
    dataset_id: int = EXPECTED_DATASET_ID,
    expected_hash: str = EXPECTED_DATASET_HASH,
    expected_row_count: int = EXPECTED_DATASET_ROWS,
) -> tuple[ArtifactContract, TrainingDataset, pd.Series]:
    if int(dataset_id) != EXPECTED_DATASET_ID:
        raise ArtifactContractError("This artifact contract permits Dataset 49 only; Dataset 50 is not allowed.")
    build = _read_dataset_build(db_path, dataset_id)
    stored_hash = str(build.get("data_hash") or "")
    stored_rows = int(build.get("row_count") or 0)
    if stored_hash != expected_hash or stored_rows != int(expected_row_count):
        raise ArtifactContractError(
            f"Stored Dataset 49 contract mismatch: rows={stored_rows}, hash={stored_hash}."
        )
    columns, rows = _frozen_rows_readonly(db_path, dataset_id)
    streamed_rows, streamed_hash = _frozen_hash_readonly(columns, rows)
    if int(streamed_rows) != int(expected_row_count) or streamed_hash != expected_hash:
        raise ArtifactContractError(
            "Frozen Dataset 49 rows do not reproduce the required content hash; artifact fitting is blocked."
        )

    training = _load_training_readonly(build, columns, rows)
    feature_columns = select_feature_columns(training.feature_columns, "technical_core")
    if feature_columns != TECHNICAL_CORE_CANDIDATES:
        missing = [column for column in TECHNICAL_CORE_CANDIDATES if column not in feature_columns]
        raise ArtifactContractError(
            "The exact technical_core feature contract is unavailable"
            + (f": missing {', '.join(missing)}" if missing else ".")
        )
    if any(column not in training.X.columns for column in feature_columns):
        raise ArtifactContractError("Dataset 49 does not contain every frozen technical_core feature.")

    y = pd.to_numeric(training.y, errors="coerce")
    valid = y.notna()
    if not valid.any():
        raise ArtifactContractError("Dataset 49 contains no eligible 5-session excess-return labels.")
    start, cutoff, identifiers = _metadata_dates(training, valid)
    lower = float(y.loc[valid].quantile(0.01))
    upper = float(y.loc[valid].quantile(0.99))
    X = training.X.loc[valid, feature_columns].copy()
    probe = MatrixPreprocessor.fit(X.replace([np.inf, -np.inf], np.nan))

    universe = json.loads(str(build.get("ticker_universe_json") or "[]"))
    universe = [str(value) for value in universe]
    feature_manifest_hash = _sha256_json(feature_columns)
    universe_hash = _sha256_json(universe)
    specification = {
        "artifact_name": ARTIFACT_NAME,
        "artifact_version": ARTIFACT_VERSION,
        "dataset_id": dataset_id,
        "dataset_hash": expected_hash,
        "dataset_rows": expected_row_count,
        "features": feature_columns,
        "target": TARGET_COLUMN,
        "winsorization": {"lower_quantile": 0.01, "upper_quantile": 0.99, "lower": lower, "upper": upper},
        "model": {"family": "sklearn.linear_model.Ridge", **MODEL_PARAMETERS},
        "preprocessing": "matrix_preprocessor_v1_inf_to_missing_no_scaling",
        "training_cutoff": cutoff,
    }
    specification_hash = _sha256_json(specification)
    artifact_id = f"{ARTIFACT_NAME}_{specification_hash[:12]}"
    artifact_path = str(Path(artifact_root) / artifact_id)
    contract = ArtifactContract(
        dataset_id=dataset_id,
        dataset_hash=expected_hash,
        dataset_row_count=expected_row_count,
        feature_columns=feature_columns,
        feature_manifest_hash=feature_manifest_hash,
        target_column=TARGET_COLUMN,
        training_row_count=int(valid.sum()),
        excluded_row_count=int((~valid).sum()),
        training_start=start,
        training_cutoff=cutoff,
        training_row_identifier_hash=_sha256_json(identifiers),
        universe=universe,
        universe_hash=universe_hash,
        winsor_lower=lower,
        winsor_upper=upper,
        numeric_columns=list(probe.numeric_columns),
        categorical_columns=list(probe.categorical_columns),
        dependencies=_dependency_versions(),
        artifact_id=artifact_id,
        specification_hash=specification_hash,
        intended_artifact_path=artifact_path,
        duplicate_artifact_id=_registered_artifact_by_spec(db_path, specification_hash),
    )
    return contract, training, valid


def dry_run_artifact_build(
    db_path: str | Path,
    artifact_root: str | Path,
    *,
    dataset_id: int = EXPECTED_DATASET_ID,
    expected_hash: str = EXPECTED_DATASET_HASH,
    expected_row_count: int = EXPECTED_DATASET_ROWS,
) -> dict[str, Any]:
    contract, _training, _valid = resolve_artifact_contract(
        db_path,
        artifact_root,
        dataset_id=dataset_id,
        expected_hash=expected_hash,
        expected_row_count=expected_row_count,
    )
    return {
        "status": "ready" if contract.duplicate_artifact_id is None else "already_registered",
        "dry_run": True,
        "database_mutated": False,
        "artifact_written": False,
        "contract": asdict(contract),
        "model_parameters": MODEL_PARAMETERS,
        "preprocessing": "median imputation, deterministic categorical encoding, no scaling",
        "warnings": [
            "This artifact is exploratory_shadow, is not run 145 or run 283, and is not validated alpha."
        ],
    }


def _backup_database(db_path: str | Path) -> Path:
    source = Path(db_path)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    backup = source.with_name(f"{source.stem}_backup_phase3a0b_{stamp}{source.suffix}")
    with sqlite3.connect(source) as src, sqlite3.connect(backup) as dst:
        src.backup(dst)
    return backup


def _select_replay_rows(training: TrainingDataset, valid: pd.Series, features: list[str], count: int = 12) -> pd.DataFrame:
    source = training.X.loc[valid, features]
    if source.empty:
        raise ArtifactContractError("No eligible rows are available for the replay fixture.")
    positions = set(np.linspace(0, len(source) - 1, min(count, len(source)), dtype=int).tolist())
    missing = source.replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    if missing.any():
        positions.add(int(np.flatnonzero(missing.to_numpy())[0]))
    return source.iloc[sorted(positions)].head(count).copy()


def _preprocessing_state(preprocessor: MatrixPreprocessor) -> dict[str, Any]:
    return {
        "version": "matrix_preprocessor_v1_inf_to_missing_no_scaling",
        "numeric_columns": preprocessor.numeric_columns,
        "categorical_columns": preprocessor.categorical_columns,
        "medians": preprocessor.medians,
        "categories": preprocessor.categories,
        "output_columns": preprocessor.output_columns,
        "infinity_handling": "replace positive and negative infinity with missing before imputation",
        "scaling": "none",
        "missing_value_indicators": [],
        "dropped_columns": [],
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _register_artifact(db_path: str | Path, manifest: dict[str, Any]) -> None:
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO model_artifacts (
                artifact_id, artifact_name, artifact_version, model_family, status,
                evaluation_regime, derived_from_run_id, design_reference_run_ids_json,
                dataset_id, dataset_hash, feature_manifest_hash, universe_hash,
                training_cutoff, artifact_path, manifest_path, artifact_checksum,
                code_commit_hash, specification_hash, created_at, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                manifest["artifact_id"], manifest["artifact_name"], manifest["artifact_version"],
                manifest["model_family"], manifest["status"], manifest["evaluation_regime"], None,
                _canonical_json(manifest["design_reference_run_ids"]), manifest["dataset_id"],
                manifest["dataset_hash"], manifest["feature_manifest_hash"], manifest["universe_hash"],
                manifest["training_cutoff"], manifest["artifact_path"], manifest["manifest_path"],
                manifest["artifact_checksum"], manifest["code_commit_hash"], manifest["specification_hash"],
                manifest["created_at"], manifest["notes"],
            ),
        )


def apply_artifact_build(
    db_path: str | Path,
    artifact_root: str | Path,
    *,
    project_root: str | Path,
    dataset_id: int = EXPECTED_DATASET_ID,
    expected_hash: str = EXPECTED_DATASET_HASH,
    expected_row_count: int = EXPECTED_DATASET_ROWS,
) -> dict[str, Any]:
    contract, training, valid = resolve_artifact_contract(
        db_path,
        artifact_root,
        dataset_id=dataset_id,
        expected_hash=expected_hash,
        expected_row_count=expected_row_count,
    )
    if contract.duplicate_artifact_id:
        raise ArtifactContractError(f"Artifact specification is already registered as {contract.duplicate_artifact_id}.")
    final_dir = Path(contract.intended_artifact_path)
    if final_dir.exists():
        raise ArtifactContractError(f"Artifact path already exists and will not be overwritten: {final_dir}")

    backup_path = _backup_database(db_path)
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{contract.artifact_id}-", dir=root))
    try:
        X = training.X.loc[valid, contract.feature_columns].copy()
        y = pd.to_numeric(training.y.loc[valid], errors="raise").clip(contract.winsor_lower, contract.winsor_upper)
        fit_frame = X.replace([np.inf, -np.inf], np.nan)
        preprocessor = MatrixPreprocessor.fit(fit_frame)
        matrix = preprocessor.transform(_sanitize_frame(fit_frame, preprocessor.numeric_columns))
        model = Ridge(**MODEL_PARAMETERS)
        model.fit(matrix, y)
        bundle = ExecutableShadowArtifact(
            feature_columns=contract.feature_columns,
            preprocessor=preprocessor,
            model=model,
            artifact_id=contract.artifact_id,
            target_column=TARGET_COLUMN,
            winsor_lower=contract.winsor_lower,
            winsor_upper=contract.winsor_upper,
        )

        model_path = temp_dir / "model.joblib"
        joblib.dump(bundle, model_path, compress=3)
        coefficients_path = temp_dir / "coefficients.csv"
        with coefficients_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["output_feature", "coefficient", "intercept"])
            writer.writeheader()
            for column, coefficient in zip(preprocessor.output_columns, np.asarray(model.coef_).ravel(), strict=True):
                writer.writerow({"output_feature": column, "coefficient": format(float(coefficient), ".17g"), "intercept": ""})
            writer.writerow({"output_feature": "__INTERCEPT__", "coefficient": "", "intercept": format(float(model.intercept_), ".17g")})

        preprocessing_path = temp_dir / "preprocessing_state.json"
        _write_json(preprocessing_path, _preprocessing_state(preprocessor))
        fixture = _select_replay_rows(training, valid, contract.feature_columns)
        fixture_path = temp_dir / "replay_fixture.csv"
        fixture.to_csv(fixture_path, index=False, float_format="%.17g")
        expected = bundle.predict(fixture)
        expected_path = temp_dir / "replay_expected_predictions.csv"
        pd.DataFrame({"fixture_row": range(len(expected)), "predicted_value": expected}).to_csv(
            expected_path, index=False, float_format="%.17g"
        )

        project = Path(project_root)
        code_commit, code_dirty = _git_state(project)
        created_at = _now_iso()
        core_files = [model_path, coefficients_path, preprocessing_path, fixture_path, expected_path]
        core_checksums = {path.name: _file_sha256(path) for path in core_files}
        artifact_checksum = _sha256_json(core_checksums)
        manifest_path = temp_dir / "model_manifest.json"
        manifest = {
            "artifact_id": contract.artifact_id,
            "artifact_name": ARTIFACT_NAME,
            "artifact_version": ARTIFACT_VERSION,
            "status": "frozen_exploratory",
            "evaluation_regime": "exploratory_shadow",
            "derived_from_run_id": None,
            "design_reference_run_ids": DESIGN_REFERENCE_RUN_IDS,
            "equivalence_warning": "This artifact is not run 145 or run 283 and does not recreate either experiment.",
            "model_family": "sklearn.linear_model.Ridge",
            "model_parameters": MODEL_PARAMETERS,
            "dataset_id": contract.dataset_id,
            "dataset_hash": contract.dataset_hash,
            "dataset_row_count": contract.dataset_row_count,
            "training_row_count": contract.training_row_count,
            "excluded_row_count": contract.excluded_row_count,
            "excluded_row_reasons": {"missing_target": contract.excluded_row_count},
            "training_start": contract.training_start,
            "training_cutoff": contract.training_cutoff,
            "training_row_identifier_hash": contract.training_row_identifier_hash,
            "target_name": TARGET_COLUMN,
            "target_transformation": {
                "name": "full_training_set_winsorization",
                "lower_quantile": 0.01,
                "upper_quantile": 0.99,
                "lower_threshold": contract.winsor_lower,
                "upper_threshold": contract.winsor_upper,
            },
            "feature_set_name": "technical_core",
            "feature_columns": contract.feature_columns,
            "feature_count": len(contract.feature_columns),
            "feature_manifest_hash": contract.feature_manifest_hash,
            "preprocessing": _preprocessing_state(preprocessor),
            "universe": contract.universe,
            "universe_hash": contract.universe_hash,
            "dependencies": contract.dependencies,
            "code_commit_hash": code_commit,
            "code_worktree_was_dirty": code_dirty,
            "specification_hash": contract.specification_hash,
            "artifact_path": str(final_dir),
            "manifest_path": str(final_dir / "model_manifest.json"),
            "artifact_checksum": artifact_checksum,
            "artifact_file_checksums": core_checksums,
            "replay_tolerance": REPLAY_TOLERANCE,
            "created_at": created_at,
            "notes": "Frozen exploratory technical Ridge for prospective shadow research only; not validated alpha or trade-ready.",
        }
        _write_json(manifest_path, manifest)
        checksums = {path.name: _file_sha256(path) for path in [*core_files, manifest_path]}
        _write_json(temp_dir / "checksums.json", checksums)

        integrity = validate_artifact_directory(temp_dir, expected_artifact_id=contract.artifact_id)
        if integrity["status"] != "passed":
            raise ArtifactIntegrityError(f"Artifact validation failed: {integrity}")
        os.replace(temp_dir, final_dir)
        try:
            _register_artifact(db_path, manifest)
        except Exception:
            shutil.rmtree(final_dir, ignore_errors=True)
            raise
        return {
            "status": "created",
            "artifact_id": contract.artifact_id,
            "artifact_path": str(final_dir),
            "manifest_path": str(final_dir / "model_manifest.json"),
            "artifact_checksum": artifact_checksum,
            "database_backup": str(backup_path),
            "contract": asdict(contract),
            "integrity": integrity,
            "shadow_predictions_created": 0,
        }
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def _registry_row(db_path: str | Path, artifact_id: str) -> dict[str, Any] | None:
    with storage.connect(db_path) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_artifacts'"
        ).fetchone()
        if exists is None:
            return None
        row = conn.execute("SELECT * FROM model_artifacts WHERE artifact_id = ?", (artifact_id,)).fetchone()
    return None if row is None else dict(row)


def validate_artifact_directory(path: str | Path, expected_artifact_id: str | None = None) -> dict[str, Any]:
    directory = Path(path)
    required = {
        "model.joblib", "model_manifest.json", "coefficients.csv", "preprocessing_state.json",
        "replay_fixture.csv", "replay_expected_predictions.csv", "checksums.json",
    }
    missing_files = sorted(name for name in required if not (directory / name).is_file())
    if missing_files:
        return {"status": "failed", "errors": [f"Missing artifact files: {', '.join(missing_files)}"]}
    errors: list[str] = []
    try:
        manifest = json.loads((directory / "model_manifest.json").read_text(encoding="utf-8"))
        checksums = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "failed", "errors": [f"Artifact metadata is invalid: {exc}"]}
    if expected_artifact_id and manifest.get("artifact_id") != expected_artifact_id:
        errors.append("Manifest artifact_id mismatch.")
    for filename, expected in checksums.items():
        file_path = directory / filename
        if not file_path.is_file() or _file_sha256(file_path) != expected:
            errors.append(f"Checksum mismatch: {filename}")
    core_checksums = manifest.get("artifact_file_checksums", {})
    if _sha256_json(core_checksums) != manifest.get("artifact_checksum"):
        errors.append("Aggregate artifact checksum mismatch.")
    for filename, expected in core_checksums.items():
        if not (directory / filename).is_file() or _file_sha256(directory / filename) != expected:
            errors.append(f"Manifest checksum mismatch: {filename}")
    max_difference: float | None = None
    try:
        bundle = joblib.load(directory / "model.joblib")
        fixture = pd.read_csv(directory / "replay_fixture.csv")
        fixture = fixture.reindex(columns=bundle.feature_columns)
        expected = pd.read_csv(directory / "replay_expected_predictions.csv")["predicted_value"].to_numpy(float)
        actual = bundle.predict(fixture)
        max_difference = float(np.max(np.abs(actual - expected))) if len(actual) else 0.0
        if max_difference > REPLAY_TOLERANCE:
            errors.append(f"Replay difference {max_difference} exceeds tolerance {REPLAY_TOLERANCE}.")
    except Exception as exc:
        errors.append(f"Replay failed: {exc}")
    installed = _dependency_versions()
    dependency_warnings = [
        f"{name}: artifact={version}, installed={installed.get(name)}"
        for name, version in (manifest.get("dependencies") or {}).items()
        if installed.get(name) != version
    ]
    return {
        "status": "passed" if not errors else "failed",
        "artifact_id": manifest.get("artifact_id"),
        "dataset_id": manifest.get("dataset_id"),
        "dataset_hash": manifest.get("dataset_hash"),
        "feature_manifest_hash": manifest.get("feature_manifest_hash"),
        "feature_contract_status": "passed" if not any("feature" in error.lower() for error in errors) else "failed",
        "replay_row_count": int(len(expected)) if "expected" in locals() else 0,
        "maximum_absolute_prediction_difference": max_difference,
        "replay_tolerance": REPLAY_TOLERANCE,
        "dependency_version_status": "passed" if not dependency_warnings else "warn",
        "dependency_warnings": dependency_warnings,
        "errors": errors,
    }


def check_registered_artifact(db_path: str | Path, artifact_id: str) -> dict[str, Any]:
    row = _registry_row(db_path, artifact_id)
    if row is None:
        return {"status": "failed", "artifact_id": artifact_id, "errors": ["Artifact is not registered."]}
    integrity = validate_artifact_directory(row["artifact_path"], expected_artifact_id=artifact_id)
    try:
        manifest = json.loads(Path(row["manifest_path"]).read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "failed", "artifact_id": artifact_id, "errors": [f"Manifest cannot be read: {exc}"]}
    registry_errors = []
    for field in ("dataset_hash", "feature_manifest_hash", "universe_hash", "artifact_checksum", "specification_hash"):
        if str(row[field]) != str(manifest.get(field)):
            registry_errors.append(f"Registry/manifest mismatch: {field}")
    if registry_errors:
        integrity["status"] = "failed"
        integrity.setdefault("errors", []).extend(registry_errors)
    integrity["registry_status"] = "passed" if not registry_errors else "failed"
    return integrity


def list_model_artifacts(db_path: str | Path) -> pd.DataFrame:
    with storage.connect(db_path) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_artifacts'"
        ).fetchone()
        if exists is None:
            return pd.DataFrame()
        return pd.read_sql_query("SELECT * FROM model_artifacts ORDER BY created_at, artifact_id", conn)
