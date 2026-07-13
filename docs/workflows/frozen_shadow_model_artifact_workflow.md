# Frozen Shadow Model Artifact Workflow

## Purpose

An exploratory model run is a historical experiment record. It is not an executable model unless its fitted estimator and preprocessing state were persisted. Runs 145 and 283 are design references only; `shadow_ridge_technical_v1` does not recreate or replace either run.

The frozen artifact is a prospective research instrument. It is labeled `exploratory_shadow`, is not validated alpha, is not production, and is not trade-ready.

## Frozen Contract

- Dataset: exact stored Dataset 49 rows only
- Required row count: 19,495
- Required hash: `e9523e1134b7eb32b142cb628d51bde76d5a6d139f4be2aba2545f3ca4416184`
- Features: the ordered `technical_core` feature set stored in Dataset 49
- Target: `label_5_session_excess_return`
- Target transformation: one 1st/99th percentile winsorization fit across all eligible training rows
- Model: `sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True, solver="auto")`
- Preprocessing: infinity to missing, fitted numeric median imputation, deterministic categorical encoding, no scaling
- Dataset 50: prohibited

If the frozen row hash, feature contract, target, or preprocessing contract cannot be resolved exactly, the build stops without fitting or registration.

## Build

Dry-run first:

```bash
.venv/bin/python scripts/build_shadow_model_artifact.py \
  --dataset-id 49 \
  --artifact-name shadow_ridge_technical_v1 \
  --dry-run
```

The dry-run verifies both gates and reports rows, cutoff, target thresholds, feature count, dependencies, and intended paths without writing an artifact or registry row.

Explicit application:

```bash
.venv/bin/python scripts/build_shadow_model_artifact.py \
  --dataset-id 49 \
  --artifact-name shadow_ridge_technical_v1 \
  --apply
```

Application creates a timestamped SQLite backup, fits once, validates in a temporary directory, finalizes the files atomically, and registers immutable metadata. Existing artifact specifications and paths are never overwritten.

## Artifact Files

Local ignored artifacts live under `data/model_artifacts/<artifact_id>/`:

- `model.joblib`: fitted preprocessor, Ridge estimator, and strict feature contract
- `model_manifest.json`: provenance, hashes, versions, rows, cutoff, target thresholds, and warnings
- `coefficients.csv`: output-feature coefficients and intercept
- `preprocessing_state.json`: medians, categorical mappings, output ordering, and no-scaling declaration
- `replay_fixture.csv`: deterministic frozen Dataset 49 feature sample
- `replay_expected_predictions.csv`: predictions produced by this artifact
- `checksums.json`: file integrity hashes

The SQLite `model_artifacts` registry is append-only through update/delete prevention triggers.

## Replay And Integrity

```bash
.venv/bin/python scripts/quality_harness.py model-artifact-check \
  --artifact-id <artifact_id>
```

The check verifies registry/manifest agreement, file checksums, dependency versions, strict feature order, and saved-artifact replay. Replay compares the reloaded artifact with its own expected fixture predictions at an absolute tolerance of `1e-12`.

It intentionally does not compare against run 283 fold predictions because those were produced by different fold-specific estimators that were not saved.

## Future Inference Guardrails

- Daily prediction records are a separate phase and do not exist merely because this artifact exists.
- Inference must supply every feature in the exact stored order.
- Missing, extra, or reordered features fail closed.
- Shadow predictions must remain separate from scanner scoring and trading actions.
- Dataset 50 cannot be used for fitting or replay and remains governed by its holdout workflow.
