from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.modeling.daily_shadow_cycle import (
    DEFAULT_SHADOW_ARTIFACT_ID,
    DailyShadowCycleError,
    cycle_lock,
    run_daily_shadow_cycle,
)
from src.modeling.shadow_portfolios import (
    POLICY_ID,
    apply_outcomes as apply_portfolio_outcomes,
    create_cohort,
    dry_run_cohort,
    dry_run_outcomes as dry_run_portfolio_outcomes,
    portfolio_shadow_status,
)
from src.options_research.snapshots import (
    DEFAULT_PROVIDER,
    collect_options_snapshots,
    options_status_report,
)
from src.utils.trading_calendar import latest_expected_trading_day


class DailyResearchCycleError(RuntimeError):
    def __init__(self, message: str, exit_code: int = 2):
        super().__init__(message)
        self.exit_code = exit_code


def _options_run_exists(status: dict[str, Any], resolved_session: str) -> int | None:
    latest = status.get("latest_run") or {}
    if (
        str(latest.get("snapshot_date") or "") == resolved_session
        and str(latest.get("provider") or "") == DEFAULT_PROVIDER
    ):
        return int(latest["run_id"])
    return None


def _portfolio_cohort_step(db_path: Path, *, apply: bool) -> dict[str, Any]:
    created: list[dict[str, Any]] = []
    planned: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    while True:
        status = portfolio_shadow_status(db_path, policy_id=POLICY_ID)
        next_run = status.get("next_available_prediction_run")
        if next_run is None:
            break
        run_id = int(next_run["run_id"])
        plan = dry_run_cohort(db_path, run_id)
        if not apply:
            planned.append(
                {
                    "prediction_run_id": run_id,
                    "prediction_date": plan["plan"]["prediction_date"],
                    "status": plan["status"],
                }
            )
            break
        result = create_cohort(db_path, run_id)
        if result["status"] == "recorded":
            created.append(result)
            continue
        skipped.append(
            {
                "prediction_run_id": run_id,
                "status": result["status"],
                "cohort_id": result.get("cohort_id"),
            }
        )
        break

    final_status = portfolio_shadow_status(db_path, policy_id=POLICY_ID)
    return {
        "status": "created" if created else "planned" if planned else "skipped",
        "created": created,
        "planned": planned,
        "skipped": skipped,
        "created_count": len(created),
        "planned_count": len(planned),
        "sample_status": final_status.get("sample_status"),
        "next_available_prediction_run": final_status.get("next_available_prediction_run"),
    }


def _portfolio_outcome_step(db_path: Path, *, apply: bool) -> dict[str, Any]:
    plan = dry_run_portfolio_outcomes(db_path)
    result: dict[str, Any] = {
        "status": plan["status"],
        "planned_count": int(plan["outcomes_planned"]),
        "matured_count": 0,
        "pending_count": int(plan["pending_outcomes"]),
    }
    if apply and plan["outcomes_planned"]:
        applied = apply_portfolio_outcomes(db_path)
        result.update(
            {
                "status": applied["status"],
                "matured_count": int(applied["outcomes_created"]),
                "pending_count": int(applied["pending_outcomes"]),
            }
        )
    return result


def _options_step(
    db_path: Path,
    *,
    apply: bool,
    network_authorized: bool,
    resolved_session: str,
    reference_time: datetime | date | None,
) -> dict[str, Any]:
    before = options_status_report(db_path)
    duplicate_run_id = _options_run_exists(before, resolved_session)
    if duplicate_run_id is not None:
        return {
            "status": "skipped",
            "reason": "duplicate_snapshot_date_provider",
            "run_id": duplicate_run_id,
            "snapshot_date": resolved_session,
            "network_calls_made": False,
            "database_mutated": False,
            "failed_tickers": [],
            "sample_status": before.get("sample_status"),
        }
    if apply and not network_authorized:
        return {
            "status": "skipped",
            "reason": "network_not_authorized",
            "snapshot_date": resolved_session,
            "network_calls_made": False,
            "database_mutated": False,
            "failed_tickers": [],
            "sample_status": before.get("sample_status"),
        }

    collected = collect_options_snapshots(
        db_path,
        apply=bool(apply and network_authorized),
        reference_time=reference_time,
    )
    after = options_status_report(db_path)
    return {
        "status": collected["status"],
        "reason": None,
        "run_id": collected.get("run_id"),
        "snapshot_date": collected["snapshot_date"],
        "contract_count": int(collected.get("contract_count", 0)),
        "successful_tickers": collected.get("successful_tickers", []),
        "failed_tickers": collected.get("failed_tickers", []),
        "network_calls_made": bool(collected.get("network_calls_made", False)),
        "database_mutated": bool(collected.get("database_mutated", False)),
        "sample_status": after.get("sample_status"),
    }


def _overall_status(report: dict[str, Any], *, apply: bool) -> str:
    errors = report["errors"]
    provider_failures = report["provider_failures"]
    components = report["components"]
    if components["shadow"].get("status") == "failed":
        return "failed"
    if errors or provider_failures or any(
        component.get("status") == "failed" for component in components.values()
    ):
        return "partial_failure"
    if apply and not report["database_mutated"]:
        return "no_op"
    return "healthy"


def run_daily_research_cycle(
    db_path: str | Path,
    *,
    apply: bool = False,
    refresh_market_data: bool = False,
    artifact_id: str = DEFAULT_SHADOW_ARTIFACT_ID,
    reference_time: datetime | date | None = None,
    lock_path: str | Path | None = None,
) -> dict[str, Any]:
    db_path = Path(db_path)
    resolved_session = latest_expected_trading_day(reference_time).isoformat()
    network_authorized = bool(apply and refresh_market_data)
    master_lock = (
        Path(lock_path)
        if lock_path is not None
        else db_path.with_suffix(db_path.suffix + ".research-cycle.lock")
    )
    report: dict[str, Any] = {
        "status": "healthy",
        "mode": "apply" if apply else "dry_run",
        "resolved_completed_session": resolved_session,
        "artifact_id": artifact_id,
        "policy_id": POLICY_ID,
        "network_authorized": network_authorized,
        "database_mutated": False,
        "components": {},
        "provider_failures": [],
        "sample_statuses": {},
        "warnings": [],
        "errors": [],
    }

    try:
        with cycle_lock(master_lock):
            try:
                shadow = run_daily_shadow_cycle(
                    db_path,
                    artifact_id=artifact_id,
                    apply=apply,
                    refresh_market_data=network_authorized,
                    reference_time=reference_time,
                    lock_path=db_path.with_suffix(db_path.suffix + ".shadow-cycle.lock"),
                )
            except Exception as exc:
                shadow = {"status": "failed", "error": str(exc), "database_mutated": False}
                report["errors"].append({"component": "shadow", "error": str(exc)})
            report["components"]["shadow"] = shadow
            report["database_mutated"] |= bool(shadow.get("database_mutated", False))
            report["provider_failures"].extend(
                {"component": "market_data", **item} for item in shadow.get("refresh_failures", [])
            )
            report["sample_statuses"]["shadow"] = shadow.get("sample_status")

            try:
                cohorts = _portfolio_cohort_step(db_path, apply=apply)
            except Exception as exc:
                cohorts = {"status": "failed", "created_count": 0, "error": str(exc)}
                report["errors"].append({"component": "portfolio_cohorts", "error": str(exc)})
            report["components"]["portfolio_cohorts"] = cohorts
            report["database_mutated"] |= bool(cohorts.get("created_count", 0))

            try:
                portfolio_outcomes = _portfolio_outcome_step(db_path, apply=apply)
            except Exception as exc:
                portfolio_outcomes = {"status": "failed", "matured_count": 0, "error": str(exc)}
                report["errors"].append({"component": "portfolio_outcomes", "error": str(exc)})
            report["components"]["portfolio_outcomes"] = portfolio_outcomes
            report["database_mutated"] |= bool(portfolio_outcomes.get("matured_count", 0))

            try:
                options = _options_step(
                    db_path,
                    apply=apply,
                    network_authorized=network_authorized,
                    resolved_session=resolved_session,
                    reference_time=reference_time,
                )
            except Exception as exc:
                options = {
                    "status": "failed",
                    "error": str(exc),
                    "database_mutated": False,
                    "network_calls_made": False,
                    "failed_tickers": [],
                }
                report["errors"].append({"component": "options", "error": str(exc)})
            report["components"]["options"] = options
            report["database_mutated"] |= bool(options.get("database_mutated", False))
            report["provider_failures"].extend(
                {"component": "options", **item} for item in options.get("failed_tickers", [])
            )

            try:
                portfolio_status = portfolio_shadow_status(db_path, policy_id=POLICY_ID)
                options_status = options_status_report(db_path)
                report["sample_statuses"]["portfolio"] = portfolio_status.get("sample_status")
                report["sample_statuses"]["options"] = options_status.get("sample_status")
            except Exception as exc:
                report["warnings"].append(f"Post-cycle status audit failed: {exc}")
    except DailyShadowCycleError as exc:
        raise DailyResearchCycleError(str(exc), exc.exit_code) from exc

    shadow = report["components"]["shadow"]
    cohorts = report["components"]["portfolio_cohorts"]
    portfolio_outcomes = report["components"]["portfolio_outcomes"]
    options = report["components"]["options"]
    report["shadow_run"] = shadow.get("prediction_run", {"status": shadow.get("status")})
    report["shadow_outcomes_added"] = shadow.get(
        "outcomes_added_by_horizon", {"1": 0, "5": 0, "20": 0}
    )
    report["portfolio_cohorts"] = {
        "status": cohorts.get("status"),
        "created_count": int(cohorts.get("created_count", 0)),
        "planned_count": int(cohorts.get("planned_count", 0)),
        "skipped": cohorts.get("skipped", []),
    }
    report["portfolio_outcomes_matured"] = int(portfolio_outcomes.get("matured_count", 0))
    report["options_snapshot"] = {
        "status": options.get("status"),
        "reason": options.get("reason"),
        "run_id": options.get("run_id"),
        "snapshot_date": options.get("snapshot_date"),
        "contract_count": int(options.get("contract_count", 0)),
    }
    report["status"] = _overall_status(report, apply=apply)
    return report
