from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.modeling import daily_research_cycle as research_cycle
from src.modeling.daily_research_cycle import DailyResearchCycleError, run_daily_research_cycle
from src.modeling.daily_shadow_cycle import cycle_lock


REFERENCE_TIME = datetime(2026, 7, 14, 23, 0, tzinfo=UTC)


def _portfolio_status(next_run: int | None = None) -> dict:
    return {
        "status": "passed",
        "policy_registered": True,
        "next_available_prediction_run": (
            None if next_run is None else {"run_id": next_run, "prediction_date": "2026-07-14"}
        ),
        "sample_status": "insufficient_forward_sample",
    }


def _options_status(run_id: int | None = None) -> dict:
    return {
        "status": "passed",
        "latest_run": (
            None
            if run_id is None
            else {"run_id": run_id, "snapshot_date": "2026-07-14", "provider": "yfinance"}
        ),
        "sample_status": "collection_only",
    }


def _install_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        research_cycle,
        "run_daily_shadow_cycle",
        lambda *_args, **_kwargs: {
            "status": "no_op",
            "database_mutated": False,
            "refresh_failures": [],
            "prediction_run": {"status": "skipped", "reason": "duplicate_date_artifact_run"},
            "outcomes_added_by_horizon": {"1": 0, "5": 0, "20": 0},
            "sample_status": "insufficient_forward_sample",
        },
    )
    monkeypatch.setattr(research_cycle, "portfolio_shadow_status", lambda *_args, **_kwargs: _portfolio_status())
    monkeypatch.setattr(
        research_cycle,
        "dry_run_portfolio_outcomes",
        lambda *_args, **_kwargs: {
            "status": "no_newly_matured_outcomes",
            "outcomes_planned": 0,
            "pending_outcomes": 0,
        },
    )
    monkeypatch.setattr(research_cycle, "options_status_report", lambda *_args, **_kwargs: _options_status())
    monkeypatch.setattr(
        research_cycle,
        "collect_options_snapshots",
        lambda *_args, **kwargs: {
            "status": "planned",
            "snapshot_date": "2026-07-14",
            "run_id": None,
            "contract_count": 0,
            "successful_tickers": [],
            "failed_tickers": [],
            "network_calls_made": bool(kwargs.get("apply")),
            "database_mutated": False,
        },
    )


def test_complete_successful_cycle_uses_required_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    cohort_created = False

    def shadow(*_args, **_kwargs):
        calls.append("shadow")
        return {
            "status": "completed",
            "database_mutated": True,
            "refresh_failures": [],
            "prediction_run": {"status": "created", "run_id": 5},
            "outcomes_added_by_horizon": {"1": 2, "5": 1, "20": 0},
            "sample_status": "insufficient_forward_sample",
        }

    def portfolio_status(*_args, **_kwargs):
        calls.append("portfolio_status")
        return _portfolio_status(None if cohort_created else 5)

    def cohort_plan(*_args, **_kwargs):
        calls.append("cohort_dry_run")
        return {"status": "ready", "plan": {"prediction_date": "2026-07-14"}}

    def cohort_create(*_args, **_kwargs):
        nonlocal cohort_created
        calls.append("cohort_create")
        cohort_created = True
        return {"status": "recorded", "cohort_id": 1, "prediction_run_id": 5}

    def portfolio_outcome_plan(*_args, **_kwargs):
        calls.append("portfolio_outcome_dry_run")
        return {"status": "ready", "outcomes_planned": 1, "pending_outcomes": 0}

    def portfolio_outcome_apply(*_args, **_kwargs):
        calls.append("portfolio_outcome_apply")
        return {"status": "recorded", "outcomes_created": 1, "pending_outcomes": 0}

    option_status_calls = 0

    def options_status(*_args, **_kwargs):
        nonlocal option_status_calls
        calls.append("options_status")
        option_status_calls += 1
        return _options_status(1 if option_status_calls > 1 else None)

    def options_collect(*_args, **kwargs):
        calls.append("options_collect")
        assert kwargs["apply"] is True
        return {
            "status": "completed",
            "snapshot_date": "2026-07-14",
            "run_id": 1,
            "contract_count": 100,
            "successful_tickers": [{"ticker": "AAPL"}],
            "failed_tickers": [],
            "network_calls_made": True,
            "database_mutated": True,
        }

    monkeypatch.setattr(research_cycle, "run_daily_shadow_cycle", shadow)
    monkeypatch.setattr(research_cycle, "portfolio_shadow_status", portfolio_status)
    monkeypatch.setattr(research_cycle, "dry_run_cohort", cohort_plan)
    monkeypatch.setattr(research_cycle, "create_cohort", cohort_create)
    monkeypatch.setattr(research_cycle, "dry_run_portfolio_outcomes", portfolio_outcome_plan)
    monkeypatch.setattr(research_cycle, "apply_portfolio_outcomes", portfolio_outcome_apply)
    monkeypatch.setattr(research_cycle, "options_status_report", options_status)
    monkeypatch.setattr(research_cycle, "collect_options_snapshots", options_collect)

    report = run_daily_research_cycle(
        tmp_path / "alpha.db",
        apply=True,
        refresh_market_data=True,
        reference_time=REFERENCE_TIME,
    )

    assert report["status"] == "healthy"
    assert report["shadow_run"] == {"status": "created", "run_id": 5}
    assert report["shadow_outcomes_added"] == {"1": 2, "5": 1, "20": 0}
    assert report["portfolio_cohorts"]["created_count"] == 1
    assert report["portfolio_outcomes_matured"] == 1
    assert report["options_snapshot"]["status"] == "completed"
    assert calls.index("shadow") < calls.index("cohort_create")
    assert calls.index("cohort_create") < calls.index("portfolio_outcome_apply")
    assert calls.index("portfolio_outcome_apply") < calls.index("options_collect")


def test_safe_no_op_skips_duplicate_options_without_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_defaults(monkeypatch)
    monkeypatch.setattr(research_cycle, "options_status_report", lambda *_args, **_kwargs: _options_status(7))
    monkeypatch.setattr(
        research_cycle,
        "collect_options_snapshots",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("provider must not run")),
    )

    report = run_daily_research_cycle(
        tmp_path / "alpha.db",
        apply=True,
        refresh_market_data=True,
        reference_time=REFERENCE_TIME,
    )

    assert report["status"] == "no_op"
    assert report["options_snapshot"] == {
        "status": "skipped",
        "reason": "duplicate_snapshot_date_provider",
        "run_id": 7,
        "snapshot_date": "2026-07-14",
        "contract_count": 0,
    }


def test_shadow_success_with_options_failure_is_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_defaults(monkeypatch)
    monkeypatch.setattr(
        research_cycle,
        "collect_options_snapshots",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic options failure")),
    )

    report = run_daily_research_cycle(
        tmp_path / "alpha.db",
        apply=True,
        refresh_market_data=True,
        reference_time=REFERENCE_TIME,
    )

    assert report["status"] == "partial_failure"
    assert report["components"]["shadow"]["status"] == "no_op"
    assert report["components"]["options"]["status"] == "failed"
    assert report["errors"] == [{"component": "options", "error": "synthetic options failure"}]


def test_run_five_is_first_planned_cohort_and_earlier_runs_are_not_considered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_defaults(monkeypatch)
    observed: list[int] = []
    monkeypatch.setattr(
        research_cycle,
        "portfolio_shadow_status",
        lambda *_args, **_kwargs: _portfolio_status(5),
    )

    def plan(_db_path, run_id):
        observed.append(run_id)
        return {"status": "ready", "plan": {"prediction_date": "2026-07-14"}}

    monkeypatch.setattr(research_cycle, "dry_run_cohort", plan)

    report = run_daily_research_cycle(
        tmp_path / "alpha.db",
        reference_time=REFERENCE_TIME,
    )

    assert observed == [5]
    assert report["portfolio_cohorts"]["planned_count"] == 1


def test_dry_run_is_non_mutating_and_never_authorizes_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_defaults(monkeypatch)
    shadow_args: list[dict] = []
    options_args: list[dict] = []

    def shadow(*_args, **kwargs):
        shadow_args.append(kwargs)
        return {
            "status": "dry_run_complete",
            "database_mutated": False,
            "refresh_failures": [],
            "prediction_run": {"status": "planned"},
            "outcomes_added_by_horizon": {"1": 0, "5": 0, "20": 0},
            "sample_status": "insufficient_forward_sample",
        }

    def options(*_args, **kwargs):
        options_args.append(kwargs)
        return {
            "status": "planned",
            "snapshot_date": "2026-07-14",
            "run_id": None,
            "contract_count": 0,
            "successful_tickers": [],
            "failed_tickers": [],
            "network_calls_made": False,
            "database_mutated": False,
        }

    monkeypatch.setattr(research_cycle, "run_daily_shadow_cycle", shadow)
    monkeypatch.setattr(research_cycle, "collect_options_snapshots", options)

    report = run_daily_research_cycle(
        tmp_path / "alpha.db",
        apply=False,
        refresh_market_data=True,
        reference_time=REFERENCE_TIME,
    )

    assert report["status"] == "healthy"
    assert report["database_mutated"] is False
    assert report["network_authorized"] is False
    assert shadow_args[0]["apply"] is False
    assert shadow_args[0]["refresh_market_data"] is False
    assert options_args[0]["apply"] is False


def test_apply_without_refresh_skips_options_and_cannot_call_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_defaults(monkeypatch)
    monkeypatch.setattr(
        research_cycle,
        "collect_options_snapshots",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network path called")),
    )

    report = run_daily_research_cycle(
        tmp_path / "alpha.db",
        apply=True,
        refresh_market_data=False,
        reference_time=REFERENCE_TIME,
    )

    assert report["status"] == "no_op"
    assert report["network_authorized"] is False
    assert report["options_snapshot"]["reason"] == "network_not_authorized"


def test_master_lock_blocks_concurrent_research_cycles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_defaults(monkeypatch)
    lock_path = tmp_path / "research-cycle.lock"

    with cycle_lock(lock_path):
        with pytest.raises(DailyResearchCycleError, match="lock already exists"):
            run_daily_research_cycle(
                tmp_path / "alpha.db",
                reference_time=REFERENCE_TIME,
                lock_path=lock_path,
            )


def test_combined_report_is_json_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_defaults(monkeypatch)

    first = run_daily_research_cycle(tmp_path / "alpha.db", reference_time=REFERENCE_TIME)
    second = run_daily_research_cycle(tmp_path / "alpha.db", reference_time=REFERENCE_TIME)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert set(first) == {
        "artifact_id",
        "components",
        "database_mutated",
        "errors",
        "mode",
        "network_authorized",
        "options_snapshot",
        "policy_id",
        "portfolio_cohorts",
        "portfolio_outcomes_matured",
        "provider_failures",
        "resolved_completed_session",
        "sample_statuses",
        "shadow_outcomes_added",
        "shadow_run",
        "status",
        "warnings",
    }
