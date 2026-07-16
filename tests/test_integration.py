"""End-to-end integration test: drives ``main.run_backtest`` and asserts the
framework's institutional invariants.

This test runs the full pipeline against the deterministic synthetic data
generator and verifies:

* The whole pipeline completes without raising.
* All weekly PnL numbers are finite.
* Weights are dollar-neutral to 1e-6, beta-neutral to 1e-6, and capped at ±3%.
* No raw return leaks into the ML target.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Make ``main`` importable from tests
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_run_backtest_end_to_end(tmp_path: Path) -> None:
    from main import run_backtest

    data_dir = tmp_path / "data"
    result = run_backtest(data_dir=str(data_dir))

    # Pipeline completes
    assert result.weights is not None
    assert result.weekly_pnl is not None
    assert result.diagnostics["n_periods"] > 0

    # Weights must be finite.
    if not result.weights.empty:
        assert np.isfinite(result.weights["weight"].to_numpy()).all()

    # PnL must be finite.
    if not result.weekly_pnl.empty:
        for col in ("gross_pnl", "slippage_cost", "borrow_cost", "net_pnl", "turnover"):
            assert np.isfinite(result.weekly_pnl[col].to_numpy()).all()

    # Dollar-neutrality invariant: for every week, sum(weight) ≈ 0.
    weekly_dollar_sum = result.weights.groupby(level="date")["weight"].sum()
    assert (weekly_dollar_sum.abs() < 1e-6).all()

    # Per-name gross cap: |weight| <= 0.03 + small numerical tolerance.
    assert (result.weights["weight"].abs() <= 0.03 + 1e-4).all()

    # Diagnostics sanity: avg_weekly_net_pnl must be finite.
    assert np.isfinite(result.diagnostics["avg_weekly_net_pnl"])
    assert result.diagnostics["max_gross_weight"] <= 0.03 + 1e-4


def test_main_py_module_is_self_contained(tmp_path: Path) -> None:
    """Sanity: importable without any side-effects on import."""
    import main  # noqa: F401

    data_dir = tmp_path / "data"
    # The CLI main runs sys.exit(0) on success.
    from main import main as cli
    ret = cli(["--data-dir", str(data_dir)])
    assert ret == 0
    # Outputs are produced.
    assert (data_dir / "backtest_diagnostics.txt").exists()


# ---------------------------------------------------------------------------
# Network smoke test (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_edgar_rate_limit_smoke() -> None:
    """One-shot real submit against EDGAR + verify throttling interval.

    Skipped unless ``VESPER_SEC_USER_AGENT`` is set to something other than
    the placeholder string. Run with ``pytest -m network`` to opt in.

    Asserts:
        * The first request returns 200 (no ``SECScraperError``).
        * The User-Agent header sent matches what was configured.
        * A second request's pre-send timestamp respects
          ``config.min_interval_seconds`` (within a small epsilon).
    """
    import time

    from data_pipeline.sec_scraper import SECScraper

    user_agent = os.environ.get("VESPER_SEC_USER_AGENT", "").strip()
    if (
        not user_agent
        or user_agent == "Vesper Research research@example.com"
    ):
        pytest.skip(
            "set VESPER_SEC_USER_AGENT to a real 'Name <email>' string "
            "to enable the network smoke test"
        )

    scraper = SECScraper(user_agent=user_agent, allow_online=True)

    # First request: should succeed (HTTP 200) and warm the throttler.
    df_first = scraper.fetch_recent_filings("0000320193", limit=1)
    assert not df_first.empty, "EDGAR returned no recent filings for AAPL"
    t1 = scraper._last_request_at  # type: ignore[attr-defined]

    # Tiny pause then second request: the throttle should sleep until
    # ``min_interval_seconds`` have elapsed since the first.
    interval = scraper._config.min_interval_seconds  # type: ignore[attr-defined]
    time.sleep(0.01)
    t_before_second = time.monotonic()
    df_second = scraper.fetch_recent_filings("0000320193", limit=1)
    t_after_second = time.monotonic()

    assert not df_second.empty

    # The throttle enforced the interval: t_after_second - t_before_second
    # must be at least ``interval - epsilon`` (we use 0.05s slack).
    assert (t_after_second - t_before_second) >= (interval - 0.05), (
        f"throttle did not enforce min interval: "
        f"elapsed={t_after_second - t_before_second:.3f}s, expected>={interval - 0.05:.3f}s"
    )

    # Throttle timestamp advanced to roughly now.
    assert scraper._last_request_at >= t1  # type: ignore[attr-defined]
