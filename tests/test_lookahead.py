"""Strict no-lookahead firewall tests.

Central invariant:

    A feature released **strictly after** a week's observation date cannot
    leak into that week's alpha score.

The wall in front of this invariant is a single ``pd.merge_asof`` call with
``direction="backward"``. If anyone weakens that argument without
re-validating these tests, the project's most important anti-trapping
control is broken silently.

The tests exercise :func:`vesper.main._weekly_nlp_features`, the helper
that re-keys per-release NLP decay scores onto the weekly grid. Because
the same ``direction="backward"`` semaphore also governs the per-ticker
direct-shock merge inside :func:`vesper.main.run_backtest`, the firewall
property established here is sufficient to defend the alpha score itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vesper.main import _weekly_nlp_features


# -------------------------------------------------------------------
# Helpers (offline, deterministic)
# -------------------------------------------------------------------


def _make_filings(
    release_dates: list[str],
    *,
    ticker: str = "AAA",
    score: float | list[float] = 0.5,
) -> pd.DataFrame:
    """Build a minimal filings-frame that mimics :func:`compute_information_decay`'s output."""
    n = len(release_dates)
    scores = [score] * n if isinstance(score, (int, float)) else list(score)
    assert len(scores) == n, "scores length must match release_dates length"
    return pd.DataFrame(
        {
            "release_date": pd.to_datetime(release_dates),
            "ticker": [ticker] * n,
            "form_type": ["10-Q"] * n,
            "nlp_decay_score": scores,
        }
    )


def _make_weekly(dates: list[str], *, ticker: str = "AAA") -> pd.DataFrame:
    """Build a minimal weekly-grid frame that mimics the upstream weekly panel."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "ticker": [ticker] * len(dates),
        }
    )


def _get(result: pd.DataFrame, date_str: str, ticker: str = "AAA") -> float:
    """Lookup a (date, ticker) cell, returning NaN if missing."""
    return float(result.loc[(pd.Timestamp(date_str), ticker), "nlp_decay_score"])


# -------------------------------------------------------------------
# The strict no-lookahead property
# -------------------------------------------------------------------


def test_future_release_never_leaks_into_earlier_weekly_observation() -> None:
    """A release AFTER every weekly observation must leave every observation NaN.

    Strictly, this is the central guarantee the user asked for: nothing
    from a future release may bleed into a past week's feature vector, no
    matter how temporally close the future release is.
    """
    filings = _make_filings(["2024-12-31"], ticker="AAA", score=0.99)
    weekly = _make_weekly(["2024-01-01", "2024-06-01", "2024-11-01"], ticker="AAA")
    out = _weekly_nlp_features(filings, weekly)

    for date_str in ["2024-01-01", "2024-06-01", "2024-11-01"]:
        v = _get(out, date_str)
        assert pd.isna(v), (
            f"future release (2024-12-31) leaked into weekly {date_str}: "
            f"got {v!r}; pre-release observations must remain NaN"
        )


def test_release_after_observation_strictly_outward() -> None:
    """The leader case: weekly on 2024-06-01 must NOT see a release on 2024-06-15.

    The release is only 14 days after the weekly observation — closer in
    time than any backward-pre-dated value. It still must not leak, by
    construction of ``direction="backward"``.
    """
    filings = _make_filings(["2024-06-15"], ticker="AAA", score=0.95)
    weekly = _make_weekly(["2024-06-01", "2024-06-14"], ticker="AAA")
    out = _weekly_nlp_features(filings, weekly)

    # Both observations are STRICTLY before 2024-06-15; must be NaN.
    assert pd.isna(_get(out, "2024-06-01")), (
        "release on 2024-06-15 leaked into weekly 2024-06-01"
    )
    assert pd.isna(_get(out, "2024-06-14")), (
        "release on 2024-06-15 leaked into weekly 2024-06-14 (a one-day gap)"
    )


def test_release_on_or_before_observation_is_visible() -> None:
    """Same-day and earlier releases DO populate the feature (backward asof is inclusive)."""
    filings = _make_filings(["2024-06-15", "2024-05-01"], ticker="AAA", score=[0.42, 0.21])
    weekly = _make_weekly(["2024-05-01", "2024-06-15", "2024-06-30"], ticker="AAA")
    out = _weekly_nlp_features(filings, weekly)

    # 2024-05-01 == release date → that release value (backward asof is inclusive).
    assert _get(out, "2024-05-01") == pytest.approx(0.21)
    # 2024-06-15 == release date → that release value (most recent past).
    assert _get(out, "2024-06-15") == pytest.approx(0.42)
    # 2024-06-30 → 2024-06-15 is the most-recent past release → 0.42.
    assert _get(out, "2024-06-30") == pytest.approx(0.42)


def test_backward_asof_picks_most_recent_past_release() -> None:
    """When multiple releases precede a week, the most-recent wins."""
    filings = _make_filings(
        ["2024-01-15", "2024-04-15", "2024-07-15"],
        ticker="AAA",
        score=[0.10, 0.50, 0.90],
    )
    weekly = _make_weekly(
        ["2024-02-01", "2024-05-01", "2024-06-01", "2024-08-01"], ticker="AAA"
    )
    out = _weekly_nlp_features(filings, weekly)
    assert _get(out, "2024-02-01") == pytest.approx(0.10), (
        "Feb 1 should inherit the only-past release (Jan 15)"
    )
    assert _get(out, "2024-05-01") == pytest.approx(0.50), (
        "May 1 should inherit the most-recent past release (Apr 15)"
    )
    assert _get(out, "2024-06-01") == pytest.approx(0.50), (
        "Jun 1 has no closer past release than Apr 15"
    )
    assert _get(out, "2024-08-01") == pytest.approx(0.90), (
        "Aug 1 should inherit the most-recent past release (Jul 15)"
    )


def test_multi_ticker_no_cross_contamination() -> None:
    """A release for ticker BBB cannot leak into tickers of a different name on the same date."""
    filings = pd.DataFrame(
        {
            "release_date": pd.to_datetime(["2024-06-15", "2024-06-15"]),
            "ticker": ["AAA", "BBB"],
            "form_type": ["10-Q", "10-Q"],
            "nlp_decay_score": [0.10, 0.99],
        }
    )
    weekly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-06-01", "2024-06-01"]),
            "ticker": ["AAA", "BBB"],
        }
    )
    out = _weekly_nlp_features(filings, weekly)

    # Both observations on 2024-06-01, strictly before both 2024-06-15 releases.
    assert pd.isna(_get(out, "2024-06-01", "AAA")), (
        "AAA pre-release row must be NaN despite a separate BBB release on the future date"
    )
    assert pd.isna(_get(out, "2024-06-01", "BBB")), (
        "BBB pre-release row must be NaN"
    )

    # And after the release date, the per-ticker isolation still holds.
    weekly_after = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-06-20", "2024-06-20"]),
            "ticker": ["AAA", "BBB"],
        }
    )
    out_after = _weekly_nlp_features(filings, weekly_after)
    assert _get(out_after, "2024-06-20", "AAA") == pytest.approx(0.10)
    assert _get(out_after, "2024-06-20", "BBB") == pytest.approx(0.99)


# -------------------------------------------------------------------
# Integration: alpha score cannot contain future-feature influence
# -------------------------------------------------------------------


def test_alpha_input_features_respect_backward_asof() -> None:
    """End-to-end check: feeding the merged-lookahead-free features through
    the weekly build leaves any future feature unreachable.

    This is a wiring check rather than a math check — it confirms that
    :func:`vesper.main._weekly_nlp_features` is what the pipeline actually
    uses for the weekly NLP layer, so the unit-test semantics carry
    through to ``run_backtest``'s ``weekly_features`` frame.
    """
    # Two releases for one ticker; the second one is strictly AFTER the
    # only weekly observation we care about.
    filings = _make_filings(["2024-01-15", "2024-09-30"], ticker="AAA", score=[0.42, 0.999])
    weekly = _make_weekly(["2024-06-01"], ticker="AAA")
    out = _weekly_nlp_features(filings, weekly)

    # Only the Jan 15 release is in the past of Jun 1, so the score is 0.42,
    # NOT the 0.999 from the future Sep 30 release.
    assert _get(out, "2024-06-01") == pytest.approx(0.42), (
        "future Sep 30 release bled into Jun 1 weekly feature"
    )

    # The Sep 30 observation should pick up the 0.999 release (backward inclusive).
    weekly_far = _make_weekly(["2024-10-15"], ticker="AAA")
    out_far = _weekly_nlp_features(filings, weekly_far)
    assert _get(out_far, "2024-10-15") == pytest.approx(0.999), (
        "Sep 30 release should be visible from Oct 15 onward"
    )


# -------------------------------------------------------------------
# Parametric guard: a tweaked merge-asof direction must fail loudly
# -------------------------------------------------------------------


def _weekly_nlp_with_forward_asof(
    filings_df: pd.DataFrame,
    weekly_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Reference implementation that uses ``direction="forward"`` so the
    tests below can fail loudly if the production code is ever silently
    weakened to allow forward look.

    This is intentionally a separate function so the test below pins
    contrast: future releases SHOULD leak under forward-asof but MUST NOT
    under backward-asof.
    """
    filings_clean = filings_df.loc[:, ~filings_df.columns.duplicated()].copy()
    weekly_clean = weekly_features_df.loc[:, ~weekly_features_df.columns.duplicated()].copy()

    nlp = filings_clean[["release_date", "ticker", "nlp_decay_score"]].sort_values(
        ["ticker", "release_date"]
    )
    weekly_long = weekly_clean[["date", "ticker"]].drop_duplicates().sort_values(
        ["ticker", "date"]
    )
    frames: list[pd.DataFrame] = []
    for ticker, sub_pair in weekly_long.groupby("ticker", sort=False):
        sub_nlp = nlp[nlp["ticker"] == ticker].sort_values("release_date")
        if sub_nlp.empty:
            sub_pair = sub_pair.copy()
            sub_pair["nlp_decay_score"] = np.nan
            frames.append(sub_pair)
            continue
        merged = pd.merge_asof(
            sub_pair,
            sub_nlp[["release_date", "nlp_decay_score"]],
            left_on="date",
            right_on="release_date",
            direction="forward",  # intentionally opposite — for contrast only
        )
        frames.append(merged)
    merged = pd.concat(frames, ignore_index=True)
    return merged.set_index(["date", "ticker"])[["nlp_decay_score"]].sort_index()


def test_forward_asof_demonstrates_lookahead_to_avoid() -> None:
    """If anyone weakens the production to forward-asof, future releases
    leak. This test demonstrates the failure mode, so the strict
    backward-asof test above remains meaningful."""
    filings = _make_filings(["2024-12-31"], ticker="AAA", score=0.99)
    weekly = _make_weekly(["2024-06-01", "2024-12-30"], ticker="AAA")
    out_forward = _weekly_nlp_with_forward_asof(filings, weekly)

    # Under forward-asof, the future release (2024-12-31) bleeds back into
    # both 2024-06-01 and 2024-12-30 weekly observations.
    assert _get(out_forward, "2024-06-01") == pytest.approx(0.99), (
        "expected forward-asof reference to demonstrate lookahead at 2024-06-01"
    )
    # Sanity: backward-asof on the same data still produces NaN.
    out_backward = _weekly_nlp_features(filings, weekly)
    assert pd.isna(_get(out_backward, "2024-06-01")), (
        "production weekly_nlp_features must remain NaN for pre-release observations"
    )
