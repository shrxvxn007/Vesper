"""Regression tests for the rename-defensive step 3 propagation helper.

Before the fix, ``vesper.main._run_backtest`` step 3 hardcoded
``propagate["ticker"]`` and ``propagate["graph_shock_score"]``, so a
benign upstream rename in ``propagate_shock_scores`` (e.g.,
``ticker`` -> ``asset_id`` and ``graph_shock_score`` -> ``transmitted_risk``)
silently raised a ``KeyError``. Routing ticker/score column discovery
through ``propagate.index.name`` and ``propagate.columns[-1]`` neutralises
that whole bug class.
"""

from __future__ import annotations

import pandas as pd

from vesper.main import _week_shock_records


def _propagate_frame(
    index_name: str,
    columns: tuple[str, ...],
    values: list[list[float]],
) -> pd.DataFrame:
    """Build a propagate-shaped DataFrame with the given shape.

    Mirrors what ``propagate_shock_scores`` returns: a DataFrame indexed
    by ticker, with the propagated score as the rightmost column.
    """
    return pd.DataFrame(
        dict(zip(columns, values)),
        index=pd.Index(["AAPL", "MSFT"], name=index_name),
    )


def test_week_shock_records_uses_index_name_for_ticker() -> None:
    """The ticker identity is read from ``propagate.index.name``."""
    propagate = _propagate_frame(
        index_name="ticker",
        columns=("direct_shock", "graph_shock_score"),
        values=[[0.1, 0.2], [0.3, 0.4]],
    )
    records = _week_shock_records(pd.Timestamp("2024-01-05"), propagate)

    assert len(records) == 2
    assert records[0]["ticker"] == "AAPL"
    assert records[1]["ticker"] == "MSFT"
    # The discovered score column (last column) carries the propagated value.
    assert records[0]["graph_shock_score"] == 0.3
    assert records[1]["graph_shock_score"] == 0.4


def test_week_shock_records_uses_last_column_as_score() -> None:
    """The propagated score is identified by ``propagate.columns[-1]`` only."""
    propagate = _propagate_frame(
        index_name="ticker",
        columns=("direct_shock", "graph_shock_score"),
        values=[[0.1, 0.2], [0.3, 0.4]],
    )
    records = _week_shock_records(pd.Timestamp("2024-01-05"), propagate)

    # The score column should be the LAST one (positional contract), not
    # any earlier column. list comp yields all non-(date, ticker) keys.
    score_keys = [k for k in records[0].keys() if k not in {"date", "ticker"}]
    assert score_keys == ["graph_shock_score"], (
        f"unexpected score keys: {score_keys} (positional contract broken)"
    )


def test_week_shock_records_survives_full_upstream_rename() -> None:
    """If upstream renames ``ticker`` -> ``asset_id`` AND the score column
    (``graph_shock_score`` -> ``transmitted_risk``), the helper must
    silently adapt — no ``KeyError``; the new names appear in the records.
    """
    propagate = _propagate_frame(
        index_name="asset_id",
        columns=("echo", "transmitted_risk"),
        values=[[0.1, 0.2], [0.3, 0.4]],
    )
    records = _week_shock_records(pd.Timestamp("2024-01-05"), propagate)

    assert len(records) == 2
    # The ticker column is still produced with the canonical key "ticker"
    # (we always emit "ticker" downstream — rename-defense is in INPUT, not
    # output of the helper).
    assert records[0]["ticker"] == "AAPL"
    assert records[1]["ticker"] == "MSFT"
    # The score column now carries the *renamed* upstream key, NOT the old.
    assert records[0]["transmitted_risk"] == 0.3
    assert records[1]["transmitted_risk"] == 0.4
    # The old name must NOT appear in any record.
    assert "graph_shock_score" not in records[0]
    assert "asset_id" not in records[0]


def test_week_shock_records_handles_empty_propagate() -> None:
    """Empty propagate yields an empty list (no IndexError on
    ``columns[-1]`` because that indexes the *schema*, not the rows).
    """
    empty = pd.DataFrame(
        {"direct_shock": [], "graph_shock_score": []},
        index=pd.Index([], name="ticker"),
    )
    assert _week_shock_records(pd.Timestamp("2024-01-05"), empty) == []


def test_week_shock_records_handles_unnamed_index() -> None:
    """If the propagator forgot to name its index, fall back to ``"ticker"``
    rather than producing dict entries with the literal key ``"index"``
    (which would silently look like a valid ticker string downstream).
    """
    propagate = pd.DataFrame(
        {"direct_shock": [0.1, 0.2], "graph_shock_score": [0.3, 0.4]},
        # No index.name set.
        index=pd.Index(["AAPL", "MSFT"]),
    )
    records = _week_shock_records(pd.Timestamp("2024-01-05"), propagate)

    # Tickers come straight from the index values regardless of name.
    assert records[0]["ticker"] == "AAPL"
    assert records[1]["ticker"] == "MSFT"
    assert records[0]["graph_shock_score"] == 0.3
