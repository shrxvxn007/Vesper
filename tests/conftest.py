"""Shared pytest fixtures for the Vesper test suite.

All fixtures are *offline*: they produce small but realistic synthetic
artefacts so tests run without internet or external services.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vesper.data_pipeline.graph_builder import build_supply_chain_graph
from vesper.features.nlp_decay import compute_information_decay
from vesper.features.shock_propagation import propagate_shock_scores


# ---------------------------------------------------------------------------
# Mini-universe fixtures
# ---------------------------------------------------------------------------

MINI_UNIVERSE = {
    "AAA": "Tech",
    "BBB": "Software",
    "CCC": "Semis",
    "DDD": "Internet",
}


@pytest.fixture()
def mini_universe() -> dict[str, str]:
    return dict(MINI_UNIVERSE)


@pytest.fixture()
def mini_supply_chain(mini_universe: dict[str, str]) -> object:
    edges = [
        ("CCC", "AAA", 0.30),
        ("CCC", "BBB", 0.20),
        ("BBB", "AAA", 0.15),
        ("DDD", "AAA", 0.10),
        ("AAA", "BBB", 0.05),
    ]
    return build_supply_chain_graph(
        edges,
        ticker_to_sector=mini_universe,
        point_in_time_universe=tuple(mini_universe.keys()),
        constituents_as_of="2024-01-01",
    )


@pytest.fixture()
def mini_filings(mini_universe: dict[str, str]) -> pd.DataFrame:
    rows = []
    base_ts = pd.Timestamp("2024-01-01")
    for ticker in mini_universe.keys():
        for i in range(4):
            rows.append(
                {
                    "ticker": ticker,
                    "release_date": base_ts + pd.Timedelta(days=63 * i),
                    "form_type": "10-K" if i == 3 else "10-Q",
                    "text_clean": (
                        f"This is the {ticker} quarter {i} MD&A. We discuss revenue trends, "
                        "operating expenses, capital allocation, and forward guidance. "
                        + (
                            "There are supply chain constraints impacting gross margin."
                            if (i == 1 and ticker == "CCC")
                            else "Operating leverage remains strong across geographies."
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def mini_nlp_decay(mini_filings: pd.DataFrame) -> pd.DataFrame:
    return compute_information_decay(mini_filings, text_column="text_clean")


@pytest.fixture()
def mini_graph_shock(mini_nlp_decay: pd.DataFrame, mini_supply_chain) -> pd.DataFrame:
    rows = []
    for date in mini_nlp_decay.index.get_level_values(0).unique():
        direct = mini_nlp_decay.loc[date]["nlp_decay_score"]
        out = propagate_shock_scores(direct, mini_supply_chain)
        for ticker, row in out.iterrows():
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "graph_shock_score": float(row["graph_shock_score"]),
                }
            )
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()


# ---------------------------------------------------------------------------
# Weekly panel fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def mini_weekly_panel(mini_universe: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (weekly_returns_df, weekly_sector_returns_df)."""
    rng = np.random.default_rng(123)
    dates = pd.bdate_range(start="2024-01-05", periods=20, freq="B")
    weekly_dates = pd.date_range(start=dates[0].normalize(), periods=4, freq="W-FRI")

    rows = []
    for ticker, sector in mini_universe.items():
        for d in weekly_dates:
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "sector": sector,
                    "raw_ret": float(rng.normal(0.001, 0.01)),
                    "beta": float(rng.uniform(0.8, 1.4)),
                }
            )
    weekly = pd.DataFrame(rows)

    # Synthetic sector returns: one row per week, one col per sector + MARKET.
    sectors = list(set(mini_universe.values())) + ["MARKET"]
    sec_ret = rng.normal(0.0, 0.005, size=(len(weekly_dates), len(sectors)))
    sec_ret_df = pd.DataFrame(sec_ret, index=weekly_dates, columns=sectors)

    return weekly, sec_ret_df


# ---------------------------------------------------------------------------
# CV fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def supervised_training_df(mini_universe: dict[str, str]) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    weekly_dates = pd.date_range(start="2024-01-05", periods=20, freq="W-FRI")
    for d in weekly_dates:
        for t in mini_universe.keys():
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "nlp_decay_score": float(rng.uniform(0.0, 1.0)),
                    "graph_shock_score": float(rng.uniform(0.0, 1.0)),
                    "target": float(rng.normal(0.0, 0.02)),
                }
            )
    return pd.DataFrame(rows)
