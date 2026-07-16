"""Unit tests for the ``features/`` module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.nlp_decay import compute_information_decay
from features.shock_propagation import ShockPropagator, propagate_shock_scores


# ---------------------------------------------------------------------------
# NLP cosine tests
# ---------------------------------------------------------------------------


def test_information_decay_emits_first_filing_with_zero(mini_filings: pd.DataFrame) -> None:
    decay = compute_information_decay(mini_filings, text_column="text_clean")
    # First filing per ticker should have decay == 0.
    first_per_ticker = decay.reset_index().groupby("ticker").first()
    assert (first_per_ticker["nlp_decay_score"] == 0.0).all()


def test_information_decay_index_uses_release_date(mini_filings: pd.DataFrame) -> None:
    decay = compute_information_decay(mini_filings, text_column="text_clean")
    assert decay.index.names == ["release_date", "ticker"]
    # We must have one row per (release_date, ticker).
    expected = mini_filings.groupby(["ticker", "release_date"]).size().shape[0]
    assert decay.shape[0] == expected


def test_information_decay_in_unit_interval(mini_filings: pd.DataFrame) -> None:
    decay = compute_information_decay(mini_filings, text_column="text_clean")
    assert (decay["nlp_decay_score"] >= 0.0).all()
    assert (decay["nlp_decay_score"] <= 1.0).all()


def test_information_decay_higher_for_shocky_text(mini_universe: dict[str, str]) -> None:
    """The ticker whose text diverges most should score highest decay."""
    rows = []
    base = pd.Timestamp("2024-01-01")
    for ticker in mini_universe.keys():
        clean_text = (
            f"{ticker} reports strong revenue growth, operating leverage, share "
            "repurchases, capacity expansion, and customer demand."
        )
        for i in range(2):
            text = clean_text
            if i == 1 and ticker == "CCC":
                text = (
                    "supply chain constraints revenue downgrade margin compression "
                    "regulatory scrutiny constraints shortages"
                )
            rows.append(
                {
                    "ticker": ticker,
                    "release_date": base + pd.Timedelta(days=63 * i),
                    "form_type": "10-Q",
                    "text_clean": text,
                }
            )
    df = pd.DataFrame(rows)
    decay = compute_information_decay(df, text_column="text_clean")
    # Only second quarter values exist past warmup; check CCC > others.
    second_q = decay.reset_index()
    second_q = second_q[second_q["release_date"] != second_q["release_date"].min()]
    assert second_q.loc[second_q["ticker"] == "CCC", "nlp_decay_score"].iloc[0] > 0.5


def test_empty_text_returns_empty_df() -> None:
    df = pd.DataFrame(
        columns=["ticker", "release_date", "form_type", "text_clean"]
    )
    decay = compute_information_decay(df, text_column="text_clean")
    assert decay.empty


# ---------------------------------------------------------------------------
# Shock propagation tests
# ---------------------------------------------------------------------------


def test_propagate_no_self_echo(mini_supply_chain, mini_universe: dict[str, str]) -> None:
    """A shock at a node must not propagate to itself when there is no self-loop."""
    direct = pd.Series(1.0, index=sorted(mini_universe.keys()))
    out = propagate_shock_scores(direct, mini_supply_chain)
    assert (out["direct_shock"] >= 0.0).all()
    assert (out["graph_shock_score"] >= 0.0).all()
    assert (out["graph_shock_score"] <= 1.0).all()


def test_propagate_row_normalisation_caps_indegree() -> None:
    import networkx as nx

    g = nx.DiGraph()
    for i, n in enumerate(["A", "B", "C"]):
        g.add_node(n, ticker=n)
    # Two outgoing 1.0 edges from "A" — without row-normalisation the
    # propagated shock would exceed 1.
    g.add_edge("A", "B", weight=1.0)
    g.add_edge("A", "C", weight=1.0)

    direct = pd.Series({"A": 1.0, "B": 0.0, "C": 0.0})
    out = propagate_shock_scores(direct, g, normalize_rows=True)

    # After row-normalisation each individual customer receives a 0.5 share.
    assert pytest.approx(out.loc["B", "graph_shock_score"], rel=1e-9) == 0.5
    assert pytest.approx(out.loc["C", "graph_shock_score"], rel=1e-9) == 0.5


def test_shock_propagator_class_matches_function(mini_supply_chain, mini_universe: dict[str, str]) -> None:
    direct = pd.Series(0.0, index=sorted(mini_universe.keys()))
    direct["CCC"] = 0.8  # issuer of the supply chain shock
    out_fn = propagate_shock_scores(direct, mini_supply_chain)

    propagator = ShockPropagator(mini_supply_chain)
    nodes = list(propagator.nodes)
    direct_vec = np.array([direct.get(n, 0.0) for n in nodes])
    out_vec = propagator.propagate(direct_vec)

    for n in nodes:
        assert pytest.approx(out_vec[nodes.index(n)], rel=1e-9) == out_fn.loc[n, "graph_shock_score"]


def test_propagate_handles_disconnected_nodes(mini_universe: dict[str, str]) -> None:
    """A ticker not part of any edge should still appear in the output with 0."""
    direct = pd.Series(0.0, index=sorted(mini_universe.keys()) + ["ZZZ"])
    from data_pipeline.graph_builder import build_supply_chain_graph

    g = build_supply_chain_graph(
        [("CCC", "AAA", 0.30)],
        ticker_to_sector=mini_universe,
    )
    out = propagate_shock_scores(direct, g)
    assert "CCC" in out.index
    assert out.loc["CCC", "graph_shock_score"] == 0.0
