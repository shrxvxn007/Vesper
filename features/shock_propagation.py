"""Graph shock propagation engine.

Implements a 1-step, row-normalised network diffusion algorithm that maps a
*direct* per-ticker shock (the NLP-decay risk score, in ``[0, 1]``) onto the
rest of the universe.

Edge semantics:
    A directed edge ``supplier -> customer`` with weight ``w`` means that the
    supplier accounts for a fraction ``w`` of the customer's revenue. We
    treat that as the *channel* along which a text-shock at the supplier
    flows to the customer; the propagated received shock at the customer is
    therefore the in-edge-weighted sum of upstream direct shocks.

Concretely, let ``A`` be the (sparse) adjacency matrix whose rows are the
*supplier* nodes and whose columns are the *customer* nodes. We row-normalise
so a supplier's outflow weights sum to at most 1 (this prevents path
amplification in graphs with density > 1/n). The received shock at every
customer is then::

    propagated = (A_norm.T) @ direct_shock

Because this is a 1-step operator, **cycles cannot cause shocks to amplify
indefinitely** — we compute a *single* application of ``A_norm.T`` per call.
For multi-hop propagation compose ``A_norm`` powers explicitly upstream.
"""

from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

from data_pipeline.graph_builder import adjacency_matrix


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def propagate_shock_scores(
    direct_shock: pd.Series,
    graph: nx.DiGraph,
    *,
    node_key: str = "ticker",
    normalize_rows: bool = True,
) -> pd.DataFrame:
    """Propagate per-ticker direct shocks one hop along supply-chain edges.

    Args:
        direct_shock: A :class:`pandas.Series` indexed by ticker whose values
            are the direct (NLP-decay) risk scores in ``[0, 1]``. Tickers not
            present will be treated as ``0.0``.
        graph: A :class:`networkx.DiGraph` built by
            :func:`data_pipeline.graph_builder.build_supply_chain_graph`.
            Edge ``weight`` attribute gives the per-edge dependency.
        node_key: Unused; kept for forward compatibility.
        normalize_rows: If ``True`` (recommended), row-normalise the
            adjacency matrix so each *supplier*'s outflow weights sum to
            ≤ 1. This caps the maximum incoming shock at any customer.

    Returns:
        :class:`pandas.DataFrame` indexed by ``ticker`` with two columns:
        ``direct_shock`` (echo of the input) and ``graph_shock_score`` (the
        propagated score, clipped to ``[0, 1]``).
    """
    del node_key  # API stability
    nodes, adjacency = adjacency_matrix(graph)

    direct_vec = np.zeros(len(nodes), dtype=float)
    for i, n in enumerate(nodes):
        if n in direct_shock.index:
            direct_vec[i] = float(direct_shock[n])

    if normalize_rows:
        row_sums = adjacency.sum(axis=1, keepdims=False)
        # Avoid division by zero — leaves rows that sum to 0 unchanged.
        safe_sums = np.where(row_sums > 0.0, row_sums, 1.0)
        adjacency = adjacency / safe_sums[:, None]

    propagated = adjacency.T @ direct_vec
    propagated = np.clip(propagated, 0.0, 1.0)

    return pd.DataFrame(
        {
            "ticker": nodes,
            "direct_shock": direct_vec,
            "graph_shock_score": propagated,
        }
    ).set_index("ticker")


class ShockPropagator:
    """Stateful wrapper around :func:`propagate_shock_scores`.

    Useful when you want to fit the propagation once and call it many times on
    streaming shock updates.
    """

    def __init__(self, graph: nx.DiGraph, *, normalize_rows: bool = True) -> None:
        self._graph = graph
        self._normalize_rows = normalize_rows
        nodes, adjacency = adjacency_matrix(graph)
        if normalize_rows:
            row_sums = adjacency.sum(axis=1, keepdims=False)
            safe_sums = np.where(row_sums > 0.0, row_sums, 1.0)
            adjacency = adjacency / safe_sums[:, None]
        self._nodes: tuple[str, ...] = tuple(nodes)
        self._transition: np.ndarray = adjacency.T  # shape (n, n)

    @property
    def nodes(self) -> tuple[str, ...]:
        return self._nodes

    def propagate(self, direct_shock: Iterable[float]) -> np.ndarray:
        vec = np.asarray(list(direct_shock), dtype=float)
        if vec.shape != (len(self._nodes),):
            raise ValueError(
                f"direct_shock must have length {len(self._nodes)}; got {vec.shape}"
            )
        out = self._transition @ vec
        return np.clip(out, 0.0, 1.0)


__all__ = ["propagate_shock_scores", "ShockPropagator"]
