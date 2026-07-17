"""Synthetic supply-chain directed-graph builder.

Models inter-firm customer-supplier relationships for our universe as a
NetworkX :class:`~networkx.DiGraph`:

* **Nodes** carry attributes ``ticker`` and ``sector``.
* **Edges** carry attribute ``weight`` (revenue dependency, in ``[0, 1]``).
  Edge direction encodes *supplier → customer*. So if ``TSMC --0.30--> AAPL``,
  TSMC supplies 30% of AAPL's revenue dependency exposure.

This module supports two construction paths:

1. :func:`build_supply_chain_graph` — build from an explicit ``(supplier,
   customer, weight)`` edge list with optional sector union.
2. :func:`build_supply_chain_graph_from_json` — load from a persisted JSON
   file produced by :func:`save_graph_to_json`.

Both paths accept a *point-in-time constituent list* hook so that callers can
inject the historical ticker universe to mitigate survivorship bias (see the
:attr:`DiGraph.graph["constituents_as_of"]` annotation below).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import networkx as nx


# Edge tuples are immutable dataclass-like tuples for type safety.
EdgeTriple = tuple[str, str, float]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_supply_chain_graph(
    edges: Sequence[EdgeTriple],
    *,
    ticker_to_sector: Mapping[str, str] | None = None,
    point_in_time_universe: Sequence[str] | None = None,
    constituents_as_of: str | None = None,
) -> nx.DiGraph:
    """Construct a directed supply-chain graph from explicit edge triples.

    Args:
        edges: Sequence of ``(supplier, customer, weight)`` triples. Weights
            should be in ``(0, 1]`` and represent the fraction of the customer's
            revenue attributed to the supplier.
        ticker_to_sector: Optional mapping ``ticker -> sector`` used to attach
            a ``sector`` attribute to every node. Unknown tickers default to
            ``"UNKNOWN"``.
        point_in_time_universe: Optional iterable of tickers that were
            *tradable at constituents_as_of*. Carried in the graph metadata so
            :class:`alpha_model.cross_sectional_model.AlphaModel` can later
            apply an ``is_tradable`` filter (see anti-trapping controls).
        constituents_as_of: ISO date string for the constituent snapshot passed
            via ``point_in_time_universe``.

    Returns:
        A :class:`networkx.DiGraph` with nodes storing ``ticker`` and
        ``sector`` attributes and edges storing ``weight`` attributes.

    Raises:
        ValueError: If any edge weight is non-positive or exceeds 1.
    """
    graph: nx.DiGraph = nx.DiGraph()
    sector_map: dict[str, str] = dict(ticker_to_sector or {})

    # First pass: register nodes from both endpoints.
    seen: set[str] = set()
    for supplier, customer, weight in edges:
        seen.add(supplier)
        seen.add(customer)

    for ticker in seen:
        graph.add_node(ticker, ticker=ticker, sector=sector_map.get(ticker, "UNKNOWN"))

    # Second pass: add weighted edges.
    for supplier, customer, weight in edges:
        if not (0.0 < weight <= 1.0):
            raise ValueError(
                f"Edge weight must be in (0, 1]; got ({supplier!r}, "
                f"{customer!r}, {weight})"
            )
        graph.add_edge(supplier, customer, weight=float(weight))

    if point_in_time_universe is not None:
        graph.graph["point_in_time_universe"] = tuple(point_in_time_universe)
        graph.graph["constituents_as_of"] = constituents_as_of
    return graph


def build_supply_chain_graph_from_json(path: str | Path) -> nx.DiGraph:
    """Load a supply-chain graph from a JSON file.

    JSON schema::

        {
          "edges": [
            {"supplier": "TSM", "customer": "AAPL", "weight": 0.3},
            ...
          ],
          "ticker_to_sector": {"AAPL": "Tech", ...},
          "point_in_time_universe": ["AAPL", ...],   # optional
          "constituents_as_of": "2024-01-01"          # optional
        }

    Args:
        path: Path to the JSON file.

    Returns:
        A :class:`networkx.DiGraph`.
    """
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    edges_raw = payload.get("edges", [])
    edges: list[EdgeTriple] = [
        (e["supplier"], e["customer"], float(e["weight"])) for e in edges_raw
    ]
    return build_supply_chain_graph(
        edges,
        ticker_to_sector=payload.get("ticker_to_sector"),
        point_in_time_universe=payload.get("point_in_time_universe"),
        constituents_as_of=payload.get("constituents_as_of"),
    )


def save_graph_to_json(graph: nx.DiGraph, path: str | Path) -> None:
    """Serialise a :class:`networkx.DiGraph` to the JSON schema above.

    Args:
        graph: The graph to serialise.
        path: Destination path. Overwrites existing files.
    """
    payload: dict[str, object] = {
        "edges": [
            {"supplier": u, "customer": v, "weight": float(d["weight"])}
            for u, v, d in graph.edges(data=True)
        ],
        "ticker_to_sector": {
            n: str(data.get("sector", "UNKNOWN"))
            for n, data in graph.nodes(data=True)
        },
    }
    if "point_in_time_universe" in graph.graph:
        payload["point_in_time_universe"] = list(
            graph.graph["point_in_time_universe"]
        )
    if "constituents_as_of" in graph.graph:
        payload["constituents_as_of"] = graph.graph["constituents_as_of"]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def adjacency_matrix(
    graph: nx.DiGraph, *, nodes: Iterable[str] | None = None
) -> tuple[list[str], "numpy.ndarray"]:  # type: ignore[name-defined]
    """Row-aligned adjacency matrix for the graph.

    Args:
        graph: A supply-chain :class:`networkx.DiGraph`.
        nodes: Optional explicit node ordering. If ``None``, we use
            ``sorted(graph.nodes)`` for determinism.

    Returns:
        Tuple ``(node_list, adjacency)`` where ``adjacency[i, j]`` is the
        edge weight from node ``node_list[i]`` to ``node_list[j]`` (or 0).
    """
    import numpy as np

    nodelist: list[str] = sorted(graph.nodes) if nodes is None else list(nodes)
    n = len(nodelist)
    adjacency = np.zeros((n, n), dtype=float)
    for i, supplier in enumerate(nodelist):
        for j, customer in enumerate(nodelist):
            data = graph.get_edge_data(supplier, customer)
            if data is not None and "weight" in data:
                adjacency[i, j] = float(data["weight"])
    return nodelist, adjacency


__all__ = [
    "EdgeTriple",
    "build_supply_chain_graph",
    "build_supply_chain_graph_from_json",
    "save_graph_to_json",
    "adjacency_matrix",
]
