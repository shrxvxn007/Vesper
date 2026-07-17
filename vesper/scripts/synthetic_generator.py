"""Synthetic data generator for deterministic end-to-end testing.

Produces:

* ``data/price_panel.parquet`` — long-form (date, ticker) daily price-returns
  panel with sector and beta attributes.
* ``data/returns_panel.parquet`` — same as above, focused on weekly returns
  joining key for downstream features.
* ``data/filings.parquet`` — long-form (release_date, ticker, form_type,
  period_end_date, text_clean) table of *synthetic* 10-Q / 10-K MD&A text
  blobs. The MD&A text is engineered to contain risk-aligned keywords at
  controlled intervals so the cosine-similarity feature varies.
* ``data/supply_chain.json`` — NetworkX DiGraph serialised via the JSON
  schema in :mod:`data_pipeline.graph_builder`.

Everything is seeded with :func:`numpy.random.default_rng(seed=42)` so the
same generated data is byte-stable across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from ..data_pipeline.graph_builder import build_supply_chain_graph, save_graph_to_json

# ---------------------------------------------------------------------------
# Default universe
# ---------------------------------------------------------------------------

DEFAULT_UNIVERSE: Final[dict[str, str]] = {
    "AAPL": "Tech",
    "MSFT": "Software",
    "NVDA": "Semis",
    "AVGO": "Semis",
    "ORCL": "Software",
    "CRM":  "Software",
    "ADBE": "Software",
    "GOOGL": "Internet",
    "META":  "Internet",
    "AMZN":  "Internet",
    "TSLA":  "Auto_Tech",
    "INTC":  "Semis",
}

# A market-equivalent sector is always defined to absorb ticker-sector gaps.
BROAD_SECTORS: Final[tuple[str, ...]] = ("MARKET", "Tech", "Software", "Semis", "Internet", "Auto_Tech")


# Risk-aligned keyword templates designed to drive TF-IDF cosine.
RISK_KEYWORDS: Final[tuple[str, ...]] = (
    "supply chain constraints",
    "inventory shortages",
    "revenue downgrade",
    "marginal compression",
    "regulatory scrutiny",
)
NEUTRAL_KEYWORDS: Final[tuple[str, ...]] = (
    "demand strength",
    "operating leverage",
    "capacity expansion",
    "share repurchase program",
    "launching platform features",
)


@dataclass(slots=True, frozen=True)
class SyntheticConfig:
    """Configuration for the synthetic data generator.

    Attributes:
        n_days: Number of daily observations (calendar days) to simulate.
        seed: RNG seed for full reproducibility.
        n_filings_per_ticker: How many 10-Q-like filings to emit per ticker.
        release_period_days: Calendar days between consecutive filings.
        shock_probability: Probability of a risk keyword appearing in a
            given filing (controls ``nlp_decay`` variance).
    """

    n_days: int = 504
    seed: int = 42
    n_filings_per_ticker: int = 8
    release_period_days: int = 63
    shock_probability: float = 0.35

    @property
    def n_tickers(self) -> int:
        return len(DEFAULT_UNIVERSE)


# ---------------------------------------------------------------------------
# Synthetic supply chain
# ---------------------------------------------------------------------------


def _build_default_supply_chain() -> tuple[list[tuple[str, str, float]], dict[str, str]]:
    """Default 25-edge supply chain across the 12-ticker universe.

    Edge direction is supplier → customer. Weights are concrete revenue-
    dependency fractions in (0, 1]. The structure follows plausible tech
    supply chain intuition (semis → mega-caps, software → internet, etc.).
    """
    edges: list[tuple[str, str, float]] = [
        ("NVDA", "MSFT", 0.25),
        ("NVDA", "GOOGL", 0.20),
        ("NVDA", "AAPL", 0.15),
        ("AVGO", "AAPL", 0.30),
        ("AVGO", "MSFT", 0.18),
        ("INTC", "DELL_NOT_USED", 0.10),  # placeholder for non-universe node, never added
        ("TSM", "NVDA", 0.30),
        ("TSM", "AVGO", 0.25),
        ("AMZN", "NVDA", 0.18),  # large cloud customer
        ("AMZN", "MSFT", 0.22),
        ("MSFT", "ORCL", 0.10),  # software interlock (small)
        ("ADBE", "CRM", 0.15),
        ("ORCL", "CRM", 0.20),
        ("GOOGL", "META", 0.18),  # ad network overlap (illustrative)
        ("GOOGL", "AMZN", 0.12),
        ("META", "CRM", 0.10),
        ("AAPL", "TSLA", 0.18),  # Apple car-play/software supplier for TSLA
        ("TSLA", "AVGO", 0.12),  # TSLA orders semis
        ("CRM", "ADBE", 0.08),
        ("NVDA", "META", 0.20),
        ("ORCL", "ADBE", 0.10),
        ("INTC", "MSFT", 0.14),
        ("AVGO", "ORCL", 0.08),
        ("META", "GOOGL", 0.10),
        ("AMZN", "GOOGL", 0.14),
    ]

    # Filter placeholder / non-universe edges:
    valid_edges: list[tuple[str, str, float]] = []
    universe = set(DEFAULT_UNIVERSE)
    for sup, cust, w in edges:
        if sup in universe and cust in universe and sup != cust and 0.0 < w <= 1.0:
            valid_edges.append((sup, cust, w))
    return valid_edges, dict(DEFAULT_UNIVERSE)


# ---------------------------------------------------------------------------
# Synthetic market data
# ---------------------------------------------------------------------------


def _simulate_market_panel(
    config: SyntheticConfig,
    rng: np.random.Generator,
    universe_dict: dict[str, str],
) -> pd.DataFrame:
    """Simulate daily market returns panel with sector + idiosyncratic components."""
    end = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(1)
    # The synthetic data sits 1 year in the past so as-of dates look
    # realistic during research walkthroughs.
    end = end - pd.tseries.offsets.BDay(config.n_days - 1)
    start = end - pd.tseries.offsets.BDay(config.n_days - 1)
    dates = pd.bdate_range(start=start, end=end)
    n_days = len(dates)
    tickers = list(universe_dict.keys())
    sectors = list(set(universe_dict.values()))
    sectors = sectors + ["MARKET"]

    # Sector returns (one row per date, one col per sector).
    sector_ret = rng.normal(loc=0.0004, scale=0.008, size=(n_days, len(sectors)))
    sector_ret_df = pd.DataFrame(sector_ret, index=dates, columns=sectors)

    # Per-ticker constants
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        sector = universe_dict[ticker]
        beta = float(rng.uniform(0.7, 1.4))
        # Idiosyncratic noise drawn per day for this ticker.
        idio = rng.normal(loc=0.0, scale=0.012, size=n_days)
        market = sector_ret_df[sector].to_numpy()
        raw = beta * market + idio
        # Random walk price level (geometric Brownian motion surrogate).
        price = 100.0 * np.exp(np.cumsum(raw))
        for d_idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "sector": sector,
                    "raw_ret": float(raw[d_idx]),
                    "beta": beta,
                    "price": float(price[d_idx]),
                }
            )
    panel = pd.DataFrame(rows)
    return panel, sector_ret_df


# ---------------------------------------------------------------------------
# Synthetic filings
# ---------------------------------------------------------------------------


def _make_filing_text(
    *,
    ticker: str,
    rng: np.random.Generator,
    is_shock: bool,
    form_type: str,
) -> str:
    """Make a synthetic MD&A-style text blob for a single filing.

    Args:
        ticker: Issuing ticker.
        rng: Shared random generator.
        is_shock: If True, the text is engineered around RISK_KEYWORDS so the
            cosine similarity to its predecessor drops sharply.
        form_type: Either ``10-Q`` (item 2) or ``10-K`` (item 7).
    """
    base = (
        f"This {form_type} filing for {ticker} contains a detailed discussion of the company's "
        "operating performance, capital allocation strategy, and forward-looking expectations. "
        "Management analyzes revenue trends across geographies, gross margin compression, "
        "operating expenses, working capital changes, cash flow generation, share repurchase "
        "activity, and capital expenditures. The discussion of the macroeconomic backdrop "
        "includes commentary on interest rate sensitivity, sector-specific demand, and "
        "share of wallet trends among enterprise and consumer customers. "
    )
    if is_shock:
        core = (
            f"Management draws attention to ongoing {RISK_KEYWORDS[rng.integers(0, len(RISK_KEYWORDS))]} "
            f"and {RISK_KEYWORDS[rng.integers(0, len(RISK_KEYWORDS))]} which have materially "
            f"affected quarterly results and {RISK_KEYWORDS[rng.integers(0, len(RISK_KEYWORDS))]} "
            "is expected to continue through the next fiscal period. The company has revised its "
            "full-year revenue guidance downward and is implementing cost actions to mitigate "
            "the impact on operating margins."
        )
    else:
        core = (
            f"Management reports that the period reflected {NEUTRAL_KEYWORDS[rng.integers(0, len(NEUTRAL_KEYWORDS))]} "
            f"and {NEUTRAL_KEYWORDS[rng.integers(0, len(NEUTRAL_KEYWORDS))]}, "
            f"with continued {NEUTRAL_KEYWORDS[rng.integers(0, len(NEUTRAL_KEYWORDS))]}. "
            "Revenue growth was in line with prior guidance, gross margin expanded modestly, "
            "and operating cash flow conversion remained strong. Capital allocation prioritised "
            "shareholder returns through buybacks and modest dividend growth, alongside "
            "targeted investment in research and development."
        )
    return base + core


def _simulate_filings(
    config: SyntheticConfig,
    rng: np.random.Generator,
    universe_dict: dict[str, str],
    sector_ret_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate a deterministic synthetic filings table."""
    universe = list(universe_dict.keys())
    end = sector_ret_df.index.max() - pd.Timedelta(days=10)
    period = pd.Timedelta(days=config.release_period_days)
    # Force-feed an explicit `is_shock` schedule so per-ticker trajectory is
    # reproducible: each ticker gets the same default schedule with 1-2
    # isolated shocks.
    rows: list[dict[str, object]] = []
    for ticker_idx, ticker in enumerate(universe):
        for filing_idx in range(config.n_filings_per_ticker):
            base_release = end - pd.Timedelta(
                days=(config.n_filings_per_ticker - filing_idx) * config.release_period_days
            )
            jitter = pd.Timedelta(days=int(rng.integers(-3, 4)))
            release = base_release + jitter
            period_end = release - pd.Timedelta(days=config.release_period_days)
            # Knife-edge: every 4th filing for each ticker is forced to be a
            # shock; all others are neutral EXCEPT one telegraphed outlier.
            force_shock = (filing_idx == config.n_filings_per_ticker // 2) or (
                (ticker_idx + filing_idx) % 4 == 0 and filing_idx < config.n_filings_per_ticker - 1
            )
            is_shock = bool(force_shock) or bool(rng.random() < config.shock_probability)
            form_type = "10-K" if filing_idx % 4 == 3 else "10-Q"
            text = _make_filing_text(
                ticker=ticker, rng=rng, is_shock=is_shock, form_type=form_type
            )
            rows.append(
                {
                    "ticker": ticker,
                    "form_type": form_type,
                    "release_date": release,
                    "period_end_date": period_end,
                    "text_clean": text,
                    "is_synthetic_shock": is_shock,
                }
            )
    return pd.DataFrame(rows).sort_values(["ticker", "release_date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------


class SyntheticGenerator:
    """Deterministic synthetic-data driver."""

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()

    # ------------------------------------------------------------------

    def generate(self, output_dir: str | Path) -> dict[str, Path]:
        """Run the full generator and write artefacts under ``output_dir``.

        Args:
            output_dir: Destination directory.

        Returns:
            Mapping of artefact name to written path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(self.config.seed)

        edges, universe_dict = _build_default_supply_chain()
        market_panel, sector_ret_df = _simulate_market_panel(
            self.config, rng, universe_dict
        )

        # Weekly-aggregated returns panel for downstream targets.
        weekly = (
            market_panel.groupby(["ticker", pd.Grouper(key="date", freq="W-FRI")])
            .agg(raw_ret=("raw_ret", "sum"), sector=("sector", "first"), beta=("beta", "first"))
            .reset_index()
            .rename(columns={"date": "date"})
        )
        weekly_sector_ret = (
            sector_ret_df.groupby(pd.Grouper(freq="W-FRI")).sum()
        )

        filings = _simulate_filings(
            self.config, rng, universe_dict, sector_ret_df
        )

        # Build the supply chain digraph.
        graph = build_supply_chain_graph(
            edges,
            ticker_to_sector=universe_dict,
            point_in_time_universe=tuple(universe_dict.keys()),
            constituents_as_of="2024-01-01",
        )

        # Persist everything.
        paths: dict[str, Path] = {}
        market_path = output_dir / "price_panel.parquet"
        market_panel.to_parquet(market_path, index=False)
        paths["price_panel"] = market_path

        weekly_path = output_dir / "weekly_returns.parquet"
        weekly.to_parquet(weekly_path, index=False)
        paths["weekly_returns"] = weekly_path

        sector_path = output_dir / "weekly_sector_returns.parquet"
        weekly_sector_ret.reset_index().rename(columns={"index": "date"}).to_parquet(
            sector_path, index=False
        )
        paths["weekly_sector_returns"] = sector_path

        filings_path = output_dir / "filings.parquet"
        filings.to_parquet(filings_path, index=False)
        paths["filings"] = filings_path

        graph_path = output_dir / "supply_chain.json"
        save_graph_to_json(graph, graph_path)
        paths["supply_chain"] = graph_path

        return paths


def generate_synthetic_dataset(output_dir: str | Path = "data") -> dict[str, Path]:
    """Convenience function: ``SyntheticGenerator().generate(output_dir)``."""
    return SyntheticGenerator().generate(output_dir)


__all__ = [
    "DEFAULT_UNIVERSE",
    "BROAD_SECTORS",
    "RISK_KEYWORDS",
    "NEUTRAL_KEYWORDS",
    "SyntheticConfig",
    "SyntheticGenerator",
    "generate_synthetic_dataset",
]
