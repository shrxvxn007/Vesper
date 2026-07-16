# Vesper — Alternative Data Information Arbitrage Signal Engine

Vesper is a modular, production-grade Python framework for **mid-frequency
statistical arbitrage** built around alternative-data signals. It produces a
weekly cross-sectional alpha score from SEC filings and a synthetic supply-
chain graph, then runs an institutional-grade convex portfolio allocator.

## Features

- **SEC EDGAR ingestion** with a SEC-compliant `User-Agent` (offline by
  default; see `data_pipeline/sec_scraper.py`).
- **MD&A extraction** for 10-Q (Item 2) and 10-K (Item 7) via BeautifulSoup
  with regex-anchored heading detection (`data_pipeline/mda_parser.py`).
- **Information-decay factor** computed on top of TF-IDF cosine similarity
  between consecutive quarterly filings (`features/nlp_decay.py`).
- **Graph shock propagation** — row-normalised 1-step directed diffusion
  across the customer/supplier graph (`features/shock_propagation.py`).
- **Idiosyncratic-return targets** via rolling-window OLS residuals
  (`alpha_model/target_formulation.py`).
- **PurgedGroupTimeSeriesSplit** + L2-regularised Ridge for the alpha model
  (`alpha_model/cross_sectional_model.py`).
- **Sector factor neutralization** via OLS-residualisation
  (`portfolio/factor_neutralization.py`).
- **Convex allocator** with cvxpy: $-neutral, $-neutral, ±3% per-name caps
  and linear transaction-cost penalty (`portfolio/convex_optimizer.py`).
- **Deterministic synthetic data generator** so the framework runs
  end-to-end out of the box without hitting SEC
  (`scripts/synthetic_generator.py`).
- **Unit + integration tests** covering all math invariants
  (`tests/`).

## Quick start

```bash
# Recommended: create a virtual environment first.
python -m venv .venv && source .venv/bin/activate

# Install runtime + test dependencies.
pip install -r requirements.txt

# Run the full backtest end-to-end on synthetic data.
python main.py --data-dir data

# Run the test suite.
pytest tests -v
```

`main.py` will (re)generate the synthetic universe under `data/` if it
doesn't already exist. The generator is seeded (`np.random.default_rng(42)`)
so re-runs are byte-stable.

## Project structure

```
.
├── data_pipeline/      # SEC scraper, MD&A parser, supply-chain graph
├── features/           # NLP cosine decay, graph shock propagation
├── alpha_model/        # Idiosyncratic targets + purged time-series CV
├── portfolio/          # Sector neutralization + convex allocator
├── scripts/            # Synthetic data generator (offline)
├── tests/              # Unit + integration tests
├── main.py             # End-to-end backtest driver
├── requirements.txt
├── pyproject.toml
└── pytest.ini
```

## Anti-trapping controls

- **Lookahead firewall.** Filings are keyed by `release_date` (public
  availability), not `period_end_date`. Weekly features use `merge_asof` with
  `direction="backward"` — strictly no-future information.
- **Survivorship hooks.** The supply-chain graph carries
  `point_in_time_universe` / `constituents_as_of` metadata and
  `main.py` filters the tradable universe against that set on every weekly
  rebalance (replace with a true historical source if you go live).
- **Execution realism.** Hardcoded borrow cost and bid-ask slippage defaults
  live in `TransactionCostConfig` and are applied to PnL via
  `apply_costs_to_pnl` — never silently zero.

## Testing

Run the zero-network test suite:

```bash
pytest tests -v
```

The integration test (`tests/test_integration.py`) drives `main.py` against
the synthetic generator and asserts:

- `sum(weights) == 0`  (dollar-neutrality, to 1e-6)
- `weights @ beta == 0` (beta-neutrality, to 1e-6)
- `|weight| <= 0.03` per name
- All PnL numbers are finite
- No raw returns are used as targets anywhere

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and on every pull
request. The single CI job:

1. Sets up Python 3.11 with pip caching.
2. Installs `requirements.txt`.
3. Runs `pytest tests -v` (the `network` marker is skipped by default).
4. Runs `python main.py --data-dir data` end-to-end.
5. Re-asserts `dollar_neutrality_violation < 1e-6` and
   `max_gross_weight <= 0.0301` directly from the persisted
   `data/backtest_diagnostics.txt`, so the invariants are enforced even if
   the test fixtures are refactored.
6. Executes `notebooks/evaluation.ipynb` headlessly and uploads the
   resulting PNGs + `.ipynb` as the `evaluation-notebook` artefact for
   visual diff review.
7. Uploads `backtest_diagnostics.txt` as the `backtest-diagnostics`
   artefact.

The network smoke test (`test_edgar_rate_limit_smoke`) is **not** run by
CI — opt in locally with `pytest -m network -v` after exporting
`VESPER_SEC_USER_AGENT`.

## Static HTML export + GitHub Pages

`docs/evaluation.html` is a static export of the notebook with all four
charts embedded as base64 PNGs. You can browse it:

* directly from the repo: open `docs/evaluation.html` on `github.com`
  (renders as text in the web UI; download for offline viewing), or
* live via **GitHub Pages** if Pages is enabled on the repository with
  source = "GitHub Actions" (Settings → Pages → Source). The Pages site
  URL will be of the form `https://<owner>.github.io/Vesper/evaluation.html`.

The Pages workflow at `.github/workflows/pages.yml` re-builds and
re-deploys the HTML on every push to `main`, so the live site always
reflects the latest committed notebook.

## License

MIT. See file headers.
