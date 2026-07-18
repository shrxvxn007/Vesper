"""Regression test for the latent dedent bug in ``SECScraper.iter_filings``.

Before the fix, the inner ``df_reset = df.reset_index()`` and the
``for row in df_reset.itertuples():`` body were dedented out of the
``for cik in ciks:`` loop, so the generator drained the outer loop during
construction (assigning ``df`` repeatedly), then yielded only the
**last** CIK's filings once the generator was iterated. The regression
tests below assert that records span **all** input CIKs and stay
chronologically ordered within each cik.
"""

from __future__ import annotations

import pandas as pd

from vesper.data_pipeline.sec_scraper import SECScraper


def _fixture_for_cik(cik: str, ticker: str, n: int) -> pd.DataFrame:
    """Build a filings DataFrame in the shape returned by ``fetch_recent_filings``.

    The DataFrame has ``filing_date`` as the index (sorted ascending), the
    other columns mirror what ``FilingRecord.to_dict`` produces.
    """
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        rows.append(
            {
                "cik": cik,
                "ticker": ticker,
                "form_type": "10-Q" if i % 2 == 0 else "10-K",
                "filing_date": base + pd.Timedelta(days=90 * i),
                "period_end_date": base + pd.Timedelta(days=89 * i),
                "accession_number": f"{cik}-{i:04d}",
                "primary_document": f"doc{i}.htm",
                "url": f"https://example.com/edgar/{cik}/{i}",
            }
        )
    return pd.DataFrame(rows).set_index("filing_date").sort_index()


class _StubSECScraper(SECScraper):
    """``SECScraper`` stub: ``fetch_recent_filings`` returns canned DataFrames.

    Bypasses ``SECScraper.__init__`` so the user-agent validation
    (which requires ``"@"``) does not get in the way of a focused unit test.
    The real network path is never exercised.
    """

    def __init__(self, fixtures_by_cik: dict[str, pd.DataFrame]) -> None:
        self._fixtures = fixtures_by_cik

    def fetch_recent_filings(
        self,
        cik: str | int,
        *,
        form_types: tuple[str, ...] = ("10-Q", "10-K"),
    ) -> pd.DataFrame:
        cik_padded = f"{int(cik):010d}"
        df = self._fixtures.get(cik_padded)
        if df is None:
            return pd.DataFrame()
        return df.copy()


def test_iter_filings_yields_records_for_every_input_cik() -> None:
    """Reproduces the dedent bug if it ever regresses.

    Pre-fix, only the **last** CIK's filings were yielded (3 records
    spanning a single cik). Post-fix, all 9 records across all 3 CIKs
    are yielded.
    """
    fixtures = {
        "0000000001": _fixture_for_cik("0000000001", "AAA", n=3),
        "0000000002": _fixture_for_cik("0000000002", "BBB", n=3),
        "0000000003": _fixture_for_cik("0000000003", "CCC", n=3),
    }
    scraper = _StubSECScraper(fixtures)

    records = list(scraper.iter_filings(ciks=["1", "2", "3"]))

    yielded_ciks = {r.cik for r in records}
    assert yielded_ciks == {"0000000001", "0000000002", "0000000003"}, (
        f"iter_filings yielded {len(records)} records spanning only "
        f"{yielded_ciks}; expected all three CIKs - regression of the "
        f"dedent-instead-of-indent bug in iter_filings."
    )
    assert len(records) == 9, (
        f"expected 3 filings * 3 tickers = 9 records, got {len(records)}"
    )


def test_iter_filings_yields_in_chrono_order_per_cik() -> None:
    """cik-major order; within each cik, filings stay in chronological order."""
    fixtures = {
        "0000000001": _fixture_for_cik("0000000001", "AAA", n=3),
        "0000000002": _fixture_for_cik("0000000002", "BBB", n=3),
    }
    scraper = _StubSECScraper(fixtures)

    records = list(scraper.iter_filings(ciks=["1", "2"]))

    dates_by_cik: dict[str, list[pd.Timestamp]] = {}
    for r in records:
        dates_by_cik.setdefault(r.cik, []).append(r.filing_date)
    for cik, dates in dates_by_cik.items():
        assert dates == sorted(dates), (
            f"cik {cik} filings are not in chronological order: {dates}"
        )


def test_iter_filings_preserves_caller_supplied_cik_order() -> None:
    """Caller-supplied CIK order is preserved: cik-major in iteration.

    Pre-fix, only the last CIK's filings came out (so this would fail
    trivially too), but post-fix we additionally assert position-level
    ordering so any future re-order regression surfaces.
    """
    fixtures = {
        "0000000001": _fixture_for_cik("0000000001", "AAA", n=2),
        "0000000002": _fixture_for_cik("0000000002", "BBB", n=2),
        "0000000003": _fixture_for_cik("0000000003", "CCC", n=2),
    }
    scraper = _StubSECScraper(fixtures)

    records = list(scraper.iter_filings(ciks=["3", "1", "2"]))

    seen_order: list[str] = [r.cik for r in records]
    expected_first = ["0000000003", "0000000003", "0000000001", "0000000001", "0000000002", "0000000002"]
    assert seen_order == expected_first, (
        f"expected CIK-major ordering {expected_first}, got {seen_order}"
    )
