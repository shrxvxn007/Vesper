"""SEC EDGAR filing scraper.

This module provides a thin client for fetching SEC EDGAR filings (10-Q and 10-K)
with a SEC-compliant User-Agent header. The client is intentionally defensive:
SEC's EDGAR fair-access rules require a descriptive User-Agent string and rate-
limited access (max ~10 requests/sec). Use :py:meth:`SECScraper.fetch_recent_filings`
only when explicitly authorised to call EDGAR.

For offline research, all callers in this repository use the synthetic data
generator in :mod:`scripts.synthetic_generator`, so live scraping is opt-in.

Typical usage::

    scraper = SECScraper(user_agent="Vesper Research research@example.com")
    filings = scraper.fetch_recent_filings(cik="0000320193", form_types=("10-Q",))
    text = scraper.fetch_filing_text(filings.iloc[0]["url"])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
EDGAR_ARCHIVE_URL_TEMPLATE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{primary_document}"
DEFAULT_REQUEST_TIMEOUT = 30  # seconds
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 1.5
EDGAR_MIN_INTERVAL_SECONDS = 0.15  # ~6-7 req/sec keeps us well under the 10/sec cap


class SECScraperError(RuntimeError):
    """Raised when an SEC fetch fails irrecoverably after retries."""


@dataclass(slots=True)
class SECScraperConfig:
    """Configuration for the SEC scraper.

    Attributes:
        user_agent: SEC-compliant User-Agent string. SEC requires an
            identifiable email address.
        timeout: HTTP timeout (seconds) for each request.
        max_retries: Number of retry attempts on transient failures.
        backoff_seconds: Initial backoff between retries (exponential).
        min_interval_seconds: Minimum seconds between consecutive requests
            to respect SEC rate limits.
    """

    user_agent: str = "Vesper Research research@example.com"
    timeout: int = DEFAULT_REQUEST_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS
    min_interval_seconds: float = EDGAR_MIN_INTERVAL_SECONDS


@dataclass(slots=True)
class FilingRecord:
    """A single SEC filing record.

    Attributes:
        cik: SEC Central Index Key (zero-padded string, 10 digits).
        ticker: Lower-case ticker symbol as reported by SEC.
        form_type: Filing form code (e.g., ``10-Q`` or ``10-K``).
        filing_date: Date the filing was publicly released.
        period_end_date: Financial period covered by the filing.
        accession_number: SEC accession number (with dashes).
        primary_document: Filename of the primary document inside the archive.
        url: Direct URL to the primary document HTML/text.
    """

    cik: str
    ticker: str
    form_type: str
    filing_date: pd.Timestamp
    period_end_date: pd.Timestamp
    accession_number: str
    primary_document: str
    url: str

    def to_dict(self) -> dict[str, object]:
        """Return a plain dict suitable for ``pd.DataFrame`` construction."""
        return {
            "cik": self.cik,
            "ticker": self.ticker,
            "form_type": self.form_type,
            "filing_date": self.filing_date,
            "period_end_date": self.period_end_date,
            "accession_number": self.accession_number,
            "primary_document": self.primary_document,
            "url": self.url,
        }


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


class SECScraper:
    """SEC EDGAR filing scraper with deterministic rate limiting.

    Args:
        user_agent: SEC-compliant user agent. Must include an email.
        session: Optional pre-existing :class:`requests.Session`.
        config: Optional :class:`SECScraperConfig` override.
        allow_online: If ``False`` (the default outside the synthetic data
            path), HTTP calls will raise :class:`SECScraperError`. We default
            to offline to make the framework safe to import in any environment.
    """

    def __init__(
        self,
        user_agent: str = "Vesper Research research@example.com",
        *,
        session: requests.Session | None = None,
        config: SECScraperConfig | None = None,
        allow_online: bool = False,
    ) -> None:
        if "@" not in user_agent:
            # SEC explicitly requires: "Sample Company Name AdminContact@<sample company domain>"
            raise ValueError(
                "SEC requires a User-Agent containing an email address; "
                "got user_agent=" + repr(user_agent),
            )

        self._config = config or SECScraperConfig(user_agent=user_agent)
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "User-Agent": self._config.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Host": "www.sec.gov",
            }
        )
        self._last_request_at: float = 0.0
        self.allow_online: bool = allow_online

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_recent_filings(
        self,
        cik: str | int,
        *,
        form_types: Iterable[str] = ("10-Q", "10-K"),
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Fetch a metadata table of recent filings for a given CIK.

        Args:
            cik: SEC CIK for the issuer. Accepts integer or 10-digit string.
            form_types: Filing types to keep (e.g., ``{"10-Q", "10-K"}``).
            limit: Optional maximum number of rows to return (most-recent first).

        Returns:
            :class:`pandas.DataFrame` with one row per filing. Columns:
            ``cik``, ``ticker``, ``form_type``, ``filing_date``,
            ``period_end_date``, ``accession_number``, ``primary_document``,
            ``url``. Indexed by ``filing_date`` (ascending).
        """
        cik_padded = self._pad_cik(cik)
        submissions = self._get_json(EDGAR_SUBMISSIONS_URL.format(cik_padded=cik_padded))
        records: list[FilingRecord] = []
        form_set = set(form_types)

        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        period_ends = recent.get("reportDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        ticker = (submissions.get("tickers") or ["UNKNOWN"])[0]

        for form, fdate, pend, accession, primary in zip(
            forms, dates, period_ends, accessions, primary_docs
        ):
            if form not in form_set:
                continue
            accession_clean = accession.replace("-", "")
            url = EDGAR_ARCHIVE_URL_TEMPLATE.format(
                cik_int=int(cik_padded),
                access=accession,
                primary_document=primary,
            )
            records.append(
                FilingRecord(
                    cik=cik_padded,
                    ticker=ticker,
                    form_type=form,
                    filing_date=pd.Timestamp(fdate),
                    period_end_date=pd.Timestamp(pend),
                    accession_number=accession,
                    primary_document=primary,
                    url=url,
                )
            )
            if limit is not None and len(records) >= limit:
                break

        df = pd.DataFrame([r.to_dict() for r in records])
        if df.empty:
            return df
        return df.set_index("filing_date").sort_index()

    def fetch_filing_text(self, url: str) -> str:
        """Fetch the raw HTML/text body of a filing document.

        Args:
            url: Direct URL to the filing document (``.htm`` or ``.txt``).

        Returns:
            Raw HTML/text body as a string.
        """
        return self._get_text(url)

    def iter_filings(
        self,
        ciks: Iterable[str | int],
        *,
        form_types: Iterable[str] = ("10-Q", "10-K"),
    ) -> Iterator[FilingRecord]:
        """Yield :class:`FilingRecord`s for a sequence of CIKs.

        Order is (cik-major, date-minor), which keeps each issuer's filings
        in chronological order — a useful invariant for downstream NLP.
        """
        for cik in ciks:
            df = self.fetch_recent_filings(cik, form_types=form_types)
            df_reset = df.reset_index()
            for row in df_reset.itertuples():
                yield FilingRecord(
                    cik=row.cik,
                    ticker=row.ticker,
                    form_type=row.form_type,
                    filing_date=row.filing_date,
                    period_end_date=row.period_end_date,
                    accession_number=row.accession_number,
                    primary_document=row.primary_document,
                    url=row.url,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pad_cik(self, cik: str | int) -> str:
        return f"{int(cik):010d}"

    def _throttle(self) -> None:
        """Sleep if necessary to enforce min_interval between requests."""
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._config.min_interval_seconds:
            time.sleep(self._config.min_interval_seconds - elapsed)

    def _request(self, method: str, url: str) -> requests.Response:
        if not self.allow_online:
            raise SECScraperError(
                f"Online SEC access is disabled (allow_online=False). "
                f"Refusing to {method.upper()} {url}. Enable allow_online=True "
                "only when running an authorised research pull."
            )
        attempt = 0
        last_exc: Exception | None = None
        while attempt <= self._config.max_retries:
            self._throttle()
            try:
                response = self._session.request(
                    method, url, timeout=self._config.timeout
                )
            except requests.RequestException as exc:  # noqa: PERF203
                last_exc = exc
                attempt += 1
                time.sleep(self._config.backoff_seconds ** attempt)
                continue
            self._last_request_at = time.monotonic()
            if response.status_code == 200:
                return response
            if response.status_code in (403, 429):
                # SEC rate-limit response; back off harder.
                time.sleep(self._config.backoff_seconds ** (attempt + 2))
                attempt += 1
                continue
            if 500 <= response.status_code < 600:
                attempt += 1
                time.sleep(self._config.backoff_seconds ** attempt)
                continue
            response.raise_for_status()
        raise SECScraperError(
            f"Exhausted {self._config.max_retries} retries for {url}; last error: {last_exc}"
        )

    def _get_json(self, url: str) -> dict[str, object]:
        response = self._request("GET", url)
        return response.json()

    def _get_text(self, url: str) -> str:
        response = self._request("GET", url)
        return response.text


__all__ = [
    "EDGAR_SUBMISSIONS_URL",
    "EDGAR_ARCHIVE_URL_TEMPLATE",
    "EDGAR_MIN_INTERVAL_SECONDS",
    "SECScraper",
    "SECScraperConfig",
    "SECScraperError",
    "FilingRecord",
]
