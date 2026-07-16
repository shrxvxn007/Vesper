"""Information-Decay feature via TF-IDF cosine similarity.

Implements the "Information Decay Factor": for each ticker on each public
release date, compute the cosine similarity between the cleaned MD&A text of
the latest filing and its immediate predecessor for the same ticker. We then
convert that into a *risk score* equal to ``1 - cos_sim`` (clipped to ``[0, 1]``)
so a *drop* in similarity maps to an *increased* risk score.

The vectorisation uses :class:`sklearn.feature_extraction.text.TfidfVectorizer`,
which L2-normalises its rows so the inner product of two rows is exactly the
cosine similarity. Empty or newly-listed tickers are imputed to ``0.0`` decay
(no shock), preventing spurious short signals during warm-up.

All output is keyed by ``(release_date, ticker)`` to make downstream joins
strictly chronological — the merge from filings to features uses public
release dates, **never** financial period end dates.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Cap the decay score into [0, 1] for downstream normalisation.
RISK_FLOOR: Final[float] = 0.0
RISK_CEIL: Final[float] = 1.0

DEFAULT_TFIDF_PARAMS: Final[dict[str, object]] = {
    "ngram_range": (1, 2),
    "min_df": 1,
    # ``max_df`` defaults to 1.0 (no upper-frequency pruning) because our
    # pairwise-compare path fits a fresh ``TfidfVectorizer`` on only two
    # documents at a time — a fractional document count like 0.95 would
    # silently drop every token that appears in both, leaving the vectoriser
    # with an empty vocabulary and raising ``ValueError: After pruning, no
    # terms remain``.
    "max_df": 1.0,
    "sublinear_tf": True,
    "strip_accents": "unicode",
    "lowercase": True,
    "norm": "l2",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_information_decay(
    filings_df: pd.DataFrame,
    *,
    text_column: str = "text_clean",
    ticker_column: str = "ticker",
    release_column: str = "release_date",
    tfidf_params: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Compute per-filing information-decay risk score.

    Args:
        filings_df: A DataFrame with at least the columns
            ``release_column`` (datetime-like), ``ticker_column``, and
            ``text_column``.
        text_column: Column with clean text (the result of
            :func:`data_pipeline.mda_parser.clean_text`).
        ticker_column: Column identifying the issuer.
        release_column: Column with the *public release* timestamp. **Do not**
            pass ``period_end_date`` here.
        tfidf_params: Optional overrides for
            :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.

    Returns:
        :class:`pandas.DataFrame` indexed by ``(release_date, ticker)`` with
        a single ``nlp_decay_score`` column in ``[0, 1]``. The first filing
        per ticker returns ``0.0`` (no prior to compare against).
    """
    if filings_df.empty:
        # Build an empty frame whose index is a properly-labelled 2-level
        # MultiIndex. ``rename_axis`` would fail on a default RangeIndex
        # because pandas requires the number of names to match the number
        # of levels.
        empty_index = pd.MultiIndex.from_arrays(
            [[], []], names=[release_column, ticker_column]
        )
        return pd.DataFrame(
            {"nlp_decay_score": pd.Series([], dtype=float, index=empty_index)}
        )

    needed = {text_column, ticker_column, release_column}
    missing = needed - set(filings_df.columns)
    if missing:
        raise KeyError(f"filings_df is missing required columns: {sorted(missing)}")

    sorted_df = filings_df.sort_values([ticker_column, release_column]).reset_index(drop=True)
    rows: list[tuple[pd.Timestamp, str, float]] = []

    for ticker, group in sorted_df.groupby(ticker_column, sort=False):
        cleaned_texts = group[text_column].tolist()
        # First filing for this ticker: define decay = 0 (no prior).
        rows.append((group[release_column].iloc[0], ticker, 0.0))
        if len(cleaned_texts) < 2:
            continue
        params = {**DEFAULT_TFIDF_PARAMS, **(tfidf_params or {})}
        vectorizer = TfidfVectorizer(**params)
        # ``fit_transform`` on the two latest consecutive texts to keep the
        # vocabulary aligned to those documents — earlier documents in the
        # group would otherwise leak into the vectoriser state.
        for i in range(1, len(cleaned_texts)):
            pair = [cleaned_texts[i - 1], cleaned_texts[i]]
            # ``fit_transform`` only on this pair so vocab does not drift.
            matrix = vectorizer.fit_transform(pair)
            # matrix has shape (2, vocab). Both rows are L2-normalised, so
            # ``matrix[0] @ matrix[1].T`` is the cosine similarity.
            sim = (matrix[0] @ matrix[1].T).toarray().item()
            sim = float(np.clip(sim, 0.0, 1.0))
            decay = float(np.clip(1.0 - sim, RISK_FLOOR, RISK_CEIL))
            rows.append((group[release_column].iloc[i], ticker, decay))

    out = pd.DataFrame(rows, columns=[release_column, ticker_column, "nlp_decay_score"])
    out = out.set_index([release_column, ticker_column]).sort_index()
    return out


class NLPDecayCalculator:
    """Pre-fitted TF-IDF vectoriser wrapper for amortised batch scoring.

    Useful when the corpus is large and you want to score many new texts in
    one go while reusing the lexicon learned upstream. Most callers in this
    codebase prefer the stateless :func:`compute_information_decay` helper.
    """

    def __init__(self, **tfidf_overrides: object) -> None:
        params = {**DEFAULT_TFIDF_PARAMS, **tfidf_overrides}
        self._vectorizer = TfidfVectorizer(**params)

    @property
    def vectorizer(self) -> TfidfVectorizer:
        return self._vectorizer

    def fit(self, texts: list[str]) -> "NLPDecayCalculator":
        self._vectorizer.fit(texts)
        return self

    def score_decay(self, prev_text: str, curr_text: str) -> float:
        matrix = self._vectorizer.transform([prev_text, curr_text])
        sim = (matrix[0] @ matrix[1].T).toarray().item()
        sim = float(np.clip(sim, 0.0, 1.0))
        return float(np.clip(1.0 - sim, RISK_FLOOR, RISK_CEIL))


__all__ = ["compute_information_decay", "NLPDecayCalculator", "RISK_FLOOR", "RISK_CEIL", "DEFAULT_TFIDF_PARAMS"]
