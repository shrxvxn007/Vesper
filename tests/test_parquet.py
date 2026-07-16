"""Parquet I/O smoke test.

The synthetic generator (``scripts/synthetic_generator.py``) and both
notebooks (``notebooks/evaluation.ipynb``,
``notebooks/cap_binding_diagnostics.ipynb``) read and write parquet
files. If ``pyarrow`` or ``fastparquet`` is missing, pandas fails with
the generic ``ImportError: Unable to find a usable engine; tried using:
'pyarrow', 'fastparquet'`` — which is exactly what bit CI on the first
``pytest tests -v`` run after Pages was added.

These two tests are deliberately unit-level and tiny so the failure
mode surfaces immediately with a clear message instead of being buried
in a stack trace from the integration / notebook level.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest


def test_parquet_roundtrip_via_in_memory_buffer() -> None:
    """Write a one-row DataFrame to an in-memory parquet buffer and read it back.

    Asserts frame equality on the round trip. If pyarrow / fastparquet
    is missing, ``to_parquet`` raises ``ImportError`` from pandas'
    engine dispatch and the test fails fast with a clear stack trace.
    """
    df = pd.DataFrame(
        {
            "col_int": [1],
            "col_str": ["x"],
            "col_float": [1.5],
            "col_bool": [True],
        }
    )
    buf = io.BytesIO()
    df.to_parquet(buf)
    buf.seek(0)
    roundtripped = pd.read_parquet(buf)

    # Equality on values + dtypes + index; the parquet round trip preserves
    # all three, so this is the most thorough check we can do at this level.
    pd.testing.assert_frame_equal(df, roundtripped)


def test_parquet_engine_is_installed() -> None:
    """Explicitly verify that at least one of pyarrow / fastparquet is importable.

    Gives a much clearer error than pandas' generic ``ImportError`` if
    both are missing, by naming the two packages directly in the
    failure message.
    """
    try:
        import pyarrow  # noqa: F401
        engine = "pyarrow"
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            engine = "fastparquet"
        except ImportError:
            pytest.fail(
                "Neither pyarrow nor fastparquet is installed; pandas cannot "
                "read or write parquet files. Add one of them to "
                "requirements.txt (pyarrow is preferred)."
            )
    assert engine in {"pyarrow", "fastparquet"}
