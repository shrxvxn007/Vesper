"""Regression tests pinning the notebook REPO_ROOT discovery block.

The notebook cell that resolves ``REPO_ROOT`` historically probed for a
top-level ``main.py`` (``(NOTEBOOK_DIR / "main.py").exists()``). After the
package layout was re-organised under ``vesper/`` (see the repo commit
history), that probe silently pointed at nothing — both notebooks
silently fell through to ``REPO_ROOT = NOTEBOOK_DIR.parent`` (or
worse, kept a partially-resolved path) and ``from main import
run_backtest`` then raised ``ModuleNotFoundError`` during CI's
``jupyter nbconvert --execute`` step.

That failure was fixed in commit ``042ac20``: the probe was switched to
``(NOTEBOOK_DIR / "vesper").is_dir()`` and the import lines were
rewired to ``from vesper.main import run_backtest`` /
``from vesper.data_pipeline.graph_builder import …``.

These tests are the regression guard so a future refactor that
inadvertently reverts the discovery block (or imports) immediately
breaks CI with a precise assertion message — instead of re-introducing
the original failure mode via a "looks fine in my editor" landmine.

Each test asserts (on the discovery cell of one notebook):

* the ``vesper`` path sentinel ``NOTEBOOK_DIR / "vesper"`` is present
* the legacy ``main.py`` path sentinel ``NOTEBOOK_DIR / "main.py"``
  is gone

The discovery cell is identified by the signature line
``NOTEBOOK_DIR = Path.cwd().resolve()`` (a code-cell-only search), so
the assertions are robust to cell reordering and to whether the
notebook is opened from the repo root or from a sub-directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

# Sentinel substrings frozen from the post-fix state of both notebooks.
# Their presence / absence is exactly the class of failure we want
# CI to catch, so they are intentionally a direct mirror of the cell
# text (not a regex, not a parsed-AST probe) — the failure message on
# a regression will name the literal substring that went missing.
_VESPER_PATH_SENTINEL: Final[str] = 'NOTEBOOK_DIR / "vesper"'
_LEGACY_MAIN_PY_SENTINEL: Final[str] = 'NOTEBOOK_DIR / "main.py"'

# Each notebook is parameterised by its path under ``<repo>/notebooks/``.
_NOTEBOOK_FILES: Final[tuple[Path, ...]] = (
    Path(__file__).resolve().parent.parent / "notebooks" / "evaluation.ipynb",
    Path(__file__).resolve().parent.parent
    / "notebooks"
    / "cap_binding_diagnostics.ipynb",
)


def _discovery_cell_source(notebook_path: Path) -> str:
    """Return the source of the code cell that resolves ``REPO_ROOT``.

    The cell is identified by the signature line that names ``NOTEBOOK_DIR``
    as ``Path.cwd().resolve()`` plus a ``.is_dir()`` / ``.exists()`` probe
    on a candidate package directory — that combination is unique to the
    REPO_ROOT discovery block across the notebook, so a markdown cell or
    an unrelated ``NOTEBOOK_DIR`` reference will not collide.

    Args:
        notebook_path: Absolute path to a ``.ipynb`` JSON file.

    Returns:
        The cell source flattened to a single string (the cell's
        ``source`` field may be a list of lines or a single string
        depending on how the notebook was authored).

    Raises:
        AssertionError: If the notebook is unreadable, if no code
            cell with the discovery signature exists, or if more
            than one cell matches (which would mean the tests can
            no longer pin the discovery block uniquely).
    """
    import json  # local import keeps module-load failure mode clear

    nb_text = notebook_path.read_text(encoding="utf-8")
    nb = json.loads(nb_text)

    matches: list[str] = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, list):
            cell_text = "".join(src)
        else:
            cell_text = src
        # Signature anchor: the discovery block always sets
        # NOTEBOOK_DIR = Path.cwd().resolve() (verbatim).
        if "NOTEBOOK_DIR = Path.cwd().resolve()" in cell_text:
            matches.append(cell_text)

    assert matches, (
        f"No REPO_ROOT discovery cell (containing the signature line "
        f"NOTEBOOK_DIR = Path.cwd().resolve()) found in "
        f"{notebook_path}. Was the discovery block removed or rewritten?"
    )
    assert len(matches) == 1, (
        f"Expected exactly one REPO_ROOT discovery cell in "
        f"{notebook_path}, found {len(matches)}. The test cannot "
        f"uniquely pin the block until the collision is resolved."
    )
    return matches[0]


@pytest.mark.parametrize(
    "notebook_path",
    _NOTEBOOK_FILES,
    ids=lambda p: p.name,
)
def test_notebook_repo_root_block_points_at_vesper_package(
    notebook_path: Path,
) -> None:
    """The REPO_ROOT discovery block must use the ``vesper/`` sentinel.

    This is the positive half of the guard: the cell must reference
    the package directory using ``NOTEBOOK_DIR / "vesper"``. Without
    this assertion, a future refactor could break the discovery block
    in ways that no other test catches (pytest imports use Python
    package semantics, not the notebook's own probe).
    """
    assert notebook_path.exists(), (
        f"Notebook {notebook_path} is missing — cannot verify the "
        f"REPO_ROOT discovery block."
    )
    cell_text = _discovery_cell_source(notebook_path)

    assert _VESPER_PATH_SENTINEL in cell_text, (
        f"REPO_ROOT discovery block in {notebook_path.name} no longer "
        f"probes via {_VESPER_PATH_SENTINEL!r}. The probe has drifted "
        f"away from the vesper/ package sentinel, which is exactly the "
        f"failure mode that previously broke CI. Either the discovery "
        f"block needs to be updated to use the vesper/ package "
        f"directory, or the package layout has moved and this "
        f"sentinel must move with it."
    )


@pytest.mark.parametrize(
    "notebook_path",
    _NOTEBOOK_FILES,
    ids=lambda p: p.name,
)
def test_notebook_repo_root_block_drops_legacy_main_py_sentinel(
    notebook_path: Path,
) -> None:
    """The REPO_ROOT discovery block must NOT reference ``main.py``.

    This is the negative half of the guard: a literal reference to
    ``NOTEBOOK_DIR / "main.py"`` in the discovery cell is the exact
    broken-state marker from the pre-refactor layout (see commit
    042ac20 for the original fix). Its re-appearance would mean a refactor has
    rolled the package layout back to the top-level ``main.py`` form
    without fixing all of the downstream consumers — the same bug
    class as the one this entire branch originally fixed.
    """
    assert notebook_path.exists(), (
        f"Notebook {notebook_path} is missing — cannot verify the "
        f"REPO_ROOT discovery block."
    )
    cell_text = _discovery_cell_source(notebook_path)

    assert _LEGACY_MAIN_PY_SENTINEL not in cell_text, (
        f"REPO_ROOT discovery block in {notebook_path.name} still "
        f"contains the legacy sentinel {_LEGACY_MAIN_PY_SENTINEL!r}. "
        f"This is the exact broken-state marker that previously made "
        f"``python main.py --data-dir data`` (and the REPO_ROOT probe) "
        f"fail. The package layout has moved to vesper/ — please update "
        f"the discovery block to match."
    )
