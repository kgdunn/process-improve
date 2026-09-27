"""Run compare_cv_criteria on every PLS model that the pid-book fits.

The book's chapters are executed exactly as its own checker runs them
(``tools/check_code_blocks.py`` in a pid-book checkout: one namespace per chapter, files
in toctree order). Every ``PLS.fit`` reached from book code is recorded, including the fit
inside a wrapper such as ``BatchPLS.fit``, but not the refits inside a library routine
such as ``select_n_components``. Each distinct data set then goes through
``compare_cv_criteria`` with its missing values kept, and the recommendations are printed
as a markdown table: the book-validation table of process-improve #605. A data set that
several blocks fit (a bootstrap loop, or a later block refitting the same data) is listed
once, under the first block that fits it.

Usage
-----
    python scripts/validate_book_pls_examples.py PATH_TO_PID_BOOK [CHAPTER ...]

The chapters default to the two with PLS models, ``latent-variable-modelling`` and
``product-development-product-improvement``. The data sets are read from openmv.net, so
the run needs network access; it takes about 20 minutes.
"""

from __future__ import annotations

import hashlib
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from process_improve.multivariate import PLS, compare_cv_criteria

CHAPTERS = ("latent-variable-modelling", "product-development-product-improvement")
RULES = ["q2_max", "q2_1se", "van_der_voet", "score_correlation", "covariance_permutation", "subspace_stability"]
BOOK_MODULE = "__pid_book__"  # the __name__ the checker gives the namespace of book code


def _called_from_book() -> bool:
    """Report whether PLS.fit was reached from book code, directly or through other ``fit`` methods."""
    frame = sys._getframe(2)  # the caller of the recording wrapper
    while frame is not None and frame.f_code.co_name == "fit":
        frame = frame.f_back
    return frame is not None and frame.f_globals.get("__name__") == BOOK_MODULE


def record_fits(checker: object, chapters: tuple[str, ...]) -> dict[tuple, dict]:
    """Run the chapters and return one record per distinct data set that book code fitted."""
    distinct: dict[tuple, dict] = {}
    current_block = ["?"]
    original_fit, original_run_block = PLS.fit, checker.run_block

    def recording_fit(self: PLS, X: object, Y: object = None, *args: object, **kwargs: object) -> PLS:
        if _called_from_book():
            X_df = pd.DataFrame(X)
            Y_df = Y.to_frame() if isinstance(Y, pd.Series) else pd.DataFrame(Y)
            key = (current_block[0], tuple(map(str, X_df.columns)), tuple(map(str, Y_df.columns)), len(X_df))
            entry = distinct.setdefault(
                key, {"block": current_block[0], "X": X_df.copy(), "Y": Y_df.copy(), "A": set()}
            )
            entry["A"].add(self.n_components)
        return original_fit(self, X, Y, *args, **kwargs)

    def tracking_run_block(block: object, namespace: dict, **kwargs: object) -> object:
        current_block[0] = block.label
        return original_run_block(block, namespace, **kwargs)

    PLS.fit, checker.run_block = recording_fit, tracking_run_block
    try:
        checker.configure_environment()
        for unit in checker.build_units():
            if unit.name in chapters:
                outcomes = checker.run_unit(unit)
                failed = [o.block.label for o in outcomes if o.status == "failed"]
                print(f"{unit.name}: {len(outcomes)} blocks, {len(failed)} failed {failed}", flush=True)
    finally:
        PLS.fit, checker.run_block = original_fit, original_run_block
    return distinct


def criteria_row(entry: dict) -> str:
    """One markdown row: the block, the data shape, the book's A, each rule's pick, the run time."""
    X, Y = entry["X"], entry["Y"]
    book_a = sorted(a for a in entry["A"] if a is not None)
    gaps = int((X.isna().any(axis=1) | Y.isna().any(axis=1)).sum())
    shape = f"{X.shape[0]} x {X.shape[1]} x {Y.shape[1]}" + (f" ({gaps} rows with gaps)" if gaps else "")
    book = f"{book_a[0]}-{book_a[-1]}" if len(book_a) > 1 else str(book_a[0] if book_a else "")
    start = time.time()
    try:
        result = compare_cv_criteria(
            X, Y, max_components=max([*book_a, 2]) + 3, cv=7, random_state=0, n_permutations=499, n_cv_permutations=199
        )
        picks = " | ".join(str(result.recommendations.loc[rule, "n_components"]) for rule in RULES)
    except Exception as exc:  # noqa: BLE001 - report every failure in the table rather than stop the run
        picks = f"{type(exc).__name__}: {exc}" + " |" * (len(RULES) - 1)
    return f"| `{entry['block'].split('/')[-1]}` | {shape} | {book} | {picks} | {time.time() - start:.1f} s |"


def main(argv: list[str]) -> None:
    """Record the book's PLS fits and print the table of recommendations."""
    if not argv:
        raise SystemExit(__doc__)
    book = Path(argv[0]).resolve()
    sys.path.insert(0, str(book / "tools"))
    import check_code_blocks  # noqa: PLC0415 - importable only once the book's path is known

    chapters = tuple(argv[1:]) or CHAPTERS
    distinct = record_fits(check_code_blocks, chapters)
    numeric: dict[str, dict] = {}
    for entry in distinct.values():
        if not all(np.issubdtype(t, np.number) for t in [*entry["X"].dtypes, *entry["Y"].dtypes]):
            continue
        values = pd.concat([entry["X"], entry["Y"]], axis=1).to_numpy(dtype=float)
        fingerprint = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        numeric.setdefault(fingerprint, entry)["A"] |= entry["A"]
    print(f"{len(numeric)} distinct numeric data sets\n", flush=True)
    print("| Book section (block) | N x K x M | Book's A | " + " | ".join(f"`{r}`" for r in RULES) + " | time |")
    print("|---|---|---|" + "---|" * len(RULES) + "---|", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # NIPALS convergence notes on noise components
        for entry in numeric.values():
            print(criteria_row(entry), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
