"""Every warning a user's call can raise in multivariate/ and batch/ names that call's line (#1079, #1092).

Each one passes ``stacklevel=user_stacklevel()``, which walks out of process_improve, scikit-learn
and mutmut. A fixed count named a line in the library as soon as a frame came between the call
and the warning: a private helper, scikit-learn's ``_fit_context`` around ``fit``, a subclass's
``fit`` or another model's inner fit, or the trampoline through which mutmut calls each function
it mutates. Every case below calls the library from this file, so the warning's file is this one.
"""

from __future__ import annotations

import itertools
import re
import types
import warnings
from collections.abc import Callable
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from process_improve.batch import BatchScaler, DTWAligner, determine_scaling, smooth_trajectories
from process_improve.batch.preprocessing import batch_dtw
from process_improve.multivariate._projection import operator_for_pattern
from process_improve.multivariate.methods import (
    ASCA,
    CA,
    FAMD,
    FPCA,
    GPA,
    MBPCA,
    MBPLS,
    MCA,
    MFA,
    OPLS,
    PCA,
    PLS,
    PLSDA,
    PRM,
    TPLS,
    AdaptivePLS,
    DataFrameDict,
    NotEnoughVarianceError,
    count_discoveries_under_null,
    hotellings_t2_limit,
    randomization_test_mbpls,
)

_RNG = np.random.default_rng(0)
#: Thirty rows of four columns and one response that depends on three of them.
X = pd.DataFrame(_RNG.normal(size=(30, 4)), columns=list("abcd"))
Y = pd.DataFrame({"y": X.to_numpy() @ [1.0, -0.5, 0.3, 0.0] + _RNG.normal(0.0, 0.3, 30)})
#: Forty rows of six columns on three latent directions, whose Q2 rises over the first two components.
RANK_THREE = pd.DataFrame(
    (_RNG.normal(size=(40, 3)) * [6.0, 4.0, 2.5]) @ _RNG.normal(size=(3, 6)) + _RNG.normal(size=(40, 6))
)
#: The same columns autoscaled, which the cross-validation helpers say they will scale again.
STANDARDISED = (X - X.mean()) / X.std()
#: Two classes, which the first column separates.
LABELS = pd.Series(np.where(X["a"] > 0, "high", "low"))
#: A replicated 2 x 2 design whose response has an interaction that no column of X reaches, so
#: the second component finds nothing left in Y that X can explain.
_A, _B = np.tile([-1.0, 1.0, -1.0, 1.0], 2), np.tile([-1.0, -1.0, 1.0, 1.0], 2)
DESIGN = pd.DataFrame({"a": _A, "b": _B})
INTERACTION = pd.DataFrame({"y": _A + 2.0 * _B + 0.5 * _A * _B})


def _with_an_empty_column(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(empty=np.nan)


def _curves() -> pd.DataFrame:
    """Five curves on a grid of 51 points, each a mix of two smooth shapes."""
    grid = np.linspace(0.0, 1.0, 51)
    return pd.DataFrame(_RNG.normal(size=(5, 2)) @ np.vstack([np.sin(np.pi * grid), grid**2]), columns=grid)


def _t_shaped(*, constant_f_column: bool = False, empty_f_column: bool = False) -> tuple[dict, DataFrameDict]:
    """Five materials of three properties, blended ten ways; Y follows the blends' properties."""
    rng = np.random.default_rng(0)
    blends = [f"b{i}" for i in range(10)]
    d_matrix = {"A": pd.DataFrame(rng.normal(size=(5, 3)), index=[f"m{i}" for i in range(5)], columns=list("pqr"))}
    f = pd.DataFrame(rng.random((10, 5)), index=blends, columns=d_matrix["A"].index)
    y = f.to_numpy() @ d_matrix["A"].to_numpy() @ rng.normal(size=(3, 2)) + 0.3 * rng.normal(size=(10, 2))
    if constant_f_column:
        f["m2"] = 0.5
    if empty_f_column:
        f["m2"] = np.nan
    blocks = {"F": {"A": f}, "Z": {}, "Y": {"Q": pd.DataFrame(y, index=blends, columns=["y1", "y2"])}}
    return d_matrix, DataFrameDict(blocks)


def _unbalanced_asca() -> ASCA:
    """Fit ASCA to a crossed 2 x 3 design with one cell short of its replicates."""
    rng = np.random.default_rng(1)
    design = pd.DataFrame([{"A": a, "B": b} for a in ("lo", "hi") for b in ("m1", "m2", "m3") for _ in range(4)])
    values = pd.DataFrame(rng.normal(size=(len(design), 3)), columns=["v0", "v1", "v2"])
    values.loc[design["A"] == "hi", "v0"] += 3.0
    keep = np.r_[False, np.ones(len(design) - 1, dtype=bool)]
    return ASCA(n_components=1).fit(values[keep].reset_index(drop=True), design[keep].reset_index(drop=True))


def _plsda_permutations_that_cannot_all_be_fitted() -> object:
    """Every second shuffled fit raises, as a shuffle that leaves nothing to fit does."""
    model = PLSDA(n_components=1).fit(X, LABELS)
    original, calls = PLSDA._clone_unfitted, itertools.count()

    def clone_that_may_not_fit(self: PLSDA) -> PLSDA:
        fresh = original(self)
        if next(calls) % 2:

            def refuse(*_args: object, **_kwargs: object) -> PLSDA:
                raise NotEnoughVarianceError("simulated: nothing left to fit")

            fresh.fit = refuse  # type: ignore[method-assign]
        return fresh

    with mock.patch.object(PLSDA, "_clone_unfitted", clone_that_may_not_fit):
        return model.permutation_test(X, LABELS, n_permutations=4, cv=None, random_state=0)


def _a_selector_with_its_own_randomness() -> object:
    """Count discoveries with a selector whose two identical calls return different names."""
    counts = iter([1, 2, 0])
    names = pd.DataFrame({"a": np.arange(8.0)}, index=[f"p{i}" for i in range(8)])
    response = pd.DataFrame({"u": np.arange(8.0)}, index=names.index)
    return count_discoveries_under_null(
        lambda _x, _y: [f"c{i}" for i in range(next(counts))], names, response, n_perm=1
    )


def _batches(*, flat: bool = False, lengths: tuple[int, ...] = (40, 40)) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(2)
    return {
        f"B{i}": pd.DataFrame({"temp": rng.normal(size=n).cumsum(), "setpoint": 50.0 if flat else rng.normal(size=n)})
        for i, n in enumerate(lengths)
    }


def _spiked_without_noise() -> dict[str, pd.DataFrame]:
    """LOWESS's robustness weights collapse on a noiseless trajectory with spikes."""
    spiked = np.linspace(0.0, 5.0, 100)
    spiked[[30, 31, 70]] += 20.0
    return {"B1": pd.DataFrame({"y": spiked})}


#: Each warning, by estimator family: the call that raises it, and a pattern its message matches.
CASES: dict[str, tuple[Callable[[], object], str]] = {
    # PCA and its relatives
    "PCA: more components than the data hold": (
        lambda: PCA(n_components=5).fit(X),
        r"more than can be computed from data",
    ),
    "PCA: a column with nothing observed": (
        lambda: PCA(n_components=1, algorithm="nipals").fit(_with_an_empty_column(STANDARDISED)),
        r"Block X has columns with no observed values",
    ),
    "PCA ek: folds that cannot reach the components": (
        lambda: PCA.select_n_components(X, max_components=4, cv=10, cv_scheme="ek", random_state=0),
        r"cv_scheme='ek' can evaluate at most",
    ),
    "PCA: a criterion that never turns over": (
        lambda: PCA.select_n_components(RANK_THREE, max_components=2, cv=5, cv_scheme="ekf", random_state=0),
        r"did not turn over",
    ),
    "PCA ekf: data that are already autoscaled": (
        lambda: PCA.select_n_components(STANDARDISED, max_components=2, cv=5, cv_scheme="ekf", random_state=0),
        r"X is already centred and unit-variance scaled",
    ),
    "PCA: rows too sparse to estimate their scores": (
        lambda: PCA(n_components=2).fit(X).diagnose(X.iloc[:2].assign(b=np.nan, c=np.nan, d=np.nan)),
        r"too few observed values to estimate 2 scores",
    ),
    "PCA: the T² limit's judged_rows left at its default": (
        lambda: hotellings_t2_limit(0.95, 2, 20),
        r"judged_rows",
    ),
    "MBPCA: a column with nothing observed": (
        lambda: MBPCA(n_components=1).fit({"p": X[["a", "b"]], "q": _with_an_empty_column(X[["c", "d"]])}),
        r'Block X\["q"\] has columns with no observed values',
    ),
    "FPCA: more components than the curves support": (
        lambda: FPCA(n_components=9).fit(_curves()),
        r"components were asked for, but 5 curves",
    ),
    # PLS and its relatives
    "PLS: copy is deprecated": (
        lambda: PLS(n_components=1, copy=True).fit(X, Y),
        r"PLS\.copy is deprecated",
    ),
    "PLS: blocks that are not centred": (
        lambda: PLS(n_components=1, scale=False).fit(X + 100.0, Y - Y.mean()),
        r"PLS\(scale=False\) fits no intercept",
    ),
    "PLS: a column of X with nothing observed": (
        lambda: PLS(n_components=1).fit(_with_an_empty_column(X), Y),
        r"Block X has columns with no observed values",
    ),
    "PLS: a column of Y with nothing observed": (
        lambda: PLS(n_components=1).fit(X, _with_an_empty_column(Y)),
        r"Block Y has columns with no observed values",
    ),
    "PLS: more components than the data hold": (
        lambda: PLS(n_components=5).fit(X, Y),
        r"more than can be computed from data",
    ),
    "PLS: nothing left in Y that X can explain": (
        lambda: PLS(n_components=2).fit(DESIGN, INTERACTION),
        r"nothing left in Y that X can explain",
    ),
    "PLS: centring and scaling outside the folds": (
        lambda: PLS.select_n_components(
            X, Y, max_components=2, cv=5, n_repeats=1, scale_inside_folds=False, random_state=0
        ),
        r"scale_inside_folds=False leaks",
    ),
    "PLS: cross-validation criteria on data that are already autoscaled": (
        lambda: PLS.compare_cv_criteria(STANDARDISED, Y, max_components=2, random_state=0),
        r"X is already centred and unit-variance scaled",
    ),
    "OPLS: copy is deprecated": (
        lambda: OPLS(n_orthogonal_components=1, copy=True).fit(X, Y),
        r"OPLS\.copy is deprecated",
    ),
    "OPLS: more components than the data hold": (
        lambda: OPLS(n_orthogonal_components=4).fit(X, Y),
        r"more than can be computed from data",
    ),
    "PLSDA: copy is deprecated": (
        lambda: PLSDA(n_components=1, copy=True).fit(X, LABELS),
        r"PLSDA\.copy is deprecated",
    ),
    "PLSDA: a column with nothing observed, named by the PLS it fits": (
        lambda: PLSDA(n_components=1).fit(_with_an_empty_column(X), LABELS),
        r"Block X has columns with no observed values",
    ),
    "PLSDA: permutations that cannot be fitted": (
        _plsda_permutations_that_cannot_all_be_fitted,
        r"permutations could not be fitted",
    ),
    "PRM: weight_tol is deprecated": (
        lambda: PRM(n_components=1, weight_tol=1e-4).fit(X, Y),
        r"PRM\.weight_tol is deprecated",
    ),
    "PRM: copy is deprecated": (
        lambda: PRM(n_components=1, copy=True).fit(X, Y),
        r"PRM\.copy is deprecated",
    ),
    "PRM: more components than the data hold, from the PLS it fits": (
        lambda: PRM(n_components=5).fit(X, Y),
        r"more than can be computed from data",
    ),
    "MBPLS: nothing left in Y that X can explain": (
        lambda: MBPLS(n_components=2).fit({"a": DESIGN[["a"]], "b": DESIGN[["b"]]}, INTERACTION),
        r"nothing left in Y that X can explain",
    ),
    "MBPLS: a column with nothing observed": (
        lambda: MBPLS(n_components=1).fit({"p": X[["a", "b"]], "q": _with_an_empty_column(X[["c", "d"]])}, Y),
        r'Block X\["q"\] has columns with no observed values',
    ),
    "MBPLS: the randomization test's seed is deprecated": (
        lambda: randomization_test_mbpls(
            MBPLS(n_components=1).fit({"p": X[["a", "b"]], "q": X[["c", "d"]]}, Y),
            {"p": X[["a", "b"]], "q": X[["c", "d"]]},
            Y,
            n_permutations=2,
            seed=0,
        ),
        r"randomization_test_mbpls\(seed=\.\.\.\) is deprecated",
    ),
    "TPLS: a column with no variance": (
        lambda: TPLS(n_components=1, d_matrix=_t_shaped(constant_f_column=True)[0]).fit(
            _t_shaped(constant_f_column=True)[1]
        ),
        r"has columns with no variance",
    ),
    "TPLS: a column with nothing observed": (
        lambda: TPLS(n_components=1, d_matrix=_t_shaped(empty_f_column=True)[0]).fit(_t_shaped(empty_f_column=True)[1]),
        r"has columns with no observed values",
    ),
    "TPLS: a component that does not converge": (
        lambda: TPLS(n_components=1, d_matrix=_t_shaped()[0], max_iter=1).fit(_t_shaped()[1]),
        r"TPLS NIPALS did not converge within max_iter=1",
    ),
    "AdaptivePLS: its batch PLS finds nothing left in Y": (
        lambda: AdaptivePLS(n_components=2).fit(DESIGN, INTERACTION),
        r"nothing left in Y that X can explain",
    ),
    # Correspondence analysis and its relatives
    "CA: more axes than the table has": (
        lambda: CA(n_components=2).fit(pd.DataFrame([[10, 20, 30, 40], [20, 40, 60, 80], [30, 10, 5, 1]], dtype=float)),
        r"Asked for 2 axes",
    ),
    "MCA: more axes than the table has, from CA's fit": (
        lambda: MCA(n_components=9).fit(pd.DataFrame({"g": list("xxyyzz"), "h": list("uvuvuv")})),
        r"Asked for 9 axes",
    ),
    "FAMD: more axes than the table has": (
        lambda: FAMD(n_components=9).fit(X.assign(group=LABELS)),
        r"Asked for 9 axes",
    ),
    "MFA: more axes than the table has": (
        lambda: MFA({"p": ["a", "b"], "q": ["c", "d"]}, n_components=9).fit(X),
        r"Asked for 9 axes",
    ),
    # The other analyses
    "ASCA: an unbalanced design": (
        _unbalanced_asca,
        r"The design is unbalanced",
    ),
    "GPA: stopped before the residual settled": (
        lambda: GPA(max_iter=1).fit({name: X.iloc[:6, i : i + 2] * (i + 1) for i, name in enumerate("pqr")}),
        r"GPA stopped after max_iter=1 rounds",
    ),
    "Null counts: a selector with randomness of its own": (
        _a_selector_with_its_own_randomness,
        r"select\(\) returned different names on two identical calls",
    ),
    "Projection: TSR asked for without the training residuals": (
        lambda: operator_for_pattern(
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            np.array([4.0, 1.0]),
            np.array([True, True, False]),
        ),
        r"method='tsr' was asked for without the training residuals",
    ),
    # Batch data
    "Batch: a tag with no variance gets zero weight in batch_dtw": (
        lambda: batch_dtw(_batches(flat=True), ["temp", "setpoint"], "B0", {"show_progress": False}),
        r"zero weight: 'setpoint'",
    ),
    "Batch: a batch shorter than the smoothing window": (
        lambda: smooth_trajectories(_batches(lengths=(40, 7)), window=21),
        r"shorter than window=21",
    ),
    "Batch: LOWESS's robustness weights collapse": (
        lambda: smooth_trajectories(_spiked_without_noise(), method="lowess", frac=0.2),
        r"robustness weights collapsed",
    ),
    "Batch: a tag with a zero range": (
        lambda: determine_scaling(_batches(flat=True), columns_to_align=["temp", "setpoint"]),
        r"zero range and substituted 1\.0",
    ),
    "Batch: a tag with a zero range, through BatchScaler": (
        lambda: BatchScaler(columns_to_align=["temp", "setpoint"]).fit(_batches(flat=True)),
        r"zero range and substituted 1\.0",
    ),
    "Batch: a tag with a zero range, through DTWAligner": (
        lambda: DTWAligner(["temp", "setpoint"], reference_batch="B0", settings={"show_progress": False}).fit(
            _batches(flat=True)
        ),
        r"zero range and substituted 1\.0",
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_warning_names_the_callers_line(case: str) -> None:
    """The warning points at this file, whose line made the call, not at a line in a library."""
    call, pattern = CASES[case]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    found = [w for w in caught if re.search(pattern, str(w.message))]
    assert found, [str(w.message) for w in caught]
    assert {w.filename for w in found} == {__file__}


#: Where mutmut keeps the trampoline through which it calls each function it mutates.
_TRAMPOLINE = "/site-packages/mutmut/mutation/trampoline.py"


def _as_mutmut_calls(method: Callable) -> Callable:
    """Return ``method`` called through a frame of mutmut's trampoline, as mutmut calls a mutated function."""

    def trampoline(*args: object, **kwargs: object) -> object:
        return method(*args, **kwargs)

    code = trampoline.__code__.replace(co_filename=_TRAMPOLINE)
    return types.FunctionType(code, {"__name__": "mutmut.mutation.trampoline"}, closure=trampoline.__closure__)


def test_a_warning_under_mutation_testing_names_the_callers_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutmut's trampoline sits between a method and the library code that called it (#1092).

    Its file is outside the package, so a walk to the first frame outside the package stopped
    there, and the nightly mutation run failed before it tried a single mutant.
    """
    monkeypatch.setattr(PCA, "_fit_nipals", _as_mutmut_calls(PCA._fit_nipals))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PCA(n_components=1, algorithm="nipals", max_iter=1).fit(X)
    found = [w for w in caught if "without converging" in str(w.message)]
    assert [w.filename for w in found] == [__file__]
