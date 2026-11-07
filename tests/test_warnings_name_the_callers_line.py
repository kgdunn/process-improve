"""Every warning a user's call can raise names that call's line (#1079, #1092, #1169).

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
from collections.abc import Callable, Iterable
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from process_improve import tool_safety
from process_improve.batch import BatchScaler, determine_scaling, smooth_trajectories
from process_improve.experiments import (
    DesignRegion,
    Factor,
    analyze_experiment,
    analyze_omars,
    augment_design,
    evaluate_design,
    generate_design,
    marginal_means_prediction,
    optimize_responses,
    read_run_sheet,
    variance_dispersion,
)
from process_improve.interactions import interaction_terms
from process_improve.monitoring import ControlChart, calculate_cpk
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
from process_improve.regression import robust_regression
from process_improve.sensory import analyze_descriptive, mixed_assessor_model, validate_descriptive
from process_improve.univariate import ttest_paired

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


def _coded(names: Iterable[str]) -> list[Factor]:
    return [Factor(name=name, low=-1, high=1) for name in names]


def _two_level(names: str, replicates: int = 1) -> pd.DataFrame:
    """Return the 2^k factorial in coded units, ``replicates`` times over."""
    return pd.DataFrame(list(itertools.product([-1.0, 1.0], repeat=len(names))) * replicates, columns=list(names))


def _with_response(design: pd.DataFrame) -> pd.DataFrame:
    """Return the design with a response ``y`` that follows its first column."""
    noise = np.random.default_rng(3).normal(0.0, 0.3, len(design))
    return design.assign(y=10.0 + design.iloc[:, 0] + noise)


def _blends(decimals: int | None = None, *, off_the_simplex: bool = False) -> pd.DataFrame:
    """Ten blends of three components, a {3, 2} lattice, its centroid and three axial blends, with a response."""
    third, sixth = 1 / 3, 1 / 6
    blends = np.array(
        [
            *([1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [third, third, third]),
            *([2 * third, sixth, sixth], [sixth, 2 * third, sixth], [sixth, sixth, 2 * third]),
        ]
    )
    y = blends @ [10.0, 12.0, 8.0] + 6.0 * blends[:, 0] * blends[:, 1] + np.random.default_rng(1).normal(0, 0.5, 10)
    shown = blends if decimals is None else blends.round(decimals)
    frame = pd.DataFrame(shown, columns=["x1", "x2", "x3"]).assign(y=y)
    if off_the_simplex:
        frame.loc[7, "x1"] = 0.75  # this blend sums to 1.083
    return frame


def _feed_q_only_with_a_low() -> pd.DataFrame:
    """With feed Q, A is always run low, so A:feed cannot be estimated."""
    runs = [(a, b, "P") for a, b in itertools.product([-1, 1], repeat=2)] * 2
    runs += [(-1, b, "Q") for b in (-1, 1, -1, 1)]
    frame = pd.DataFrame(runs, columns=["A", "B", "feed"])
    return frame.assign(y=5 + frame["A"] + 2 * frame["B"] + np.random.default_rng(4).normal(0, 0.1, len(frame)))


def _plackett_burman_runs() -> pd.DataFrame:
    """Twelve runs for eight factors, too few for the default interactions model."""
    design = generate_design(_coded([f"X{i}" for i in range(1, 9)]), "plackett_burman", n_center_points=0).design
    return _with_response(design.drop(columns="RunOrder"))


def _reactor() -> pd.DataFrame:
    """Return a replicated 2^2 factorial in actual units, T from 100 to 200 and P from 1 to 3."""
    coded = _two_level("TP", replicates=2)
    y = 10 + coded["T"] + 2 * coded["P"] + coded["T"] * coded["P"] + np.random.default_rng(5).normal(0, 0.1, 8)
    return pd.DataFrame({"T": 150 + 50 * coded["T"], "P": 2 + coded["P"], "y": y})


def _tiny_spans_far_from_zero() -> pd.DataFrame:
    """Return a two-factor CCD whose spans are tiny next to their distance from zero, so the fit loses rank."""
    x = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1], [-1.414, 0], [1.414, 0], [0, -1.414], [0, 1.414], [0, 0]])
    y = 60 + 4 * x[:, 0] + 2 * x[:, 1] - 3 * x[:, 0] ** 2 - x[:, 1] ** 2 + np.linspace(-0.2, 0.2, 9)
    return pd.DataFrame({"T": 1e6 + 0.01 * x[:, 0], "P": 1e7 + 0.1 * x[:, 1], "y": y})


def _quadratic_model(response: str = "y") -> dict:
    """Return a second-order model in two coded factors, as analyze_experiment reports one."""
    terms = {"Intercept": 40.0, "A": 5.0, "B": -2.0, "I(A ** 2)": -3.0, "I(B ** 2)": -1.5}
    return {
        "response_name": response,
        "factor_names": ["A", "B"],
        "coefficients": [{"term": term, "coefficient": value} for term, value in terms.items()],
    }


def _goals_without_names() -> list[dict]:
    return [{"goal": "maximize", "low": 30.0, "high": 45.0}] * 2


#: Two factors in actual units, and a limit that the corner (150, 60) breaks.
_REGION = DesignRegion([Factor(name="T", low=100, high=150), Factor(name="D", low=20, high=60)], ["3*T + 5*D <= 600"])


def _panel(*, missing_cell: bool) -> pd.DataFrame:
    """Three panelists score four products twice on one attribute; P2 may leave one product out."""
    rng = np.random.default_rng(0)
    rows = [
        {"panelist_id": panelist, "session": 1, "product": product, "attribute": "A", "replicate": replicate}
        | {"score": 5.0 + i + rng.normal(0.0, 0.3)}
        for panelist in ("P0", "P1", "P2")
        for i, product in enumerate(["prod0", "prod1", "prod2", "prod3"])
        for replicate in (1, 2)
    ]
    panel = pd.DataFrame(rows)
    if missing_cell:
        panel = panel[~((panel["panelist_id"] == "P2") & (panel["product"] == "prod1"))]
    return panel


def _descriptive_analysis_of_an_incomplete_panel() -> object:
    panel = _panel(missing_cell=True)
    products = pd.DataFrame({"product": ["prod0", "prod1", "prod2", "prod3"], "d": range(4)})
    return analyze_descriptive(validate_descriptive(panel, products, mode="observational"), find_predictive=False)


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
    # Designed experiments: generating and reading a design
    "Design: blocks of a CCD's cube that confound an interaction": (
        lambda: generate_design(_coded("ABCD"), "ccd", n_blocks=5),
        r"confounds .+ with blocks",
    ),
    "Design: a budget that falls between two fractions": (
        lambda: generate_design(_coded("ABCDE"), "fractional_factorial", budget=12, n_center_points=0),
        r"run\(s\) for the fraction",
    ),
    "Design: a DSD with no degrees of freedom for error": (
        lambda: generate_design(_coded("ABCDEF"), "dsd", n_center_points=1),
        r"leaves no degrees of freedom for error",
    ),
    "Design: axial runs beyond the factors' ranges": (
        lambda: generate_design(_coded("AB"), "ccd", alpha="rotatable"),
        r"sets factors beyond the ranges given",
    ),
    "Design: hard_to_change that the optimal search ignores": (
        lambda: generate_design(_coded("ABC"), "d_optimal", budget=12, hard_to_change=["A"], n_blocks=2),
        r"hard_to_change=\['A'\] is ignored",
    ),
    "Design: no Plackett-Burman design of the smallest size": (
        lambda: generate_design(_coded([f"x{i}" for i in range(232)]), "plackett_burman", n_center_points=0),
        r"No 236-run Plackett-Burman design is built here",
    ),
    "Design: a supersaturated budget that aliases factors": (
        lambda: generate_design(_coded("ABCD"), "supersaturated", budget=4),
        r"pair\(s\) of factors are fully aliased",
    ),
    "Design: actual units, to evaluate_design": (
        lambda: evaluate_design(_two_level("AB").assign(A=lambda d: 150 + 50 * d["A"]), metric=["d_efficiency"]),
        r"evaluate_design expects factor settings in coded units",
    ),
    "Design: actual units, to variance_dispersion": (
        lambda: variance_dispersion(
            _two_level("AB").assign(A=lambda d: 150 + 50 * d["A"]), model="main_effects", radii=3, n_directions=50
        ),
        r"variance_dispersion expects factor settings in coded units",
    ),
    "Design: a run sheet with a setting outside its range": (
        lambda: read_run_sheet(pd.DataFrame({"T": [100.0, 200.0]}), [Factor(name="T", low=100, high=150)]),
        r"The run sheet sets factors outside their ranges",
    ),
    # Designed experiments: augmenting a design
    "Augment: metrics that cannot be computed": (
        lambda: augment_design(_two_level("AB").rename(columns={"A": "feed rate"}), "replicate"),
        r"Design metrics were not computed",
    ),
    "Augment: a CCD built on a cube that aliases interactions": (
        lambda: augment_design(_two_level("ABC").assign(D=lambda d: d["A"] * d["B"] * d["C"]), "upgrade_to_rsm"),
        r"the upgraded design supports only",
    ),
    "Augment: existing runs outside the region": (
        lambda: augment_design(
            _two_level("TD"), "add_runs_optimal", n_additional_runs=6, target_model="quadratic", region=_REGION
        ),
        r"1 run\(s\) of the existing design lie outside the region",
    ),
    # Designed experiments: analysing the runs
    "Analysis: blends rounded for printing": (
        lambda: analyze_experiment(_blends(3), response_column="y", model="scheffe_quadratic"),
        r"not to 1 exactly",
    ),
    "Analysis: a Scheffe model of blends that do not sum to 1": (
        lambda: analyze_experiment(_blends(off_the_simplex=True), response_column="y", model="scheffe_linear"),
        r"the Scheffe model holds no constant",
    ),
    "Analysis: an interaction that no run can estimate": (
        lambda: analyze_experiment(_feed_q_only_with_a_low(), response_column="y", analysis_type="coefficients"),
        r"fitted without A:feed",
    ),
    "Analysis: too few runs for the default interactions model": (
        lambda: analyze_experiment(_plackett_burman_runs(), response_column="y", analysis_type="coefficients"),
        r"the main-effects model was fitted instead",
    ),
    "Analysis: a run with no response": (
        lambda: analyze_experiment(
            _with_response(_two_level("AB", replicates=2)).assign(y=lambda d: d["y"].where(d.index > 0)),
            response_column="y",
            model="A + B",
        ),
        r"1 run\(s\) with a missing response or factor setting were left out",
    ),
    "Analysis: a Block column that an explicit formula leaves out": (
        lambda: analyze_experiment(
            _with_response(_two_level("AB", replicates=2)).assign(Block=[1] * 4 + [2] * 4),
            response_column="y",
            model="A + B",
        ),
        r"blocks are not added to an explicit formula",
    ),
    "Analysis: whole plots analysed by ordinary least squares": (
        lambda: analyze_experiment(
            _with_response(_two_level("AB", replicates=2)).assign(WholePlot=[1, 1, 2, 2, 3, 3, 4, 4]),
            response_column="y",
            model="A + B",
            analysis_type="coefficients",
        ),
        r"grouped into whole plots",
    ),
    "Analysis: a model that changes once its factors are coded": (
        lambda: analyze_experiment(
            _reactor(), response_column="y", model="y ~ T + T:P", analysis_type="coefficients", coding="coded"
        ),
        r"On the coded scale this model fits the data differently",
    ),
    "Analysis: units that lose the fit's rank, coded": (
        lambda: analyze_experiment(
            _tiny_spans_far_from_zero(),
            response_column="y",
            model="quadratic",
            analysis_type="coefficients",
            coding="coded",
        ),
        r"the fit as given lost rank in floating point",
    ),
    "Analysis: units that lose the model matrix's rank": (
        lambda: analyze_experiment(
            _tiny_spans_far_from_zero(), response_column="y", model="quadratic", analysis_type="coefficients"
        ),
        r"in the factors' units the model matrix lost rank",
    ),
    "Analysis: terms the design cannot estimate": (
        lambda: analyze_experiment(
            _with_response(_two_level("AB", replicates=2)),
            response_column="y",
            model="A + B + A:B + I(A**2)",
            analysis_type="coefficients",
        ),
        r"not estimable from this design",
    ),
    "Marginal means: runs that are not balanced": (
        lambda: marginal_means_prediction(
            pd.DataFrame({"A": [1, 1, 1, 2], "B": [1, 2, 1, 2]}), [1.0, 2.0, 3.0, 4.0], {"A": 1, "B": 2}
        ),
        r"The runs are not balanced",
    ),
    "OMARS: a design without the OMARS structure": (
        lambda: analyze_omars(_two_level("ABC", replicates=2).iloc[:-1], np.linspace(1.0, 2.0, 15)),
        r"does not have the OMARS structure",
    ),
    # Designed experiments: optimising the responses
    "Optimise: goals matched to models by position": (
        lambda: optimize_responses([_quadratic_model("y1"), _quadratic_model("y2")], goals=_goals_without_names()),
        r"Matching goals to fitted models by position",
    ),
    "Optimise: a region that the method ignores": (
        lambda: optimize_responses([_quadratic_model()], method="stationary_point", region=DesignRegion(_coded("AB"))),
        r"region is honoured by 'desirability' and 'pareto_front' only",
    ),
    "Optimise: limits that no setting reaches": (
        lambda: optimize_responses(
            [_quadratic_model()], goals=[{"response": "y", "goal": "maximize", "low": 500.0, "high": 600.0}]
        ),
        r"the composite desirability is 0",
    ),
    "Optimise: search_bounds that ridge analysis cannot follow": (
        lambda: optimize_responses(
            [_quadratic_model()], method="ridge_analysis", search_bounds={"A": (0.0, 2.0), "B": (-2.0, 2.0)}
        ),
        r"cannot follow search_bounds",
    ),
    # Monitoring
    "Control chart: calculate_limits(s=...) is deprecated": (
        lambda: ControlChart().calculate_limits(np.random.default_rng(6).normal(size=30), s=1.0),
        r"calculate_limits\(s=\.\.\.\) is deprecated",
    ),
    "Cpk: a column with no spread": (
        lambda: calculate_cpk(pd.DataFrame({"x": [5.0] * 10}), "x", specifications=(4.0, 6.0)),
        r"spread is zero or non-finite",
    ),
    # Sensory analysis
    "Sensory: a panelist who left out a product, to the MAM": (
        lambda: mixed_assessor_model(_panel(missing_cell=True)),
        r"a panelist did not score every product",
    ),
    "Sensory: a panelist who left out a product, through analyze_descriptive": (
        _descriptive_analysis_of_an_incomplete_panel,
        r"a panelist did not score every product",
    ),
    # The other modules
    "Interactions: parents that are not centred and scaled": (
        lambda: interaction_terms(
            pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 10.0], "b": [5.0, 3, 8, 1, 2]}), [("a", "b")]
        ),
        r"x_log does not look centred and unit-variance scaled",
    ),
    "Robust regression: the default intervals change in 3.0": (
        lambda: robust_regression(np.arange(10.0), 2.0 * np.arange(10.0) + np.random.default_rng(7).normal(size=10)),
        r"intervals='least_squares' is deprecated",
    ),
    "Tool safety: a DEFAULT_* name is deprecated": (
        lambda: tool_safety.DEFAULT_TIMEOUT_S,
        r"tool_safety\.DEFAULT_TIMEOUT_S is deprecated",
    ),
    "Univariate: a renamed key of ttest_paired's result": (
        lambda: ttest_paired(pd.Series([0.5, 0.1, 0.6, 0.2]))["z value"],
        r"ttest_paired\(\.\.\.\)\['z value'\] is deprecated",
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
