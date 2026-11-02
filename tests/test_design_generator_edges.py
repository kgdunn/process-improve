"""Edge cases of the design generators: refusals, fallbacks and the caps on their searches.

Each test drives a generator through its public entry point to a branch the rest of the suite
does not reach, and checks what the branch does against a property worked out independently: a
run count, a balanced column, an axial distance, ``X'X``, a determinant or a known optimum.
"""

from __future__ import annotations

import io
import itertools
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

from process_improve.experiments import (
    DesignRegion,
    Factor,
    TrueProcess,
    _blocking,
    analyze_omars,
    benchmark,
    compare_designs,
    design_catalog,
    generate_design,
    generate_omars,
    optimal,
    read_run_sheet,
    trend_robust_run_order,
)
from process_improve.experiments import designs_covariates as covariates_module
from process_improve.experiments import designs_omars_ilp as omars_ilp
from process_improve.experiments import designs_split_plot as split_plot_module
from process_improve.experiments.augment import augment_design
from process_improve.experiments.designs_covariates import CovariateOptions, covariate_design
from process_improve.experiments.designs_mixture import dispatch_mixture
from process_improve.experiments.designs_response_surface import dispatch_dsd
from process_improve.experiments.designs_split_plot import (
    SplitPlotOptions,
    StripPlotLayout,
    StripPlotOptions,
    split_plot_design,
    strip_plot_design,
)
from process_improve.experiments.designs_supersaturated import supersaturated_available, supersaturated_runs
from process_improve.experiments.simulate import simulate_design_performance
from process_improve.experiments.simulations import popcorn

CODED = [Factor(name=n, low=-1, high=1) for n in ("x1", "x2", "x3")]


def _coded(names: str) -> list[Factor]:
    return [Factor(name=n, low=-1, high=1) for n in names]


def _two_level(names: str) -> pd.DataFrame:
    """Return the 2^k full factorial, in standard order, coded -1 and +1."""
    return pd.DataFrame(list(itertools.product([-1.0, 1.0], repeat=len(names))), columns=list(names))


def _fraction(names: str, generators: list[str]) -> pd.DataFrame:
    """Return a regular two-level fraction: the full factorial in the basic factors, the others their products."""
    generated = {g.split("=")[0]: g.split("=")[1] for g in generators}
    design = _two_level("".join(n for n in names if n not in generated))
    for name, word in generated.items():
        design[name] = np.prod([design[c] for c in word], axis=0)
    return design[list(names)]


def _log_det(x: np.ndarray) -> float:
    sign, value = np.linalg.slogdet(x.T @ x)
    assert sign > 0
    return float(value)


# --------------------------------------------------------------------------- blocking
def test_the_block_exchange_stops_at_its_pass_budget_with_balanced_blocks(monkeypatch) -> None:
    """With one pass allowed, the exchange still returns two near-equal blocks that keep the main effects."""
    monkeypatch.setattr(_blocking, "_EXCHANGE_PASSES", 1)
    result = generate_design(_coded("ABCD"), "box_behnken", n_blocks=2, random_state=3)
    assert result.metadata["blocking"]["method"] == "exchange"
    blocks = result.design["Block"].to_numpy()
    assert sorted(np.bincount(blocks)[1:].tolist()) == [13, 14]
    x = result.design[list("ABCD")].to_numpy(dtype=float)
    for block in (1, 2):
        x[blocks == block] -= x[blocks == block].mean(axis=0)
    assert np.linalg.matrix_rank(x) == 4  # the main effects survive a separate mean in each block


def test_more_blocks_than_the_main_effects_leave_room_for_are_refused() -> None:
    """Eleven factors in a 12-run Plackett-Burman design: 11 main effects and 2 block means need 13 runs."""
    with pytest.raises(ValueError, match="main effects cannot be estimated once each of 2 blocks"):
        generate_design(
            [Factor(name=f"x{i}", low=-1, high=1) for i in range(11)], "plackett_burman", n_blocks=2, n_center_points=0
        )


def test_block_generators_on_an_irregular_factorial_are_refused() -> None:
    """Centre runs at a label of a categorical factor leave the design not a regular two-level fraction."""
    factors = [*_coded("AB"), Factor(name="S", type="categorical", levels=["p", "q"])]
    with pytest.raises(ValueError, match="needs a regular two-level factorial; this design is not one"):
        generate_design(factors, "full_factorial", n_center_points=2, block_generators=["AB"])


# --------------------------------------------------------------------------- augmentation
def test_metrics_that_cannot_be_computed_leave_the_degrees_of_freedom_out() -> None:
    """A factor name no model formula takes stops the metrics; the explanation then has no metric lines."""
    design = _two_level("AB").rename(columns={"A": "feed rate"})
    with pytest.warns(UserWarning, match="Design metrics were not computed"):
        result = augment_design(design, "replicate")
    assert result["before_metrics"] == result["after_metrics"] == {}
    assert result["n_runs_after"] == 8
    assert "Residual degrees of freedom" not in result["explanation"]
    assert "D-efficiency" not in result["explanation"]


def test_a_semifold_keeps_the_words_without_the_fold_factor() -> None:
    """In the 2^(5-2) with D = AB and E = AC, folding on A keeps BCDE and breaks ABD and ACE."""
    design = _fraction("ABCDE", ["D=AB", "E=AC"])
    result = augment_design(design, "semifold", generators=["D=AB", "E=AC"], fold_on="A")
    assert result["defining_relation"] == ["I=BCDE"]
    runs = pd.DataFrame(result["augmented_design"])
    assert len(runs) == 12
    assert (runs[list("BCDE")].prod(axis=1) == 1).all()  # BCDE holds in every run
    abd = runs[list("ABD")].prod(axis=1)
    assert (abd.iloc[:8] == 1).all()  # ABD holds in the original runs
    assert (abd.iloc[8:] == -1).all()  # and is reversed in the added ones
    assert "Words containing A (ABD, ACE) now hold in only part of the runs" in result["explanation"]


def test_a_semifold_on_a_factor_in_no_word_breaks_no_word() -> None:
    """E is in no word of D = ABC, so the defining word ABCD holds in all 24 runs."""
    design = _fraction("ABCDE", ["D=ABC"])
    result = augment_design(design, "semifold", generators=["D=ABC"], fold_on="E")
    assert result["defining_relation"] == ["I=ABCD"]
    runs = pd.DataFrame(result["augmented_design"])
    assert len(runs) == 24
    assert (runs[list("ABCD")].prod(axis=1) == 1).all()
    assert "now hold in only part of the runs" not in result["explanation"]


def test_generators_that_define_no_word_fold_on_the_first_factor() -> None:
    """``A=A`` defines no word, so there is nothing to choose between and the first factor is folded."""
    design = _two_level("ABC")
    result = augment_design(design, "semifold", generators=["A=A"])
    assert result["fold_on"] == "A"
    assert result["defining_relation"] is None
    new = pd.DataFrame(result["new_runs"])
    expected = design[design["A"] == -1].assign(A=1.0)
    assert new.to_numpy().tolist() == expected.to_numpy().tolist()


def test_a_second_stage_after_text_block_labels_is_numbered_after_them() -> None:
    """Two blocks labelled by text: the axial stage is block 3, its axial runs at the rotatable 8^(1/4)."""
    design = pd.concat([_two_level("AB")] * 2, ignore_index=True)
    design["Block"] = ["day 1"] * 4 + ["day 2"] * 4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = augment_design(design, "upgrade_to_rsm")
    runs = pd.DataFrame(result["augmented_design"])
    assert runs["Block"].tolist() == ["day 1"] * 4 + ["day 2"] * 4 + [3] * 9
    assert result["alpha"] == pytest.approx(8**0.25)
    axial = pd.DataFrame(result["new_runs"]).iloc[:4][["A", "B"]].to_numpy()
    np.testing.assert_allclose(np.abs(axial).max(axis=1), 8**0.25)


def test_a_block_column_named_as_a_factor_gets_axial_runs_and_no_stage_label() -> None:
    """Named as a factor, Block is upgraded like one: two axial runs of its own, and no column of stages."""
    design = _two_level("AB").assign(Block=[-1.0, -1.0, 1.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = augment_design(design, "upgrade_to_rsm", factor_names=["A", "B", "Block"])
    runs = pd.DataFrame(result["augmented_design"])
    assert runs.columns.tolist() == ["A", "B", "Block"]
    assert len(runs) == 4 + 6 + 5  # the cube, two axial runs per factor and five centre runs
    root_two = math.sqrt(2)
    assert sorted(runs["Block"].iloc[4:10].tolist()) == pytest.approx([-root_two, 0, 0, 0, 0, root_two])


# --------------------------------------------------------------------------- benchmark
def test_a_study_ends_at_the_round_cap(monkeypatch) -> None:
    """Allowed one round, the study stops after its first step and reports the truth at that step's best."""
    monkeypatch.setattr(benchmark, "_MAX_ROUNDS", 1)
    case = next(c for c in benchmark.CASES if c.name == "interior optimum, 2^3 + centre")
    study = benchmark.run_study(case, random_state=0)
    assert study["actions"] == ["augment_to_rsm"]
    assert study["ended"] == "augment_to_rsm"
    a, b = study["settings"]["A"], study["settings"]["B"]
    assert study["true_value"] == pytest.approx(60 + 2 * a - 1.5 * b - 6 * a**2 - 4 * b**2 + a * b)


# --------------------------------------------------------------------------- compare
def test_a_region_with_no_support_points_is_searched_without_them() -> None:
    """A disc that holds no grid point and crosses no grid edge is searched over its boundary and a sample."""
    # The four runs at 0.5 +/- 0.05 have sum of squares 0.01 in each factor, so the main-effects
    # prediction variance at distance r from the centre is 1/n + r^2 / 0.01: 1/n + 1 on the rim.
    region = DesignRegion(_coded("AB"), ["(A - 0.5)**2 + (B - 0.5)**2 <= 0.01"])
    square = np.array([[0.45, 0.45], [0.55, 0.45], [0.45, 0.55], [0.55, 0.55], [0.5, 0.5]])
    designs = {
        "five": pd.DataFrame(square, columns=["A", "B"]),
        "six": pd.DataFrame(np.vstack([square, [[0.5, 0.5]]]), columns=["A", "B"]),
    }
    comparison = compare_designs(designs, "main_effects", region=region).to_dict()
    assert "support points" not in comparison["g_points"]
    assert comparison["g_points"].startswith("the 0 points of")
    assert comparison["values"]["five [n=5]"]["g"] == pytest.approx(1 / 5 + 1, rel=1e-6)
    assert comparison["values"]["six [n=6]"]["g"] == pytest.approx(1 / 6 + 1, rel=1e-6)


# --------------------------------------------------------------------------- design catalogue
@pytest.mark.parametrize(
    ("note", "name"),
    [
        ("Figure 1a)", "Figure 1a)"),  # a closing bracket with no opening one is kept as written
        ("../figures/doe/pareto-plot.png (an unclosed note", "Pareto plot"),
        ("Cube plot: cube.png, lines 3-5", "Cube plot"),
    ],
    ids=["unbalanced-closing-bracket", "unclosed-note-after-the-file", "caption-before-the-file-without-brackets"],
)
def test_a_figure_note_with_unbalanced_brackets_is_still_named(note: str, name: str) -> None:
    """An unmatched bracket is not taken for a note; the caption or the file still names the figure."""
    assert design_catalog.figure_name(note) == name


# --------------------------------------------------------------------------- automatic choice
def test_an_automatic_optimal_design_keeps_the_asked_model_when_no_smaller_one_fits() -> None:
    """Three runs fit neither the 7 coefficients of the interactions model nor the 4 of the main effects."""
    candidates = pd.DataFrame(list(itertools.product([-1, 0, 1], repeat=3)), columns=list("ABC"))
    with pytest.raises(ValueError, match="cannot estimate the 'interactions' model, which has 7 coefficients"):
        generate_design(_coded("ABC"), budget=3, candidates=candidates)


# --------------------------------------------------------------------------- covariates
@pytest.mark.parametrize(
    "units",
    [
        pd.DataFrame({"t": np.linspace(-1, 1, 8), "twice_t": 2 * np.linspace(-1, 1, 8)}),
        pd.DataFrame({"t": np.linspace(-1, 1, 8), "batch": np.ones(8)}),
    ],
    ids=["collinear-covariates", "constant-covariate"],
)
def test_units_whose_covariates_no_choice_can_separate_are_refused(units: pd.DataFrame) -> None:
    """A covariate that is another's multiple, or constant like the intercept, is singular whatever is chosen."""
    with pytest.raises(ValueError, match="No choice of 8 of the 8 units estimates the model"):
        covariate_design(units, CODED, 8, CovariateOptions())


def test_the_covariate_search_stops_at_its_pass_budget(monkeypatch) -> None:
    """One pass from the same start reaches no more information than the search left to converge."""
    units = pd.DataFrame({"t": np.linspace(-1, 1, 12)})
    options = CovariateOptions(n_starts=1)

    def log_det(design: pd.DataFrame) -> float:
        return _log_det(np.column_stack([np.ones(len(design)), design[["x1", "x2", "x3", "t"]].to_numpy(float)]))

    converged, _ = covariate_design(units, CODED, 8, options, random_state=4)
    monkeypatch.setattr(covariates_module, "_MAX_PASSES", 1)
    capped, meta = covariate_design(units, CODED, 8, options, random_state=4)
    assert len(capped) == 8
    assert capped.index.is_unique
    assert meta["log_det_information"] == pytest.approx(log_det(capped))
    assert log_det(capped) <= log_det(converged) + 1e-9


# --------------------------------------------------------------------------- mixtures
def test_dispatch_mixture_refuses_a_process_factor() -> None:
    """Called directly, the mixture generator refuses a factor that is not a mixture component."""
    blends = [Factor(name=n, type="mixture") for n in "AB"]
    with pytest.raises(ValueError, match="needs every factor to be a mixture component"):
        dispatch_mixture([*blends, Factor(name="T", low=20, high=80)])


def test_a_partial_simplex_centroid_over_a_million_rows_is_refused() -> None:
    """A special cubic model on 200 components blends up to three: C(200, 3) alone is 1,313,400 rows."""
    blends = [Factor(name=f"m{i}", type="mixture") for i in range(200)]
    n_rows = 200 + math.comb(200, 2) + math.comb(200, 3) + 1
    with pytest.raises(ValueError, match=f"{n_rows} rows exceed the 1M cap"):
        generate_design(blends, "mixture", model_type="scheffe_special_cubic")


# --------------------------------------------------------------------------- OMARS by integer programming
_SOLVER = {"time_limit": 30, "msg": False}


def test_satisfice_thresholds_met_by_a_multistart_design() -> None:
    """The returned design keeps every pair of second-order columns below the correlation asked for."""
    result = generate_omars(
        _coded("ABCDE"),
        n_runs=25,
        model="main_quadratic",
        n_restarts=0,
        max_candidates=0,
        satisfice={"max_second_order_correlation": 0.8},
        solver_options=_SOLVER,
    )
    assert result.metadata["omars_search"].search_mode == "multistart"
    x = result.design[list("ABCDE")].to_numpy(dtype=float)
    second = np.column_stack([x**2, *[x[:, i] * x[:, j] for i, j in itertools.combinations(range(5), 2)]])
    unit = (second - second.mean(axis=0)) / np.linalg.norm(second - second.mean(axis=0), axis=0)
    correlation = np.abs(unit.T @ unit - np.eye(unit.shape[1])).max()
    assert correlation <= 0.8
    assert result.metadata["max_second_order_correlation"] == pytest.approx(correlation)
    assert np.allclose(x.T @ second, 0.0)  # OMARS: main effects orthogonal to every second-order column


def test_a_run_window_of_one_size_is_named_by_its_run_count(monkeypatch) -> None:
    """Every solve returns a rank-deficient design; the error names "31 runs", not a window."""
    rank_deficient = [0, 1, 2, 3, 8, 26, 35, 53, 62, 71, 79, 80, 81, 98, 116]
    design = omars_ilp._foldover(omars_ilp._half_pool(5)[rank_deficient])
    monkeypatch.setattr(omars_ilp, "solve_omars_ilp", lambda *_a, **_k: (design, "Optimal", rank_deficient))
    with pytest.raises(ValueError, match=r"were found at 31 runs, but the full_second_order model"):
        generate_omars(_coded("ABCDE"), n_runs_range=(31, 31), n_restarts=1, max_candidates=0)


# --------------------------------------------------------------------------- optimal designs
def test_invalid_fixed_runs_without_a_budget_are_reported_in_full() -> None:
    """Without a budget the default size cannot use fixed runs missing a factor; the refusal names the factor."""
    with pytest.raises(ValueError, match=r"fixed_runs is missing columns for factors: \['C'\]"):
        generate_design(_coded("ABC"), "d_optimal", fixed_runs=pd.DataFrame({"A": [0.0], "B": [0.0]}))


def test_pyoptex_quadratic_model_gives_a_categorical_factor_no_square() -> None:
    """Squares of A and B, and S by its two effect columns in every interaction: 12 coefficients, all estimable."""
    pytest.importorskip("pyoptex")
    factors = [*_coded("AB"), Factor(name="S", type="categorical", levels=["p", "q", "r"])]
    result = generate_design(factors, "d_optimal", budget=14, model_type="quadratic", backend="pyoptex", random_state=1)
    assert result.metadata["backend"] == "pyoptex"
    a, b = (result.design[n].to_numpy(dtype=float) for n in "AB")
    labels = result.design["S"].to_numpy()
    s = np.column_stack([np.where(labels == "p", 1.0, 0.0), np.where(labels == "q", 1.0, 0.0)])
    s[labels == "r"] = -1.0
    model = np.column_stack([np.ones(14), a, b, s, a**2, b**2, a * b, a[:, None] * s, b[:, None] * s])
    assert model.shape[1] == 12
    assert np.linalg.matrix_rank(model) == 12
    assert len(np.unique(a)) >= 3
    assert len(np.unique(b)) >= 3


@pytest.mark.parametrize(
    "options",
    [{"fixed_runs": pd.DataFrame({"A": [1.0], "B": [0.0], "C": [0.0]})}, {"hard_to_change": ["A"]}],
    ids=["fixed-runs", "hard-to-change"],
)
def test_mixture_optimal_designs_refuse_fixed_runs_and_hard_to_change(options: dict) -> None:
    """An optimal mixture design neither augments runs already made nor makes a split-plot design."""
    blends = [Factor(name=n, type="mixture") for n in "ABC"]
    with pytest.raises(ValueError, match="fixed_runs and hard_to_change are not supported for mixture designs"):
        generate_design(blends, "d_optimal", budget=10, **options)


@pytest.mark.parametrize(
    ("design_type", "options", "reason"),
    [
        ("d_optimal", {"candidates": pd.DataFrame({"A": [-1, 1, 0], "B": [1, -1, 0], "C": [0, 1, -1]})}, "a candidate"),
        ("e_optimal", {}, "pyoptex has no E-optimality criterion"),
    ],
    ids=["candidates", "e-optimal"],
)
def test_the_pyoptex_backend_refuses_what_it_cannot_build(design_type: str, options: dict, reason: str) -> None:
    """The pyoptex backend chooses no runs from a candidate set and has no E-criterion: asking for either is refused."""
    with pytest.raises(ValueError, match=f"backend='pyoptex' cannot be used here: {reason}"):
        generate_design(_coded("ABC"), design_type, budget=10, backend="pyoptex", **options)


def test_an_unknown_backend_is_refused() -> None:
    """A backend that is not one of the three is named in the refusal."""
    with pytest.raises(ValueError, match="backend must be 'auto', 'exchange' or 'pyoptex'; got 'fast'"):
        generate_design(_coded("ABC"), "d_optimal", budget=10, backend="fast")


def test_point_exchange_stops_at_its_pass_budget(monkeypatch) -> None:
    """One pass from each start already reaches the 2^2 corners, whose X'X = 4 I gives the score -log 64."""
    candidates = pd.DataFrame(list(itertools.product([-1.0, 0.0, 1.0], repeat=2)), columns=list("AB"))
    monkeypatch.setattr(optimal, "_MAX_PASSES", 1)
    design, value = optimal.point_exchange(candidates, 4, random_state=2)
    assert sorted(map(tuple, design.to_numpy().tolist())) == [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
    assert value == pytest.approx(-math.log(64))
    assert value == pytest.approx(-_log_det(np.column_stack([np.ones(4), design.to_numpy()])))


# --------------------------------------------------------------------------- response surface designs
@pytest.mark.parametrize("alpha", [-1.0, 0])
def test_a_non_positive_numeric_alpha_is_refused(alpha: float) -> None:
    """A CCD's axial distance must be positive."""
    with pytest.raises(ValueError, match="numeric alpha \\(the axial distance\\) must be positive"):
        generate_design(_coded("ABC"), "ccd", alpha=alpha)


@pytest.mark.parametrize("alpha", [True, [1.5]], ids=["bool", "list"])
def test_an_alpha_that_is_neither_a_number_nor_a_name_is_refused(alpha: object) -> None:
    """True is not taken as the number 1, and a list is not a distance."""
    with pytest.raises(ValueError, match=r"Unknown alpha .* for a central composite design"):
        generate_design(_coded("ABC"), "ccd", alpha=alpha)


def test_a_dsd_budget_reaching_an_unbuildable_conference_order_steps_down() -> None:
    """45 runs would allow order 22, which has no conference matrix (21 is not a sum of two squares): 20 is used."""
    # Folded over, a conference matrix of order 20 gives 40 runs with X'X = 2 (20 - 1) I; the centre run adds 0.
    factors = [Factor(name=f"x{i}", low=-1, high=1) for i in range(20)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # no error df for the full second-order model: not the point here
        result = generate_design(factors, "dsd", budget=45, n_center_points=0)
    assert result.n_runs == 41
    assert result.metadata["conference_order"] == 20
    x = result.design[[f.name for f in factors]].to_numpy(dtype=float)
    np.testing.assert_array_equal(x.T @ x, 38 * np.eye(20))


def test_dispatch_dsd_refuses_a_three_level_categorical_factor() -> None:
    """Called directly, the DSD generator refuses a categorical factor that is not two-level."""
    factors = [*_coded("ABCD"), Factor(name="S", type="categorical", levels=["a", "b", "c"])]
    with pytest.raises(ValueError, match="Factor 'S' has 3 levels; definitive screening designs take two-level"):
        dispatch_dsd(factors)


def test_more_than_seven_categorical_factors_use_the_one_at_a_time_search() -> None:
    """Eight two-level categorical factors, past the exhaustive sign search: each takes each label in 11 of 22 runs."""
    # The continuous factors keep the conference matrix's orthogonality, and sit at the centre in four
    # runs: their two zeros in the foldover and the two added runs.
    categorical = [Factor(name=f"c{i}", type="categorical", levels=["lo", "hi"]) for i in range(8)]
    result = generate_design([*_coded("AB"), *categorical], "dsd", n_center_points=0)
    assert result.n_runs == 22
    assert result.metadata["n_categorical"] == 8
    for factor in categorical:
        assert result.design[factor.name].value_counts().to_dict() == {"lo": 11, "hi": 11}
    x = result.design[["A", "B"]].to_numpy(dtype=float)
    assert x[:, 0] @ x[:, 1] == 0
    assert ((x == 0).sum(axis=0) == 4).all()
    codes = np.column_stack([np.where(result.design[f.name] == "hi", 1.0, -1.0) for f in categorical])
    assert np.linalg.matrix_rank(np.column_stack([np.ones(22), x, codes])) == 11


# --------------------------------------------------------------------------- space-filling designs
def test_a_maximin_design_still_improving_stops_at_the_pass_cap() -> None:
    """Six runs in the square need eleven exchange passes from this start; the ten allowed come close to the best."""
    # No six points of a square of side 2 are further apart than 2 * sqrt(13) / 6, the known maximin packing.
    result = generate_design(_coded("AB"), "maximin", budget=6, random_state=51)
    x = result.design[["A", "B"]].to_numpy(dtype=float)
    assert np.all(np.abs(x) <= 1.0 + 1e-12)
    closest = pdist(x).min()
    assert result.metadata["min_distance"] == pytest.approx(closest)
    best = math.sqrt(13) / 3
    assert 0.95 * best <= closest <= best + 1e-12


# --------------------------------------------------------------------------- split-plot and strip-plot designs
def test_the_split_plot_search_stops_at_its_pass_budget(monkeypatch) -> None:
    """The criterion is log|X'V^-1 X| with V = I + eta Z Z'; one pass reaches no more than the converged search."""
    options = SplitPlotOptions(model_type="main_effects", n_whole_plots=4, n_starts=1)
    _, converged_meta = split_plot_design(CODED, ["x1"], 12, options, random_state=0)
    monkeypatch.setattr(split_plot_module, "_MAX_PASSES", 1)
    design, meta = split_plot_design(CODED, ["x1"], 12, options, random_state=0)
    plots = np.asarray(meta["whole_plot"])
    for w in np.unique(plots):
        assert len(set(design[plots == w, 0])) == 1  # x1 is set once per whole plot
    z = (plots[:, None] == np.unique(plots)[None, :]).astype(float)
    v_inverse = np.linalg.inv(np.eye(12) + 0.5 * z @ z.T)
    x = np.column_stack([np.ones(12), design.astype(float)])
    assert meta["log_det_information"] == pytest.approx(np.linalg.slogdet(x.T @ v_inverse @ x)[1])
    assert meta["log_det_information"] <= converged_meta["log_det_information"] + 1e-9


def test_a_strip_plot_grid_that_cannot_vary_a_row_factor_is_refused() -> None:
    """Every run is in row 0, so the row factor x1 is one setting for all six runs."""
    cells = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 0), (0, 1))
    with pytest.raises(ValueError, match="No design on this grid of 6 runs estimates the main_effects model"):
        strip_plot_design(CODED, StripPlotLayout(("x1",), ("x3",), cells), StripPlotOptions(), random_state=1)


def test_the_strip_plot_search_stops_at_its_pass_budget(monkeypatch) -> None:
    """Rows set x1 and x2, columns set x3: the capped search keeps them so, and reaches no more than the full one."""
    cells = tuple((r, (r + c) % 4) for r in range(4) for c in range(3))
    layout = StripPlotLayout(("x1", "x2"), ("x3",), cells)
    options = StripPlotOptions(n_starts=1)
    _, converged = strip_plot_design(CODED, layout, options, random_state=1)
    monkeypatch.setattr(split_plot_module, "_MAX_PASSES", 1)
    design, meta = strip_plot_design(CODED, layout, options, random_state=1)
    rows, columns = np.asarray(meta["row"]), np.asarray(meta["column"])
    design = design.astype(float)
    for r in np.unique(rows):
        assert len(np.unique(design[rows == r, :2], axis=0)) == 1
    for c in np.unique(columns):
        assert len(np.unique(design[columns == c, 2])) == 1
    same_row = (rows[:, None] == rows[None, :]).astype(float)
    same_column = (columns[:, None] == columns[None, :]).astype(float)
    v_inverse = np.linalg.inv(np.eye(12) + same_row + same_column)
    x = np.column_stack([np.ones(12), design])
    assert meta["log_det_information"] == pytest.approx(np.linalg.slogdet(x.T @ v_inverse @ x)[1])
    assert meta["log_det_information"] <= converged["log_det_information"] + 1e-9


# --------------------------------------------------------------------------- supersaturated designs
def test_a_budget_below_two_runs_has_no_supersaturated_design() -> None:
    """A two-level design needs two runs at least, so one run gives no run count."""
    assert supersaturated_runs(5, 1) is None
    assert supersaturated_available(5, 1) is False


def test_a_supersaturated_design_needs_three_factors() -> None:
    """Two factors are refused: the generator needs three or more."""
    with pytest.raises(ValueError, match="A supersaturated design needs at least 3 factors"):
        generate_design(_coded("AB"), "supersaturated")


# --------------------------------------------------------------------------- factors and results
def test_a_categorical_factor_has_no_centre_or_range() -> None:
    """Labels have no midpoint and no numeric range."""
    factor = Factor(name="S", type="categorical", levels=["a", "b"])
    assert factor.center is None
    assert factor.range is None


def test_a_fraction_s_repr_names_its_resolution_and_generators() -> None:
    """D = AB and E = AC give the length-3 word ABD, so the resolution is 3."""
    result = generate_design(_coded("ABCDE"), "fractional_factorial", generators=["D=AB", "E=AC"], n_center_points=0)
    assert repr(result) == (
        "DesignResult(type='fractional_factorial', runs=8, factors=5, resolution=3, generators=['D=AB', 'E=AC'])"
    )


# --------------------------------------------------------------------------- OMARS analysis
def _three_level_grid() -> pd.DataFrame:
    return pd.DataFrame(list(itertools.product([-1.0, 0.0, 1.0], repeat=3)), columns=list("ABC"))


def test_without_active_main_effects_heredity_does_not_bite() -> None:
    """Pure curvature in C: no main effect is active, so strong heredity cannot rule out C^2."""
    design = _three_level_grid()
    y = 5 * design["C"].to_numpy() ** 2 + np.random.default_rng(1).normal(0, 0.3, len(design))
    result = analyze_omars(design, y, quadratic_heredity="strong", interaction_heredity="strong")
    assert result.active_main_effects == []
    assert result.active_quadratics == ["C^2"]
    assert result.active_interactions == []


def test_strong_heredity_with_one_active_parent_leaves_no_candidates() -> None:
    """A two-level design has no squares, and one active main effect makes no pair: B:C is not searched."""
    corners = _two_level("ABC")
    design = pd.concat([corners, corners], ignore_index=True)
    x = design.to_numpy()
    y = 5 * x[:, 0] + 4 * x[:, 1] * x[:, 2] + np.random.default_rng(5).normal(0, 0.2, len(design))
    result = analyze_omars(design, y, interaction_heredity="strong")
    assert result.active_main_effects == ["A"]
    assert result.second_order_overall_p_value < 0.05  # the gate opens on B:C
    assert result.active_interactions == []
    assert "candidate_terms" not in result.details


def test_forcing_an_active_main_effect_changes_nothing() -> None:
    """A is active on its own, so forcing it adds nothing and is not reported as forced."""
    design = _three_level_grid()
    x = design.to_numpy()
    y = 6 * x[:, 0] + 5 * x[:, 1] * x[:, 2] + 4 * x[:, 0] ** 2 + np.random.default_rng(2).normal(0, 0.3, len(design))
    result = analyze_omars(design, y, force_main_effects=["A"])
    assert result.active_main_effects == ["A"]
    assert result.forced_main_effects == []


def test_a_subset_limit_of_zero_declares_no_second_order_terms() -> None:
    """The gate opens on B:C and A^2, but a limit of zero terms leaves the subset search nothing to add."""
    design = _three_level_grid()
    x = design.to_numpy()
    y = 6 * x[:, 0] + 5 * x[:, 1] * x[:, 2] + 4 * x[:, 0] ** 2 + np.random.default_rng(2).normal(0, 0.3, len(design))
    result = analyze_omars(design, y, max_subset_terms=0)
    assert result.second_order_overall_p_value < 0.05
    assert (result.subset_limit, result.subset_limit_reason) == (0, "user specified (0)")
    assert result.active_interactions == []
    assert result.active_quadratics == []


# --------------------------------------------------------------------------- run order
def test_replicated_runs_are_ordered_orthogonal_to_the_linear_trend() -> None:
    """Two copies of the 2^2: identical runs are never exchanged, and both factors end orthogonal to the trend."""
    design = pd.concat([_two_level("AB")] * 2, ignore_index=True)
    found = trend_robust_run_order(design, model="main_effects", degree=1, random_state=0)
    assert sorted(found["order"]) == list(range(8))
    linear = np.arange(8) - 3.5
    np.testing.assert_allclose(design.to_numpy()[found["order"]].T @ linear, 0.0, atol=1e-12)


# --------------------------------------------------------------------------- run sheets
SHEET_FACTORS = [
    Factor(name="NCO_OH", low=0.9, high=1.3),
    Factor(name="Supplier", type="categorical", levels=["north", "south"]),
]


@pytest.mark.parametrize(
    ("source", "reason"),
    [(io.StringIO(""), "No columns to parse"), (io.BytesIO(b"\xff\xfe\xfa,\x00\n\x81"), "can't decode byte")],
    ids=["empty", "not-text"],
)
def test_a_file_object_that_is_not_csv_is_refused(source: io.IOBase, reason: str) -> None:
    """An empty or undecodable file object is refused with the reader's reason."""
    with pytest.raises(ValueError, match=f"The run sheet could not be read as CSV: .*{reason}"):
        read_run_sheet(source, SHEET_FACTORS)


def test_a_workbook_with_no_sheet_holding_every_factor_is_refused() -> None:
    """Each sheet holds one of the two factors, so none is read, and the sheets are named."""
    pytest.importorskip("openpyxl")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        pd.DataFrame({"NCO_OH": [1.0]}).to_excel(writer, sheet_name="first", index=False)
        pd.DataFrame({"Supplier": ["north"]}).to_excel(writer, sheet_name="second", index=False)
    with pytest.raises(ValueError, match=r"No sheet of the workbook \(first, second\) has a column for every factor"):
        read_run_sheet(buffer.getvalue(), SHEET_FACTORS)


def test_a_sheet_of_categorical_factors_only_is_read_in_actual_units() -> None:
    """With no numeric setting there is nothing to read as coded, so the units are actual."""
    sheet = pd.DataFrame({"Supplier": ["north", "south"], "y": [1.0, 2.0]})
    read = read_run_sheet(sheet, SHEET_FACTORS[1:])
    assert read.attrs["units"] == "actual"
    assert read["Supplier"].tolist() == ["north", "south"]
    assert read["y"].tolist() == [1.0, 2.0]


# --------------------------------------------------------------------------- simulation
def test_a_true_term_constant_on_the_runs_is_never_found() -> None:
    """On two-level runs A^2 is 1 in every run: it cannot be estimated, so no study finds it, while A always is."""
    design = generate_design(_coded("ABC"), "full_factorial", n_center_points=0, n_replicates=2, random_state=0)
    truth = TrueProcess({"Intercept": 10, "A": 3, "A^2": 2}, sigma=1)
    report = simulate_design_performance(design, truth, n_simulations=20)
    found = report["designs"]["design [n=16]"]["found"]
    assert found == {"A": 1.0, "I(A ** 2)": 0.0}


def test_popcorn_reads_a_zero_dimensional_array_as_its_number() -> None:
    """A 0-d array holds one number, and the simulator answers as it does for that number."""
    assert popcorn(t=np.array(120.0), random_state=3) == popcorn(t=120.0, random_state=3)
