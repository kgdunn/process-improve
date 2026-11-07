"""The multivariate family contract (#1149): one registry, and the rules every member keeps.

``FAMILY`` says, for each estimator of the family, how to fit it on the shared small data in raw units,
its kind (a decomposition or a regression) and its block layout. The tests below run over it, so a rule
written once is checked on every member, and a member added later is checked too:

1. **The registry is complete.** Every public estimator of ``process_improve.multivariate`` and
   ``process_improve.batch`` is in ``FAMILY`` or in ``OUT_OF_SCOPE`` with its reason.
2. **Surface parity.** A public method or attribute on one member of a kind is on every member of that
   kind, unless ``EXEMPT`` gives the reason or ``KNOWN_GAPS`` the issue. The same method takes the same
   parameter names and defaults on every member that has it.
5. **The edge-case matrix.** Each bad input of ``CASES`` is crossed with every member. A cell expects a
   result or the family's error; a raw numpy exception, a ``RuntimeWarning`` or a fit that predicts NaN
   fails it.
6. **``random_state``.** Every public callable that takes one is found by introspection, and must take
   ``None``, an int and a ``numpy.random.Generator``, say so in its signature, and give the same result
   twice for the same int. A seed that goes by another name is a gap.

Kinds 3 (behaviour rules), 4 (shapes and names), 7 (input containers), 8 (plots) and 9 (output schema)
come with the sweep of #1149.

The gaps known today are listed here, each with its issue. The lists are a ratchet, as
``tests/test_complexity_ratchet.py`` is for complexity: a gap that is fixed but still listed fails, so
each sweep PR removes the entries it fixes and the lists only shrink.
"""

from __future__ import annotations

import importlib
import inspect
import itertools
import pkgutil
import re
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold

import process_improve.batch
import process_improve.multivariate
from process_improve.batch import BatchPCA, BatchPLS
from process_improve.batch.control import evaluate_control_policies
from process_improve.multivariate.methods import (
    ASCA,
    GPA,
    MBPCA,
    MBPLS,
    OPLS,
    PCA,
    PLS,
    PLSDA,
    PRM,
    TPLS,
    AdaptivePCA,
    AdaptivePLS,
    BlockSet,
    DataFrameDict,
    NotEnoughVarianceError,
    Resampler,
    compare_cv_criteria,
    pseudo_validation_set,
    randomization_test_mbpls,
)
from process_improve.simulation import BioreactorSimulator

# --- The registry, and 1. it is complete ------------------------------------------------------------------

N_ROWS = 12
COLUMNS = ["a", "b", "c", "d", "e", "f"]
MATERIALS = [f"m{j}" for j in range(len(COLUMNS))]

#: TPLS's D: three properties of six materials. TPLS reads the six columns of X as its F, the amount of each
#: material in each blend.
D_MATRIX = {
    "G": pd.DataFrame(
        np.random.default_rng(1).normal(size=(len(MATERIALS), 3)), index=MATERIALS, columns=["p", "q", "r"]
    )
}


def raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Twelve rows of six columns on two latent directions, and one response, each in its own units."""
    rng = np.random.default_rng(0)
    latent = rng.standard_normal((N_ROWS, 2))
    centred = latent @ rng.standard_normal((2, 6)) + 0.3 * rng.standard_normal((N_ROWS, 6))
    X = pd.DataFrame(centred * [1.0, 2.0, 5.0, 10.0, 0.5, 3.0] + [50, 20, 300, 1000, 5, 80], columns=COLUMNS)
    y = pd.DataFrame({"y": 40.0 + 5.0 * (latent @ [1.0, -0.5] + 0.1 * rng.standard_normal(N_ROWS))})
    return X, y


def family_data(X: pd.DataFrame | None = None, y: pd.DataFrame | None = None) -> dict[str, Any]:
    """Every member's inputs, laid out from one X of six columns and one response; ``raw_data`` by default.

    The layouts hold the same numbers: two blocks of three columns, a batch per row of three time steps
    of two tags, or one group of six materials in TPLS's F.
    """
    if X is None or y is None:
        X, y = raw_data()
    return {
        "X": X,
        "y": y,
        "labels": np.where(y["y"].to_numpy() > np.median(y["y"].to_numpy()), "high", "low"),
        "blocks": {"b1": X[COLUMNS[:3]], "b2": X[COLUMNS[3:]]},
        "batches": {row: pd.DataFrame(X.loc[row].to_numpy().reshape(3, 2), columns=["T", "P"]) for row in X.index},
        "tpls": DataFrameDict({"F": {"G": X.set_axis(MATERIALS, axis=1)}, "Y": {"Y1": y}}),
    }


Kind = Literal["decomposition", "regression"]


@dataclass(frozen=True)
class Member:
    """One estimator of the family: how to build, fit and apply it, its kind and its block layout."""

    make: Callable[[int], BaseEstimator]  # an unfitted estimator of this many components
    fit: Callable[[BaseEstimator, dict[str, Any]], object]  # fits it on what ``family_data`` returns
    apply: Callable[[BaseEstimator, dict[str, Any]], object]  # its scores of, or predictions for, new rows
    kind: Kind
    layout: Literal["X", "blocks", "batches", "tpls"]


FAMILY: dict[str, Member] = {
    "PCA": Member(
        lambda a: PCA(n_components=a),
        lambda model, data: model.fit(data["X"]),
        lambda model, data: model.transform(data["X"]),
        "decomposition",
        "X",
    ),
    "AdaptivePCA": Member(
        lambda a: AdaptivePCA(n_components=a),
        lambda model, data: model.fit(data["X"]),
        lambda model, data: model.transform(data["X"]),
        "decomposition",
        "X",
    ),
    "MBPCA": Member(
        lambda a: MBPCA(n_components=a),
        lambda model, data: model.fit(data["blocks"]),
        lambda model, data: model.transform(data["blocks"]),
        "decomposition",
        "blocks",
    ),
    "BatchPCA": Member(
        lambda a: BatchPCA(n_components=a),
        lambda model, data: model.fit(data["batches"]),
        lambda model, data: model.transform(data["batches"]),
        "decomposition",
        "batches",
    ),
    "PLS": Member(
        lambda a: PLS(n_components=a),
        lambda model, data: model.fit(data["X"], data["y"]),
        lambda model, data: model.predict(data["X"]),
        "regression",
        "X",
    ),
    "PLSDA": Member(
        lambda a: PLSDA(n_components=a),
        lambda model, data: model.fit(data["X"], data["labels"]),
        lambda model, data: model.decision_function(data["X"]),
        "regression",
        "X",
    ),
    "PRM": Member(
        lambda a: PRM(n_components=a),
        lambda model, data: model.fit(data["X"], data["y"]),
        lambda model, data: model.predict(data["X"]),
        "regression",
        "X",
    ),
    "OPLS": Member(  # ``a`` components: one predictive, and a - 1 orthogonal to Y
        lambda a: OPLS(n_orthogonal_components=a - 1),
        lambda model, data: model.fit(data["X"], data["y"]),
        lambda model, data: model.predict(data["X"]),
        "regression",
        "X",
    ),
    "AdaptivePLS": Member(
        lambda a: AdaptivePLS(n_components=a),
        lambda model, data: model.fit(data["X"], data["y"]),
        lambda model, data: model.predict(data["X"]),
        "regression",
        "X",
    ),
    "MBPLS": Member(
        lambda a: MBPLS(n_components=a),
        lambda model, data: model.fit(data["blocks"], data["y"]),
        lambda model, data: model.predict(data["blocks"]),
        "regression",
        "blocks",
    ),
    "TPLS": Member(
        lambda a: TPLS(n_components=a, d_matrix=D_MATRIX),
        lambda model, data: model.fit(data["tpls"]),
        lambda model, data: model.diagnose(data["tpls"]).hat["Y1"],
        "regression",
        "tpls",
    ),
    "BatchPLS": Member(
        lambda a: BatchPLS(n_components=a),
        lambda model, data: model.fit(data["batches"], data["y"]),
        lambda model, data: model.predict(data["batches"]).y_hat,
        "regression",
        "batches",
    ),
}

KINDS: dict[Kind, frozenset[str]] = {
    kind: frozenset(name for name, member in FAMILY.items() if member.kind == kind)
    for kind in ("decomposition", "regression")
}

#: Public estimators that are not members of the family, each with the reason.
OUT_OF_SCOPE: dict[str, str] = {
    "CA": "Correspondence analysis of a table of counts: its rows are profiles, not measurements, and it has "
    "no SPE, T² or limits to compare. It refuses missing cells by design.",
    "MCA": "Correspondence analysis of categorical columns, as CA is of counts.",
    "FAMD": "Factor analysis of mixed continuous and categorical columns: it has no SPE, T², limits or "
    "contributions, and refuses missing cells by design.",
    "MFA": "A PCA of column groups, each weighted by its own first eigenvalue: it has no SPE, T², limits or "
    "contributions, and refuses missing cells by design.",
    "FPCA": "PCA of curves on a B-spline basis: the input is sampled curves, not rows of variables, and it "
    "has no SPE, T² or limits.",
    "ASCA": "ANOVA of a design followed by a PCA of each effect matrix: the PCA models it fits are PCA, which "
    "is in the family.",
    "GPA": "Procrustes consensus of several configurations: it fits no components.",
    "BatchMonitor": "Monitors new batches with a fitted BatchPCA or BatchPLS, which are in the family.",
}


def _is_a_preprocessing_step(cls: type) -> bool:
    """Say whether ``cls`` is a transformer with no count of components, so a preprocessing step.

    Its output has the input's variables, not components, so it has no scores, loadings, T² or SPE to
    compare: ``MCUVScaler``, ``BatchScaler``, and the batch aligners and feature extractor.
    """
    return issubclass(cls, TransformerMixin) and not {"n_components", "n_orthogonal_components"} & set(
        cls._get_param_names()
    )


def _all_subclasses(cls: type) -> set[type]:
    return {grandchild for child in cls.__subclasses__() for grandchild in (child, *_all_subclasses(child))}


PACKAGES = ("process_improve.multivariate", "process_improve.batch")


def _public_modules() -> Iterator[Any]:
    """``process_improve.multivariate`` and ``process_improve.batch``, and every module in them."""
    for package in (process_improve.multivariate, process_improve.batch):
        yield package
        for info in pkgutil.walk_packages(package.__path__, f"{package.__name__}."):
            yield importlib.import_module(info.name)


def _public_estimators() -> dict[str, type]:
    """Every public ``BaseEstimator`` defined in the two packages, found once each module is imported."""
    for _ in _public_modules():  # imports each, so the walk over the subclasses finds every class
        pass
    return {
        cls.__name__: cls
        for cls in _all_subclasses(BaseEstimator)
        if cls.__module__.startswith(PACKAGES) and not cls.__name__.startswith("_")
    }


def test_every_public_estimator_is_in_the_family_or_out_of_scope_with_its_reason() -> None:
    """A new estimator cannot skip the contract: it joins ``FAMILY``, or ``OUT_OF_SCOPE`` says why not."""
    estimators = _public_estimators()
    by_rule = {name for name, cls in estimators.items() if _is_a_preprocessing_step(cls)}
    unplaced = set(estimators) - set(FAMILY) - set(OUT_OF_SCOPE) - by_rule
    assert not unplaced, f"Add each to FAMILY, or to OUT_OF_SCOPE with its reason: {sorted(unplaced)}"
    stale = (set(FAMILY) | set(OUT_OF_SCOPE)) - set(estimators)
    assert not stale, f"Listed, but no longer a public estimator: {sorted(stale)}"
    assert not (set(FAMILY) | set(OUT_OF_SCOPE)) & by_rule, "A preprocessing step is out of scope by the rule"
    assert {"MCUVScaler", "BatchScaler"} <= by_rule


@pytest.mark.parametrize("name", list(FAMILY))
def test_every_member_fits_the_shared_data(name: str) -> None:
    """The registry's recipes work: each builds the estimator it is named for, fits it and applies it."""
    data = family_data()
    member = FAMILY[name]
    model = member.make(2)
    member.fit(model, data)
    assert type(model).__name__ == name
    assert member.layout in data
    assert np.isfinite(np.asarray(member.apply(model, data), dtype=float)).all()


# --- 2. Surface parity (#1149, aspect 10) -----------------------------------------------------------------

ADAPTIVE = frozenset({"AdaptivePCA", "AdaptivePLS"})
BATCH = frozenset({"BatchPCA", "BatchPLS"})
MULTIBLOCK = frozenset({"MBPCA", "MBPLS"})
PLS_AND_SUBCLASSES = frozenset({"PLS", "PLSDA", "PRM"})


def _all_but(*names: str) -> frozenset[str]:
    return frozenset(FAMILY) - set(names)


def _on(members: frozenset[str] | set[str], why: str, *names: str) -> dict[str, tuple[frozenset[str], str]]:
    """Say that each of ``names`` is on exactly ``members`` of the family, and why."""
    return {name: (frozenset(members), why) for name in names}


def _merged(*groups: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for group in groups:
        if twice := merged.keys() & group.keys():
            raise ValueError(f"Listed twice: {sorted(twice)}")
        merged.update(group)
    return merged


#: Names only some members of a kind have, on purpose: the reason, and the members that have each name.
EXEMPT: dict[str, tuple[frozenset[str], str]] = _merged(
    _on(
        ADAPTIVE,
        "The state and methods of the recursive update: only the adaptive models take rows one at a time "
        "after the fit.",
        *("partial_fit", "update", "adaptation_plot", "XtX_", "XtX0_", "alpha_scale_", "center_shift_"),
        *("distance_", "injection_ratio_", "kernel_update_norm_", "lambda_center_", "mx_", "scale_shift_", "sx_"),
    ),
    _on({"AdaptivePCA"}, "The adaptive PCA's loadings at the fit, before any update.", "loadings0_"),
    _on(
        {"AdaptivePLS"},
        "The Y side of the recursive update, and the prediction split into what the update changed.",
        *("XtY_", "XtY0_", "alpha_scale_y_", "beta_shift_", "decompose_prediction", "lambda_center_y_", "my_"),
        *("prediction_channels_", "sy_", "weights0_"),
    ),
    _on(
        BATCH,
        "Batch trajectories: each batch is unfolded into one row, and a batch in progress is scored online.",
        *("batch_ids_", "feature_columns_", "initial_condition_names_", "n_batches_", "n_initial_conditions_"),
        *("n_tags_", "n_timesteps_", "predict_online", "predict_online_trace", "tag_names_", "time_index_"),
        "unfold_and_scale",
    ),
    _on({"BatchPLS"}, "The error of the online predictions, as a batch goes on.", "online_rmse"),
    _on(
        MULTIBLOCK,
        "Per-block quantities: only the multiblock models have blocks.",
        *("block_hotellings_t2_", "block_informative_widths_", "block_loadings_", "block_names_", "block_scores_"),
        *("block_spe_", "block_vip_", "block_widths_", "r2_x_per_block_cumulative_", "r2_x_per_block_per_component_"),
    ),
    _on({"MBPCA"}, "The super level's loading of each block.", "super_loadings_"),
    _on(
        {"MBPLS"}, "The weight of each block, at the block and the super level.", *("block_weights_", "super_weights_")
    ),
    _on({"MBPLS"}, "The importance of each block.", "super_vip_"),
    _on(
        {"OPLS"},
        "The part of X orthogonal to Y, apart from the part that predicts it: only OPLS separates them.",
        *("correct", "orthogonal_loadings_", "orthogonal_scores_", "orthogonal_weights_", "predictive_loadings_"),
        *("predictive_scores_", "predictive_weights_", "x_filtered_"),
    ),
    _on(
        {"PLSDA"},
        "A classifier's outputs: only PLSDA predicts classes.",
        *("accuracy_", "class_statistics_", "classes_", "confusion", "confusion_matrix_", "confusion_matrix_plot"),
        *("decision_function", "n_classes_", "permutation_test", "predict_proba", "priors_", "roc_auc"),
        *("sensitivity_", "specificity_", "thresholds_"),
    ),
    _on(
        {"PRM"},
        "The robust weight of each row: only PRM weights rows down.",
        *("n_weight_iter_", "outlier_summary", "robust_weights_", "weight_shift_", "weights_converged_"),
    ),
    _on(
        {"TPLS"},
        "The blocks of T-shaped data (the properties D of the materials, the fractions F, the conditions Z) "
        "and their loadings: no other member has them.",
        *("condition_names", "material_names", "property_names", "n_conditions", "n_substances", "d_mats"),
        *("f_mats", "y_mats", "z_mats", "not_na_d", "not_na_f", "not_na_y", "not_na_z", "d_block_scaling_"),
        *("skip_f_matrix_preprocessing_", "required_blocks_", "p_loadings_f", "p_loadings_z", "q_loadings_y"),
        *("r_loadings_f", "s_loadings_d", "v_loadings_d", "w_loadings_super", "w_loadings_z"),
    ),
)

#: Names only some members of a kind have today, to be decided or fixed: the issue, and the members that
#: have each name. A name that reaches more members, or fewer, fails until its entry is updated.
KNOWN_GAPS: dict[str, tuple[frozenset[str], str]] = _merged(
    # Scoring new rows.
    _on(KINDS["regression"] - {"TPLS"}, "#1149 aspect 10: TPLS predicts through diagnose", "predict"),
    _on(_all_but("TPLS"), "#1149 aspect 10: TPLS has no transform", "transform"),
    _on(_all_but("AdaptivePCA", "AdaptivePLS", "OPLS", "BatchPLS"), "#1149 aspect 10: no diagnose", "diagnose"),
    _on(PLS_AND_SUBCLASSES | {"PCA"}, "#1149 aspect 2: project (TSR, SCP, PMP) on PCA and PLS only", "project"),
    # Contributions.
    _on(_all_but("TPLS"), "#1149 aspect 4: TPLS has no contributions", *("score_contributions", "spe_contributions")),
    _on(
        _all_but("TPLS", *MULTIBLOCK),
        "#1149 aspect 4: no T² contributions on the multiblock models or TPLS",
        "t2_contributions",
    ),
    _on(_all_but("TPLS", *BATCH), "#1169: the batch models have no group contributions", "group_contributions"),
    _on(
        _all_but("TPLS", *MULTIBLOCK, *BATCH),
        "#1149 aspect 4 and #1169: MEDA, oMEDA and the other diagnostics on the single-block models only",
        *("meda", "omeda", "observation_contributions", "project_variables", "squared_cosine"),
        "eigenvalue_summary",
    ),
    _on(_all_but(*MULTIBLOCK, *BATCH), "#1149 aspect 10: vip is a method on these", "vip"),
    # Limits.
    _on(_all_but(*MULTIBLOCK), "#1149 aspect 6: the multiblock models have block and super limits", "spe_limit"),
    _on(_all_but(*MULTIBLOCK), "#1149 aspect 6: one way to get the limits", "ellipse_coordinates"),
    _on(_all_but("TPLS", *MULTIBLOCK), "#1149 aspect 6: one way to get the limits", "score_limit"),
    _on(MULTIBLOCK, "#1149 aspect 6: one SPE-limit API", *("block_spe_limit", "super_spe_limit")),
    # Fitted attributes named their own way (#1149, Order: one naming scheme for fitted attributes).
    _on(
        _all_but("TPLS", *MULTIBLOCK),
        "#1149 aspect 10: the multiblock models have super_*, TPLS its own names",
        *("scores_", "spe_", "hotellings_t2_", "scaling_factor_for_scores_"),
    ),
    _on(
        MULTIBLOCK,
        "#1149 aspect 10: scores_ and the like under the super level's name",
        *("super_scores_", "super_hotellings_t2_", "scaling_factor_for_super_scores_"),
    ),
    _on(
        {"MBPLS"},
        "#1149 aspect 10: y_scores_, y_loadings_ and the R² of Y under MBPLS's names",
        *("super_y_scores_", "super_y_loadings_", "r2_y_cumulative_", "r2_y_per_component_", "r2_y_per_variable_"),
    ),
    _on(MULTIBLOCK, "#1149 aspect 10: r2_per_variable_ under the multiblock name", "r2_x_per_variable_"),
    _on(
        {"TPLS"},
        "#1149 aspect 10: the family's quantities under TPLS's names",
        *("t_scores_super", "hat", "hat_", "spe", "hotellings_t2", "scaling_factor_for_scores", "n_samples"),
        *("n_outputs", "observation_names", "quality_names", "fitting_statistics", "sums_of_squares_", "r2_frac"),
        *("is_fitted_", "max_iter_", "feature_importance", "help"),
    ),
    _on({"PCA", "AdaptivePCA", *BATCH}, "#1149 aspect 10: loadings_ on PCA, x_loadings_ on PLS", "loadings_"),
    _on(
        PLS_AND_SUBCLASSES | {"AdaptivePLS", "BatchPLS"},
        "#1149 aspect 10: the PLS loadings and weights, which OPLS, MBPLS and TPLS name otherwise",
        *("x_loadings_", "x_weights_", "direct_weights_"),
    ),
    _on(PLS_AND_SUBCLASSES | {"OPLS", "BatchPLS"}, "#1149 aspect 10: Y's loadings", "y_loadings_"),
    _on(PLS_AND_SUBCLASSES, "#1149 aspect 10: Y's scores and weights", *("y_scores_", "y_weights_")),
    _on(KINDS["regression"] - {"OPLS", "TPLS"}, "#1149 aspect 10: the fitted predictions", "predictions_"),
    _on(KINDS["regression"] - {"MBPLS", "TPLS"}, "#1149 aspect 10: the regression coefficients", "beta_coefficients_"),
    _on(PLS_AND_SUBCLASSES | {"BatchPLS"}, "#1149 aspect 10", "rmse_"),
    _on(PLS_AND_SUBCLASSES | {"PCA", "BatchPCA"}, "#1149 aspect 10: the R² per component", "r2_per_component_"),
    _on(PLS_AND_SUBCLASSES | {"PCA", *BATCH}, "#1149 aspect 10: the cumulative R²", "r2_cumulative_"),
    _on(PLS_AND_SUBCLASSES | {"PCA", "BatchPLS"}, "#1149 aspect 10", *("r2_per_variable_", "projection_matrix")),
    _on(PLS_AND_SUBCLASSES, "#1149 aspect 10: the R² of each Y column", "r2y_per_variable_"),
    _on(_all_but("TPLS"), "#1149 aspect 10: TPLS names these its own way", *("explained_variance_", "n_samples_")),
    _on(
        _all_but("AdaptivePCA", "AdaptivePLS", "OPLS", "TPLS", *BATCH),
        "#1149 aspect 10",
        *("fitting_info_", "has_missing_data_"),
    ),
    _on({"PCA", *MULTIBLOCK}, "#1149 aspect 10: BatchPCA takes algorithm= and does not say which", "algorithm_"),
    _on(
        {"BatchPLS"},
        "#1149 aspect 10: the names of Y's columns (TPLS: quality_names)",
        "target_names_",
    ),
    _on(
        BATCH,
        "#1149 aspect 1: the preprocessing a model learned is public under several names, or not at all",
        *("center_", "scale_"),
    ),
    _on({"BatchPLS"}, "#1149 aspect 1: the preprocessing of Y", *("y_center_", "y_scale_")),
    _on({"MBPLS"}, "#1149 aspect 1: the preprocessing of Y", "y_preproc_"),
    _on({*MULTIBLOCK, "TPLS"}, "#1149 aspect 1: the preprocessing of X", "preproc_"),
    # The sklearn estimator API (#1149, aspect 8).
    _on(_all_but("TPLS", *BATCH), "#1149 aspect 8: sklearn's count of features", "n_features_in_"),
    _on(KINDS["regression"] - {"TPLS", "BatchPLS"}, "#1149 aspect 8: the number of responses", "n_targets_"),
    _on(_all_but("TPLS", *BATCH, *ADAPTIVE), "#1149 aspect 8: sklearn's feature names", "feature_names_in_"),
    _on(_all_but(*BATCH), "#1149 aspect 8: the number of components fitted", "n_components_"),
    _on(_all_but("TPLS", "MBPLS", "BatchPLS"), "#1149 aspect 8: sklearn's TransformerMixin", "fit_transform"),
    _on(
        PLS_AND_SUBCLASSES | {"PCA", "OPLS"},
        "#1149 aspect 8: sklearn's TransformerMixin",
        *("get_feature_names_out", "set_output"),
    ),
    # Model selection and inversion.
    _on(
        PLS_AND_SUBCLASSES | {"PCA", "MBPLS"},
        "#1149 aspect 11: select_n_components on MBPCA and TPLS is to decide",
        "select_n_components",
    ),
    _on({"PCA"}, "#1149 aspect 11: ways to choose the number of components", *("minka_mle", "parallel_analysis")),
    _on(
        PLS_AND_SUBCLASSES,
        "#1149 aspect 11: cross-validation on PLS and its subclasses only",
        *("cross_validate", "nested_cv", "compare_cv_criteria", "pseudo_validation_set"),
    ),
    _on(KINDS["regression"] | {"PCA"}, "#1149 aspect 11: score on PCA, not the other decompositions", "score"),
    _on(PLS_AND_SUBCLASSES | {"PCA"}, "#1149 aspect 10", "detect_outliers"),
    _on(PLS_AND_SUBCLASSES | {"BatchPLS"}, "#1149 aspect 10", "prediction_interval"),
    _on(PLS_AND_SUBCLASSES | {"OPLS"}, "#1149 aspect 10: inversion", "invert"),
    _on(PLS_AND_SUBCLASSES | {"TPLS"}, "#1149 aspect 10: inversion", *("optimal_inversion", "inversion_sensitivity")),
    _on(PLS_AND_SUBCLASSES, "#1149 aspect 10", *("selectivity_ratio", "target_projection")),
    # Plots and display (#1149, aspect 15).
    _on(
        _all_but("TPLS", *MULTIBLOCK),
        "#1149 aspect 15: the multiblock models have super_score_plot",
        *("score_plot", "spe_plot", "t2_plot"),
    ),
    _on(MULTIBLOCK, "#1149 aspect 15: score_plot under another name", "super_score_plot"),
    _on({"MBPCA"}, "#1149 aspect 15", "super_loadings_bar_plot"),
    _on({"MBPLS"}, "#1149 aspect 15", "super_weights_bar_plot"),
    _on(
        _all_but("TPLS", "BatchPLS", *MULTIBLOCK),
        "#1149 aspect 15",
        *("loading_plot", "explained_variance_plot"),
    ),
    _on(
        _all_but("TPLS", *MULTIBLOCK, *BATCH),
        "#1149 aspect 15",
        *("correlation_loadings_plot", "meda_plot", "omeda_plot"),
    ),
    _on(PLS_AND_SUBCLASSES | {"MBPLS", "BatchPLS"}, "#1149 aspect 15", "predictions_vs_observed_plot"),
    _on(PLS_AND_SUBCLASSES, "#1149 aspect 15", "coefficient_plot"),
    _on({*MULTIBLOCK, "TPLS"}, "#1149 aspect 15: display_results on these only", "display_results"),
    _on({"TPLS"}, "#1149 aspect 15: TPLS's plot accessor and the shared theme", "plot"),
)

#: Sklearn's metadata routing: generated from each method's own signature, so it differs where they do.
_ROUTING = re.compile(r"^(set_\w+_request|get_metadata_routing)$")


@pytest.fixture(scope="module")
def fitted() -> dict[str, BaseEstimator]:
    """Every member, fitted on the shared data."""
    data = family_data()
    models = {}
    for name, member in FAMILY.items():
        models[name] = member.make(2)
        member.fit(models[name], data)
    return models


def _surface(model: BaseEstimator) -> set[str]:
    """Return the public methods and attributes of a fitted model, but its settings and sklearn's routing."""
    settings = set(type(model)._get_param_names())
    return {
        name
        for name in dir(model)
        if not name.startswith("_") and name not in settings and not _ROUTING.match(name) and hasattr(model, name)
    }


def test_a_name_on_one_member_of_a_kind_is_on_every_member(fitted: dict[str, BaseEstimator]) -> None:
    """A name some members of a kind lack is exempt, with its reason, or a known gap, with its issue.

    A gap fixed on one more member, or a name that spreads to one, fails until its entry is updated.
    """
    surfaces = {name: _surface(model) for name, model in fitted.items()}
    listed = {**EXEMPT, **KNOWN_GAPS}
    problems = []
    for attribute in sorted(set().union(*surfaces.values()) | set(listed)):
        have = frozenset(name for name, surface in surfaces.items() if attribute in surface)
        partial = [kind for kind, members in KINDS.items() if have & members and have & members != members]
        if attribute not in listed:
            if partial:
                lack = sorted(set().union(*(KINDS[kind] for kind in partial)) - have)
                problems.append(
                    f"{attribute} is on {sorted(have)} but not {lack}: add it to the others, or list it in EXEMPT "
                    "with its reason or in KNOWN_GAPS with its issue"
                )
        elif not partial:
            problems.append(f"{attribute} is listed, but every member of its kind has it, or none: remove its entry")
        elif have != listed[attribute][0]:
            problems.append(f"{attribute} is listed as on {sorted(listed[attribute][0])}, but is on {sorted(have)}")
    assert not problems, "\n".join(problems)


def test_no_name_is_both_exempt_and_a_known_gap() -> None:
    assert not EXEMPT.keys() & KNOWN_GAPS.keys()
    assert all(members <= set(FAMILY) for members, _ in [*EXEMPT.values(), *KNOWN_GAPS.values()])


#: A parameter only some members' methods take, on purpose, and the reason.
EXEMPT_PARAMETERS: dict[str, tuple[frozenset[str], str]] = {
    "initial_conditions": (BATCH, "A batch's initial conditions are a block of their own, beside its trajectories."),
}

#: Methods whose parameters differ between the members of a kind today, with the issue.
SIGNATURE_GAPS: dict[str, str] = {
    **dict.fromkeys(
        ("score_contributions", "spe_contributions", "group_contributions"),
        "#1149 aspect 4: the multiblock models take no method= or ridge=, and need group=",
    ),
    **dict.fromkeys(
        ("select_n_components", "cross_validate", "nested_cv", "compare_cv_criteria", "pseudo_validation_set"),
        "#1149 aspect 11: PLSDA takes the settings as **kwargs; MBPLS takes fewer, and names Y y",
    ),
    **dict.fromkeys(
        ("fit", "fit_transform", "transform", "score"),
        "#1149 aspect 10: the response is Y in PLS and PRM and y elsewhere; sample_weight on three; Y in transform",
    ),
    "display_results": "#1149 aspect 15: show_cumulative on the multiblock models, show_cumulative_stats on TPLS",
    "invert": "#1149 aspect 10: OPLS takes no null_space_coordinates",
    "prediction_interval": "#1149 aspect 10: BatchPLS takes no cv_result",
    "predictions_vs_observed_plot": "#1149 aspect 15: the MBPLS plot is its own, with no settings or fig",
    "vip": "#1149 aspect 16: TPLS's vip takes block= and method=, the others n_components=",
}


def _parameters(name: str, method: Callable[..., Any]) -> tuple[str, ...]:
    """Return the names and defaults a bound method takes, but ``*args``, ``**kwargs`` and exempt parameters."""
    return tuple(
        parameter.name + ("" if parameter.default is parameter.empty else f"={parameter.default!r}")
        for parameter in inspect.signature(method).parameters.values()
        if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        and name not in EXEMPT_PARAMETERS.get(parameter.name, (frozenset(), ""))[0]
    )


def test_the_same_method_takes_the_same_parameters_on_every_member_of_a_kind(
    fitted: dict[str, BaseEstimator],
) -> None:
    """Parameter names and defaults of a method agree across a kind, or ``SIGNATURE_GAPS`` gives the issue."""
    surfaces = {name: _surface(model) for name, model in fitted.items()}
    differ: dict[str, list[str]] = {}
    for kind, members in KINDS.items():
        for method in sorted(set().union(*(surfaces[name] for name in members))):
            signatures = {
                name: _parameters(name, getattr(fitted[name], method))
                for name in sorted(members)
                if method in surfaces[name]
                and callable(getattr(fitted[name], method))
                and not inspect.isclass(getattr(fitted[name], method))
            }
            if len(set(signatures.values())) > 1:
                detail = "; ".join(f"{name}({', '.join(parameters)})" for name, parameters in signatures.items())
                differ.setdefault(method, []).append(f"{kind}: {detail}")
    unlisted = [
        f"{method}: {' | '.join(details)}" for method, details in differ.items() if method not in SIGNATURE_GAPS
    ]
    assert not unlisted, "Make these agree, or list them in SIGNATURE_GAPS with the issue:\n" + "\n".join(unlisted)
    stale = set(SIGNATURE_GAPS) - set(differ)
    assert not stale, f"These now agree on every member: remove them from SIGNATURE_GAPS: {sorted(stale)}"


# --- 5. The edge-case matrix (#1149, aspect 12) -----------------------------------------------------------


@dataclass(frozen=True)
class Fitted:
    """The fit succeeds, and what the model gives complete new rows is finite.

    With ``warns``, a warning whose message matches it is emitted too.
    """

    warns: str | None = None


@dataclass(frozen=True)
class Refused:
    """The fit raises ``error``, with a message that matches ``message``."""

    error: type[Exception]
    message: str


Answer = Fitted | Refused


@dataclass(frozen=True)
class Case:
    """A bad input: X, Y and the number of components asked for, and the family's answers to it."""

    build: Callable[[], tuple[pd.DataFrame, pd.DataFrame, int]]
    answers: tuple[Answer, ...]
    kinds: frozenset[Kind] = frozenset(KINDS)
    #: The data are ``raw_data`` in other units: the model is then the same, up to those units.
    units: float = 1.0


def _design() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a 2^4 design: six orthogonal columns of X (A, B, C and their interactions), and D beside them."""
    runs = pd.DataFrame(list(itertools.product([-1.0, 1.0], repeat=4)), columns=["A", "B", "C", "D"])
    X = pd.DataFrame(
        np.column_stack([runs.A, runs.B, runs.C, runs.A * runs.B, runs.A * runs.C, runs.B * runs.C]), columns=COLUMNS
    )
    return X, runs


def _constant_x() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    X, y = raw_data()
    return X * 0.0 + X.mean(), y, 2


def _constant_y() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    X, y = raw_data()
    return X, y * 0.0 + 40.0, 2


def _y_fitted_exactly() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """One component fits ``y = A + B`` exactly, so a second has nothing left in Y to explain."""
    X, runs = _design()
    return X, pd.DataFrame({"y": runs.A + runs.B}), 2


def _y_orthogonal_to_x() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    X, runs = _design()
    return X, pd.DataFrame({"y": 2.0 * runs.D + 5.0}), 2


def _all_missing_row() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    X, y = raw_data()
    X.iloc[3, :] = np.nan
    return X, y, 2


def _all_missing_column() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    X, y = raw_data()
    X["c"] = np.nan
    return X, y, 2


def _one_row() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    X, y = raw_data()
    return X.iloc[:1], y.iloc[:1], 1


def _more_components_than_the_rank() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """X of rank 2 exactly, and four components asked for."""
    rng = np.random.default_rng(4)
    latent = rng.standard_normal((N_ROWS, 2))
    X = pd.DataFrame(latent @ rng.standard_normal((2, 6)), columns=COLUMNS)
    return X, pd.DataFrame({"y": latent @ [1.0, -0.5] + 0.1 * rng.standard_normal(N_ROWS)}), 4


def _more_components_than_rows_or_columns() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    X, y = raw_data()
    return X, y, 99


SMALL_UNITS = 1e-160


def _small_units() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    X, y = raw_data()
    return X * SMALL_UNITS, y * SMALL_UNITS, 2


#: The rule for X with no variance, decided on 6 Nov: refused everywhere (TPLS's X is its F and D).
NO_VARIANCE_IN_X = Refused(NotEnoughVarianceError, r"^There is no variance in (X|F and D): ")
REGRESSION: frozenset[Kind] = frozenset({"regression"})

CASES: dict[str, Case] = {
    "constant X": Case(_constant_x, (NO_VARIANCE_IN_X,)),
    "constant Y": Case(
        _constant_y,
        (Refused(NotEnoughVarianceError, r'^There is no variance in Y(\["[^"]+"\])?: '),),
        REGRESSION,
    ),
    "Y fitted exactly": Case(
        _y_fitted_exactly,
        (
            Refused(
                NotEnoughVarianceError,
                r"^There is no variance left in the data array for Y: cannot compute any more components beyond "
                r"component 1\.$",
            ),
        ),
        REGRESSION,
    ),
    "Y orthogonal to X": Case(
        _y_orthogonal_to_x,
        (Refused(NotEnoughVarianceError, r"^There is nothing in X to explain Y by: Y is uncorrelated with every"),),
        REGRESSION,
    ),
    # #1110: a row with nothing observed is fitted; a column with nothing observed is left out, with a warning.
    "all-missing row": Case(_all_missing_row, (Fitted(),)),
    "all-missing column": Case(_all_missing_column, (Fitted(warns=r"has columns with no observed values"),)),
    "one row": Case(_one_row, (Refused(ValueError, r"a minimum of 2 is required"),)),
    # Whether a fit refuses or goes on when X runs out part way is still to decide (#1149, Order): either is
    # the family's answer, a raw error is not.
    "more components than the rank of X": Case(
        _more_components_than_the_rank,
        (
            Fitted(),
            Refused(
                NotEnoughVarianceError,
                r"^There is no variance left in the data array for X: cannot compute any more components",
            ),
        ),
    ),
    # #505: the count is cut to what the data can give, with a warning.
    "more components than rows or columns": Case(
        _more_components_than_rows_or_columns,
        (Fitted(warns=r"^The requested number of components is more than can be computed"),),
    ),
    "small units": Case(_small_units, (Fitted(),), units=SMALL_UNITS),
}

#: A member whose answer to a case differs from the family's on purpose, with the reason.
EDGE_EXEMPT: dict[tuple[str, str], tuple[Answer, str]] = {
    ("constant Y", "PLSDA"): (
        Refused(ValueError, r"^PLSDA needs at least 2 classes"),
        "PLSDA's Y is class labels, so a constant Y is one class, and a classifier needs two.",
    ),
    ("one row", "PLSDA"): (
        Refused(ValueError, r"^PLSDA needs at least 2 classes"),
        "One row has one class, which PLSDA checks first.",
    ),
    ("Y fitted exactly", "OPLS"): (
        Fitted(),
        "OPLS fits one predictive component, which fits Y exactly here; the orthogonal ones describe X, not Y.",
    ),
    ("all-missing row", "PRM"): (
        Refused(ValueError, r"^PRM does not support missing data"),
        "PRM weights each row by its residual and leverage distances, which a row with gaps does not have.",
    ),
    ("all-missing column", "PRM"): (
        Refused(ValueError, r"^PRM does not support missing data"),
        "PRM weights each row by its residual and leverage distances, which a row with gaps does not have.",
    ),
}

#: A cell that gives another answer today, with its issue. A cell here that gives the family's answer fails.
EDGE_GAPS: dict[tuple[str, str], str] = {
    **{
        ("constant Y", name): "#1149 aspect 9: refused as 'no variance left in the data array for Y', not in the "
        "words MBPLS, OPLS and TPLS use"
        for name in ("PLS", "PRM", "AdaptivePLS", "BatchPLS")
    },
    ("Y fitted exactly", "TPLS"): "#1149 aspect 12: TPLS fits a second component, where PLS and MBPLS refuse (#1128)",
    ("Y orthogonal to X", "TPLS"): "#1169: TPLS says its scores are not finite, not that Y is uncorrelated with X",
    **{
        (case, name): "#1149 aspect 2: refuses missing values"
        for case in ("all-missing row", "all-missing column")
        for name in ("OPLS", "BatchPCA", "BatchPLS")
    },
    ("all-missing column", "AdaptivePCA"): "#1149 aspect 2: leaves the column out without the family's warning",
    ("one row", "BatchPCA"): "#1149 aspect 7: refused as an X with no variance, not as too few rows",
    ("one row", "TPLS"): "#1149 aspect 7: refused as a Y with no variance, not as too few rows",
    **{
        ("more components than rows or columns", name): "#1149 aspect 12: a raw error or a RuntimeWarning, where "
        "PCA and PLS cut the count with a warning"
        for name in ("AdaptivePCA", "AdaptivePLS", "MBPCA", "MBPLS", "TPLS")
    },
}


def _answer_matches(answer: Answer, outcome: BaseException | None, caught: list[str]) -> bool:
    if isinstance(answer, Refused):
        return isinstance(outcome, answer.error) and re.search(answer.message, str(outcome)) is not None
    return outcome is None and (answer.warns is None or any(re.search(answer.warns, m) for m in caught))


def _unit_free(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values)


def _edge_cells() -> list:
    cells = []
    for case_name, case in CASES.items():
        for name, member in FAMILY.items():
            if member.kind not in case.kinds:
                continue
            marks = []
            if (case_name, name) in EDGE_GAPS:
                reason = f"{EDGE_GAPS[case_name, name]}; once fixed, remove the cell from EDGE_GAPS"
                marks.append(pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason))
            cells.append(pytest.param(case_name, name, marks=marks, id=f"{name}-{case_name}"))
    return cells


@pytest.mark.parametrize(("case_name", "name"), _edge_cells())
def test_a_bad_input_gets_the_familys_answer_from_every_member(case_name: str, name: str) -> None:
    """A result, or the family's error: never a raw numpy error, a ``RuntimeWarning`` or a NaN prediction.

    The #1142 pattern: a guard lands in one estimator, and a sibling keeps the old behaviour. Each cell
    is run with ``RuntimeWarning`` as an error, and a fitted model is applied to complete new rows.
    """
    case, member = CASES[case_name], FAMILY[name]
    answers = (EDGE_EXEMPT[case_name, name][0],) if (case_name, name) in EDGE_EXEMPT else case.answers
    X, y, n_components = case.build()
    complete = X.fillna(raw_data()[0])  # the cells a case leaves out, back as new rows have them
    outcome: BaseException | None = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("error", RuntimeWarning)
        try:
            model = member.make(n_components)
            member.fit(model, family_data(X, y))
            applied = np.asarray(member.apply(model, family_data(complete, y)), dtype=float)
        except Exception as error:  # noqa: BLE001 - the cell judges whatever is raised
            outcome = error
    messages = [str(warning.message) for warning in caught]
    seen = f"{type(outcome).__name__}: {outcome}" if outcome is not None else f"fitted; warnings {messages}"
    assert any(_answer_matches(answer, outcome, messages) for answer in answers), f"expected {answers}, got {seen}"
    if outcome is not None:
        return
    assert np.isfinite(applied).all(), f"the fit succeeded, and what it gives new rows is not finite: {applied}"
    if case.units != 1.0:
        X_unit, y_unit = X / case.units, y / case.units
        reference = member.make(n_components)
        member.fit(reference, family_data(X_unit, y_unit))
        in_units = np.asarray(member.apply(reference, family_data(X_unit, y_unit)), dtype=float)
        np.testing.assert_allclose(_unit_free(applied), _unit_free(in_units), atol=1e-9)


def test_every_exempt_or_known_cell_is_a_cell_of_the_matrix() -> None:
    """An entry for a case or a member that is gone, or a cell both exempt and a gap, is a stale entry."""
    cells = {(case, name) for case, c in CASES.items() for name, m in FAMILY.items() if m.kind in c.kinds}
    assert set(EDGE_EXEMPT) <= cells
    assert set(EDGE_GAPS) <= cells
    assert not set(EDGE_EXEMPT) & set(EDGE_GAPS)


# --- 6. random_state (#1149, aspect 13) -------------------------------------------------------------------


def _public_callables() -> dict[str, Callable[..., Any]]:
    """Every public function, method and constructor defined in the two packages, by its name."""
    found: dict[str, Callable[..., Any]] = {}
    for module in _public_modules():
        for attribute, obj in vars(module).items():
            if attribute.startswith("_") or not str(getattr(obj, "__module__", None) or "").startswith(PACKAGES):
                continue
            if inspect.isclass(obj):
                for method in dir(obj):
                    if (method == "__init__" or not method.startswith("_")) and callable(getattr(obj, method, None)):
                        found[f"{obj.__name__}.{method}"] = getattr(obj, method)
            elif callable(obj):
                found[obj.__name__] = obj
    return found


def _parameters_of(function: Callable[..., Any]) -> dict[str, inspect.Parameter]:
    try:
        return dict(inspect.signature(function).parameters)
    except (TypeError, ValueError):  # a builtin with no signature takes no random_state either
        return {}


def _takes_random_state() -> set[str]:
    """Return the public callables that take ``random_state``, by name or through ``**kwargs``.

    A callable that takes ``**kwargs`` and has the name of one that takes ``random_state`` passes it on:
    ``PLSDA.cross_validate`` to ``PLS.cross_validate``, the function ``pseudo_validation_set`` to the method.
    """
    callables = _public_callables()
    direct = {key for key, function in callables.items() if "random_state" in _parameters_of(function)}
    passed_on = {key.rsplit(".", 1)[-1] for key in direct} - {"__init__"}
    return direct | {
        key
        for key, function in callables.items()
        if key.rsplit(".", 1)[-1] in passed_on
        and any(parameter.kind is parameter.VAR_KEYWORD for parameter in _parameters_of(function).values())
    }


Seed = int | np.random.Generator | None


def _cross_validation_calls(
    estimator: type[PLS], X: pd.DataFrame, response: object
) -> dict[str, Callable[[Seed], object]]:
    """Call each cross-validation method of PLS, or of a subclass, on small data and settings."""
    # Three rounds of PRM's row weights, not up to 30: the seed is what is tested, and the CV refits PRM often.
    settings: dict[str, Any] = {"max_weight_iter": 3} if estimator is PRM else {}

    def select(seed: Seed) -> object:
        return estimator.select_n_components(
            X, response, max_components=2, cv=3, n_repeats=2, random_state=seed, **settings
        )

    def compare(seed: Seed) -> object:
        return estimator.compare_cv_criteria(
            X, response, max_components=2, cv=3, n_permutations=9, n_cv_permutations=4, random_state=seed, **settings
        )

    # PLSDA splits an int cv without regard to the classes, so with a seed of None an inner fold of these
    # twelve rows can hold one class, which it refuses. Folds that keep both classes leave the seed to test.
    outer_folds = StratifiedKFold(n_splits=3) if estimator is PLSDA else 3

    def nested(seed: Seed) -> object:
        return estimator.nested_cv(
            X,
            response,
            max_components=2,
            outer_cv=outer_folds,
            inner_cv=3,
            n_inner_repeats=1,
            n_permutations=9,
            random_state=seed,
            **settings,
        )

    def pseudo_validation(seed: Seed) -> object:
        return estimator.pseudo_validation_set(X, response, n_components=2, cv=3, random_state=seed, **settings)

    def cross_validate(seed: Seed) -> object:
        model = estimator(n_components=2, **settings).fit(X, response)
        return model.cross_validate(X, response, cv=3, n_bootstrap=4, random_state=seed, show_progress=False)

    name = estimator.__name__
    return {
        f"{name}.select_n_components": select,
        f"{name}.compare_cv_criteria": compare,
        f"{name}.nested_cv": nested,
        f"{name}.pseudo_validation_set": pseudo_validation,
        f"{name}.cross_validate": cross_validate,
    }


def _random_state_calls() -> dict[str, Callable[[Seed], object]]:
    """Call each callable that takes ``random_state`` on small data and settings, with the seed given."""
    data = family_data()
    X, y, labels, blocks = data["X"], data["y"], data["labels"], data["blocks"]
    design = pd.DataFrame({"f": np.repeat(["low", "high"], N_ROWS // 2)})
    calls: dict[str, Callable[[Seed], object]] = {
        "ASCA.permutation_test": lambda seed: (
            ASCA(n_components=1).fit(X, design).permutation_test(n_permutations=9, random_state=seed)
        ),
        "ASCA.vasca": lambda seed: ASCA(n_components=1).fit(X, design).vasca("f", n_permutations=9, random_state=seed),
        "GPA.consensus_test": lambda seed: (
            GPA().fit({"one": X.iloc[:6], "two": X.iloc[6:].set_axis(X.index[:6])}).consensus_test(9, random_state=seed)
        ),
        "MBPLS.select_n_components": lambda seed: MBPLS.select_n_components(
            blocks, y, max_components=2, cv=3, n_repeats=2, random_state=seed
        ),
        "randomization_test_mbpls": lambda seed: randomization_test_mbpls(
            MBPLS(n_components=2).fit(blocks, y), blocks, y, n_permutations=9, random_state=seed
        ),
        "PCA.parallel_analysis": lambda seed: PCA.parallel_analysis(X, n_simulations=9, random_state=seed),
        "PCA.select_n_components": lambda seed: PCA.select_n_components(X, max_components=2, cv=3, random_state=seed),
        "PLSDA.permutation_test": lambda seed: (
            PLSDA(n_components=1).fit(X, labels).permutation_test(X, labels, n_permutations=4, cv=3, random_state=seed)
        ),
        "Resampler.__init__": lambda seed: (
            Resampler(
                MBPCA(n_components=1),
                BlockSet(blocks),
                lambda model: model.super_scores_.to_numpy(),
                use_jackknife=False,
                bootstrap_rounds=3,
                random_state=seed,
            )
            .resample(show_progress=False)
            .parameters
        ),
        "evaluate_control_policies": lambda seed: evaluate_control_policies(
            BioreactorSimulator(),
            y_target=8.0,
            n_train=30,
            n_test=1,
            per_class=False,
            include_adapted=False,
            oracle="none",
            random_state=seed,
        ),
        "compare_cv_criteria": lambda seed: compare_cv_criteria(
            X, y, max_components=2, cv=3, n_permutations=9, n_cv_permutations=4, random_state=seed
        ),
        "pseudo_validation_set": lambda seed: pseudo_validation_set(X, y, n_components=2, cv=3, random_state=seed),
    }
    for estimator, response in ((PLS, y), (PRM, y), (PLSDA, labels)):
        calls |= _cross_validation_calls(estimator, X, response)
    return calls


#: Public callables that name their seed otherwise today, with the issue.
RANDOM_STATE_GAPS: dict[str, str] = dict.fromkeys(
    ("check_predictive_signal", "count_discoveries_under_null"), "#1149 aspect 13: named seed, and an int only"
)


def _numbers(result: object) -> list[np.ndarray]:
    """Return every number in a result, in a fixed order: what two calls with the same seed must agree on.

    A fitted model or a figure in the result adds none: its numbers are in the frames beside it.
    """
    if isinstance(result, dict):  # a Bunch too
        parts = [result[key] for key in sorted(result, key=str)]
    elif isinstance(result, list | tuple):
        parts = list(result)
    else:
        if isinstance(result, pd.DataFrame):
            result = result.select_dtypes("number")
        if not isinstance(result, pd.DataFrame | pd.Series | np.ndarray | int | float | np.number):
            return []
        values = np.asarray(result)
        return [values.astype(float)] if values.dtype.kind in "biuf" else []
    return [values for part in parts for values in _numbers(part)]


def test_every_callable_that_takes_a_random_state_is_called_below() -> None:
    """Found by introspection, so a new callable with ``random_state`` joins the test, or this fails."""
    found, called = _takes_random_state(), set(_random_state_calls())
    assert found - called == set(), f"Add a small call of each to _random_state_calls: {sorted(found - called)}"
    assert called - found == set(), f"These no longer take random_state: {sorted(called - found)}"


#: The calls that take about two seconds: PRM refits its row weights in every fold.
SLOW_RANDOM_STATE_CALLS = frozenset({"PRM.compare_cv_criteria", "PRM.nested_cv"})


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(key, marks=[pytest.mark.slow] if key in SLOW_RANDOM_STATE_CALLS else [])
        for key in sorted(_random_state_calls())
    ],
)
def test_random_state_takes_none_an_int_and_a_generator_and_an_int_repeats(key: str) -> None:
    """``docs/development/reproducibility.rst``: ``int | numpy.random.Generator | None``, reproducible by int."""
    call = _random_state_calls()[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # this test is about the seed; the warnings of these small fits are not
        for seed in (None, np.random.default_rng(3)):
            try:
                call(seed)
            except Exception as error:
                raise AssertionError(f"{key} refused random_state={seed!r}: {error!r}") from error
        first, again = _numbers(call(3)), _numbers(call(3))
    assert first, f"{key} returned no numbers to compare"
    assert len(first) == len(again)
    assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(first, again, strict=True)), f"{key} differs"


def test_a_seed_is_named_random_state() -> None:
    """A public callable that takes ``seed`` or ``rng`` breaks the contract that names it ``random_state``."""
    other_names = {
        key
        for key, function in _public_callables().items()
        if {"seed", "rng", "random_seed"} & set(_parameters_of(function))
        and "random_state" not in _parameters_of(function)
    }
    assert other_names - set(RANDOM_STATE_GAPS) == set(), "Name the seed random_state, or list it in RANDOM_STATE_GAPS"
    assert set(RANDOM_STATE_GAPS) - other_names == set(), "These now take random_state: remove them from the list"


#: Callables whose signature, and docstring, say ``random_state`` is an int, though they take a Generator too.
RANDOM_STATE_ANNOTATION_GAPS: dict[str, str] = dict.fromkeys(
    ("PCA.parallel_analysis", "PCA.select_n_components"), "#1149 aspect 13: annotated int | None"
)


def test_the_signature_says_random_state_takes_a_generator() -> None:
    """``help`` and a type checker read the annotation, so a caller who reads it would not pass a Generator."""
    narrow = {
        key
        for key, function in _public_callables().items()
        if "random_state" in (parameters := _parameters_of(function))
        and "Generator" not in str(parameters["random_state"].annotation)
    }
    assert narrow - set(RANDOM_STATE_ANNOTATION_GAPS) == set(), "Annotate these int | np.random.Generator | None"
    assert set(RANDOM_STATE_ANNOTATION_GAPS) - narrow == set(), "These are annotated now: remove them from the list"
