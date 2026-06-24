# (c) Kevin Dunn, 2010-2026. MIT License.

"""Characterize every 4-factor OMARS design and map them with PCA.

This standalone analysis script does three things:

1. **Enumerate the catalog.** It reproduces, with an integer program, the
   complete catalog of non-isomorphic four-factor *basic* OMARS designs from
   Nunez Ares and Goos (2020).  Their Table 2 reports exactly **41** such
   designs for ``m = 4`` factors and ``n`` in ``[8, 24]`` runs; this script
   regenerates all 41 from scratch (no external catalog is needed or shipped).
2. **Characterize each design.** Following the user's instruction, **one center
   run is added to every basic design before any metric is computed** (a basic
   design has no center run, so its pure quadratics are otherwise inestimable).
   Each centered design is then characterized with the package's design-
   evaluation tools (:func:`omars_properties` and :func:`evaluate_design`)
   under **both** analysis models the paper uses: the main-effects-plus-two-
   factor-interactions model (``"interactions"``, *ME+IE*) and the full
   second-order model (``"quadratic"``, *SOE*).
3. **Map to a low-dimensional space.** The per-design characteristics form a
   ``designs x characteristics`` matrix that is scaled with
   :class:`MCUVScaler` and projected with the package's :class:`PCA`, with
   score, loading, and SPE plots plus an R-squared summary.

The enumeration matches the paper exactly; the ``--validate-cpp`` note at the
bottom describes how to cross-check it against the original C++ ``doe_mip``
enumerator in the ``omars-original`` repository.

References
----------
.. [1] Nunez Ares, J. and Goos, P. (2020).  "Enumeration and multicriteria
   selection of orthogonal minimally aliased response surface designs."
   Technometrics, 62(1):21-36.
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

from process_improve.experiments.designs_omars import is_omars, omars_properties
from process_improve.experiments.evaluate import evaluate_design
from process_improve.multivariate.methods import PCA, MCUVScaler

logger = logging.getLogger("characterize_4factor_omars")

N_FACTORS = 4
FACTOR_NAMES = ["A", "B", "C", "D"]
# The paper bounds the run count of four-factor basic designs at the size of the
# standard response-surface designs: even n in [8, 24] (Table 2).
RUN_SIZES = range(8, 25, 2)


# ---------------------------------------------------------------------------
# Step 1: enumerate the 41 non-isomorphic four-factor basic OMARS designs
# ---------------------------------------------------------------------------


def _omega(m: int) -> list[np.ndarray]:
    """All ``{-1, 0, 1}^m`` design points except the center point (``|Omega| = 3^m - 1``)."""
    return [np.array(p, dtype=int) for p in itertools.product((-1, 0, 1), repeat=m) if any(v != 0 for v in p)]


def _feasible_sparsity_pairs(n: int) -> list[tuple[int, int]]:
    """Feasible ``(nME0, nIE0)`` sparsity pairs for ``n`` runs (paper Section 2.1).

    ``nME0`` is the number of zeros in every main-effect column and ``nIE0`` the
    number of zeros in every two-factor-interaction column.  The necessary
    conditions are: equal parity, ``(n - nIE0) % 4 == 0``,
    ``2 <= nME0 < nIE0 < n`` and ``nIE0 <= 2 * nME0``.
    """
    pairs: list[tuple[int, int]] = []
    for n_me0 in range(2, n):
        for n_ie0 in range(n_me0 + 1, n):
            if (n_me0 % 2) != (n_ie0 % 2):
                continue
            if (n - n_ie0) % 4 != 0:
                continue
            if n_ie0 > 2 * n_me0:
                continue
            pairs.append((n_me0, n_ie0))
    return pairs


def _isomorphic_index_sets(selected: frozenset[int], omega: list[np.ndarray], index_of: dict) -> set[frozenset[int]]:
    """Every isomorphic image of a design (factor permutations x level sign flips)."""
    points = [omega[r] for r in selected]
    images: set[frozenset[int]] = set()
    for perm in itertools.permutations(range(N_FACTORS)):
        cols = list(perm)
        for signs in itertools.product((1, -1), repeat=N_FACTORS):
            s = np.array(signs)
            images.add(frozenset(index_of[tuple((pt[cols] * s).tolist())] for pt in points))
    return images


def _equality(y: list, coefficients: np.ndarray, rhs: int) -> pulp.LpConstraint:
    """Build the pulp equality ``sum_r coefficients[r] * y[r] == rhs`` (zero coefficients skipped)."""
    nz = np.flatnonzero(coefficients)
    return pulp.lpSum(int(coefficients[r]) * y[r] for r in nz) == rhs


def _solve_one_sparsity_class(n: int, n_me0: int, n_ie0: int, ctx: dict, time_limit: int) -> list[np.ndarray]:
    """Enumerate every non-isomorphic basic design for one ``(n, nME0, nIE0)`` class."""
    omega, n_points, index_of = ctx["omega"], ctx["n_points"], ctx["index_of"]
    found: list[np.ndarray] = []
    forbidden: list[frozenset[int]] = []
    while True:
        y = [pulp.LpVariable(f"y_{r}", cat="Binary") for r in range(n_points)]
        problem = pulp.LpProblem("basic_design", pulp.LpMinimize)
        problem += 0  # pure feasibility
        problem += pulp.lpSum(y) == n  # Eq (1): run count
        for col in ctx["zero_main"]:  # Eq (2): balanced main-effect sparsity
            problem += _equality(y, col, n_me0)
        for col in ctx["zero_inter"]:  # Eq (3): balanced interaction sparsity
            problem += _equality(y, col, n_ie0)
        for col in ctx["prod_inter"]:  # Eq (4): orthogonal main effects
            problem += _equality(y, col, 0)
        for col in ctx["prod_triple"]:  # Eq (5): all third moments zero
            problem += _equality(y, col, 0)
        for image in forbidden:  # Eq (7): isomorphism cuts
            problem += pulp.lpSum(y[r] for r in image) <= n - 1
        problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
        if pulp.LpStatus[problem.status] != "Optimal":
            break
        selected = frozenset(r for r in range(n_points) if y[r].value() and y[r].value() > 0.5)
        if len(selected) != n:
            break
        found.append(np.array([omega[r] for r in sorted(selected)], dtype=float))
        forbidden.extend(_isomorphic_index_sets(selected, omega, index_of))
    return found


def enumerate_basic_designs(time_limit: int = 30) -> list[dict]:
    """Enumerate all non-isomorphic four-factor basic OMARS designs.

    Implements the integer feasibility system of Nunez Ares and Goos (2020),
    Equations (1)-(7): fixed run count, balanced main-effect and interaction
    sparsity, orthogonal main effects, all third moments zero, and cumulative
    isomorphism cuts so each isomorphism class is reported once.

    Returns
    -------
    list[dict]
        One entry per design with keys ``n``, ``nME0``, ``nIE0`` and ``design``
        (the ``(n, 4)`` coded basic design, no center run).
    """
    omega = _omega(N_FACTORS)
    points = np.array(omega)
    pairs = list(itertools.combinations(range(N_FACTORS), 2))
    triples = [(i, j, k) for i in range(N_FACTORS) for j in range(i, N_FACTORS) for k in range(j, N_FACTORS)]
    ctx = {
        "omega": omega,
        "n_points": points.shape[0],
        "index_of": {tuple(p): r for r, p in enumerate(omega)},
        "zero_main": [(points[:, i] == 0).astype(int) for i in range(N_FACTORS)],
        "zero_inter": [((points[:, i] * points[:, j]) == 0).astype(int) for (i, j) in pairs],
        "prod_inter": [(points[:, i] * points[:, j]).astype(int) for (i, j) in pairs],
        "prod_triple": [(points[:, i] * points[:, j] * points[:, k]).astype(int) for (i, j, k) in triples],
    }

    designs: list[dict] = []
    for n in RUN_SIZES:
        for n_me0, n_ie0 in _feasible_sparsity_pairs(n):
            designs.extend(
                {"n": n, "nME0": n_me0, "nIE0": n_ie0, "design": coded}
                for coded in _solve_one_sparsity_class(n, n_me0, n_ie0, ctx, time_limit)
            )
        logger.info("n=%2d: cumulative designs=%d", n, len(designs))
    return designs


# ---------------------------------------------------------------------------
# Step 2: characterize each design (one center run added first)
# ---------------------------------------------------------------------------


def add_center_run(design: np.ndarray) -> np.ndarray:
    """Append a single center run (a row of zeros) to a coded design."""
    return np.vstack([design, np.zeros((1, design.shape[1]))])


def _model_features(centered: pd.DataFrame, model: str, suffix: str) -> dict[str, float]:
    """Model-aware scalar characteristics for one design under one analysis model.

    ``model`` is ``"interactions"`` (ME+IE) or ``"quadratic"`` (full second
    order, SOE).  Values that are undefined because the model is singular or
    saturated for this design come back as ``nan``.
    """
    metrics = [
        "d_efficiency",
        "i_efficiency",
        "g_efficiency",
        "a_optimality",
        "e_optimality",
        "correlation",
        "alias_matrix",
        "vif",
        "condition_number",
        "degrees_of_freedom",
    ]
    res = evaluate_design(centered, model=model, metric=metrics, n_samples=20_000, random_seed=42)

    def g(value: object) -> float:
        return float(value) if isinstance(value, (int, float)) and value is not None else np.nan

    corr = res.get("correlation") or {}
    alias = res.get("alias_matrix") or {}
    vif = res.get("vif") or {}
    dof = res.get("degrees_of_freedom") or {}
    vif_values = list(vif.values()) if isinstance(vif, dict) else []
    return {
        f"d_eff_{suffix}": g(res.get("d_efficiency")),
        f"i_eff_{suffix}": g(res.get("i_efficiency")),
        f"g_eff_{suffix}": g(res.get("g_efficiency")),
        f"a_opt_{suffix}": g(res.get("a_optimality")),
        f"e_opt_{suffix}": g(res.get("e_optimality")),
        f"corr_max_r_{suffix}": g(corr.get("max_abs_r")),
        f"corr_mean_r_{suffix}": g(corr.get("mean_abs_r")),
        f"alias_max_{suffix}": g(alias.get("max_abs")),
        f"alias_frob_{suffix}": g(alias.get("frobenius_norm")),
        f"cond_number_{suffix}": g(res.get("condition_number")),
        f"vif_max_{suffix}": max(vif_values) if vif_values else np.nan,
        f"vif_mean_{suffix}": float(np.mean(vif_values)) if vif_values else np.nan,
        f"df_residual_{suffix}": g(dof.get("residual")),
    }


def characterize(designs: list[dict]) -> pd.DataFrame:
    """Build the ``designs x characteristics`` matrix (one center run added per design)."""
    rows: list[dict] = []
    index: list[str] = []
    for d in designs:
        basic = d["design"]
        centered_matrix = add_center_run(basic)
        if not is_omars(basic):  # the center run does not change the OMARS verdict
            logger.warning("design n=%d nME0=%d nIE0=%d failed is_omars", d["n"], d["nME0"], d["nIE0"])
        props = omars_properties(centered_matrix)
        centered_df = pd.DataFrame(centered_matrix, columns=FACTOR_NAMES)

        row: dict[str, float] = {
            # Group 1: size / structure (model-free).
            "n_runs_basic": d["n"],
            "n_runs": centered_matrix.shape[0],
            "nME0": d["nME0"],
            "nIE0": d["nIE0"],
            # Group 2: OMARS-defining numerics (model-free).
            "max_me_inner": props["max_main_effect_inner_product"],
            "max_me_vs_so_inner": props["max_main_vs_second_order_inner_product"],
            "max_so_correlation": props["max_second_order_correlation"],
        }
        # Groups 3-6: model-aware, computed under both ME+IE and SOE.
        row.update(_model_features(centered_df, model="interactions", suffix="mie"))
        row.update(_model_features(centered_df, model="quadratic", suffix="soe"))

        rows.append(row)
        index.append(f"D{len(rows):02d}_n{d['n']}_{d['nME0']}_{d['nIE0']}")
    return pd.DataFrame(rows, index=index)


# ---------------------------------------------------------------------------
# Step 3: PCA mapping
# ---------------------------------------------------------------------------


def run_pca(features: pd.DataFrame, n_components: int = 3, max_missing_frac: float = 0.6) -> tuple[PCA, pd.DataFrame]:
    """Scale (MCUV) and project the characteristics matrix with the package PCA.

    Constant and (almost) all-missing columns are dropped.  Remaining missing
    values are kept: ``MCUVScaler`` computes each column's mean and standard
    deviation ignoring ``nan``, and the package :class:`PCA` fits through the
    gaps with its missing-data (NIPALS) path, so the model-aware metrics of
    designs that are rank-deficient for a given model (OMARS designs confine
    aliasing to the second-order block, so the full second-order model is not
    always estimable) need no arbitrary imputation.  Returns the fitted model
    and the cleaned feature frame actually used.
    """
    usable = features.dropna(axis=1, how="all")
    non_constant = usable.max() != usable.min()  # drop constants (e.g. always-True OMARS booleans)
    usable = usable.loc[:, non_constant]
    keep = usable.isna().mean() <= max_missing_frac
    if (~keep).any():
        logger.info("dropping mostly-missing columns: %s", list(usable.columns[~keep]))
    usable = usable.loc[:, keep]

    n_missing = int(usable.isna().sum().sum())
    if n_missing:
        logger.info("keeping %d missing values; PCA uses its native missing-data (NIPALS) path", n_missing)

    scaler = MCUVScaler()
    scaled = scaler.fit_transform(usable)
    model = PCA(n_components=n_components).fit(scaled)
    logger.info("PCA fitted (has_missing_data_=%s)", getattr(model, "has_missing_data_", None))
    return model, usable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=Path, default=Path("omars_4factor_output"), help="where to write CSVs/plots")
    parser.add_argument("--components", type=int, default=3, help="number of PCA components")
    parser.add_argument("--time-limit", type=int, default=30, help="per-solve ILP time limit (seconds)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1: enumerating four-factor basic OMARS designs ...")
    designs = enumerate_basic_designs(time_limit=args.time_limit)
    logger.info("Enumerated %d non-isomorphic basic designs (paper Table 2 reports 41).", len(designs))

    logger.info("Step 2: characterizing each design (one center run added first) ...")
    features = characterize(designs)
    features_path = args.outdir / "characteristics.csv"
    features.to_csv(features_path)
    logger.info("Wrote %d x %d characteristics matrix to %s", *features.shape, features_path)

    logger.info("Step 3: PCA mapping ...")
    model, usable = run_pca(features, n_components=args.components)
    comp_t = [f"t{i + 1}" for i in range(args.components)]
    comp_p = [f"p{i + 1}" for i in range(args.components)]
    scores = pd.DataFrame(model.scores_, index=usable.index, columns=comp_t)
    loadings = pd.DataFrame(model.loadings_, index=usable.columns, columns=comp_p)
    scores.to_csv(args.outdir / "pca_scores.csv")
    loadings.to_csv(args.outdir / "pca_loadings.csv")

    r2 = np.asarray(model.r2_per_component_).ravel()
    logger.info("R2 per component: %s", ", ".join(f"{v:.3f}" for v in r2))
    logger.info("R2 cumulative:    %s", ", ".join(f"{v:.3f}" for v in np.cumsum(r2)))

    try:
        model.score_plot().write_html(str(args.outdir / "score_plot.html"))
        model.loading_plot().write_html(str(args.outdir / "loading_plot.html"))
        model.spe_plot().write_html(str(args.outdir / "spe_plot.html"))
        logger.info("Wrote score / loading / SPE plots to %s", args.outdir)
    except Exception as exc:  # noqa: BLE001 - plotting is optional, never fatal
        logger.warning("Plot rendering skipped: %s", exc)

    logger.info("Done. Outputs in %s", args.outdir.resolve())


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Cross-validation against the original C++ enumerator (optional, manual)
# ---------------------------------------------------------------------------
# The catalog above is regenerated from first principles, so it does not depend
# on any external file.  To cross-check it against the original implementation,
# build and run the C++ ``doe_mip`` enumerator from the ``kgdunn/omars-original``
# repository for m=4 and confirm it reports the same 41 non-isomorphic basic
# designs (the same run-size distribution: one each at n=8 and n=12, two at
# n=14, five at n=16, three at n=18, five at n=20, eleven at n=22, thirteen at
# n=24).  That repository ships catalogs only for 5, 6 and 7 factors, so the
# four-factor set must be produced by running the enumerator, exactly as here.
