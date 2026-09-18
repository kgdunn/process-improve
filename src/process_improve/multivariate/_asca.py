# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""ANOVA-Simultaneous Component Analysis (#372).

ASCA joins the two halves of this package that otherwise never meet: the designed
experiments in :mod:`process_improve.experiments` and the latent-variable models in
:mod:`process_improve.multivariate`. A response matrix is partitioned by its design
terms the way classical ANOVA partitions a single response, and each term's effect
matrix is then given its own PCA. The result answers what a plain PCA of the same
matrix cannot: which *factor* is responsible for which direction of variation, whether
that factor's effect is larger than chance, and which variables carry it.

Two variants ride on the same decomposition:

* ``add_residuals=True`` puts the residual matrix back on each effect before the PCA
  (APCA / ASCA+), so a score plot shows the scatter around each factor level rather
  than the handful of coincident points a pure effect matrix produces.
* :meth:`ASCA.vasca` ranks the variables by their contribution to a term and walks
  down the ranking, which is a far more powerful test when the effect lives in a few
  variables out of many.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch

from process_improve._random import check_random_state
from process_improve.experiments._lm import validate_identifier_is_safe
from process_improve.univariate.metrics import benjamini_hochberg

from ._common import DataMatrix, _model_method, epsqrt
from ._pca import PCA
from .plots import effect_summary_plot as _effect_summary_plot

#: Name used for the residual term wherever the design terms are keyed.
RESIDUAL = "residual"


def _sum_coded_rhs(factors: list[str], model: str) -> str:
    """Build a patsy right-hand side with sum-to-zero coding for every factor.

    Sum coding (``C(name, Sum)``) is what makes the decomposition an ANOVA rather than
    a regression against an arbitrary reference level: the coefficients for a factor sum
    to zero, so each effect matrix has zero column mean and the terms are mutually
    orthogonal whenever the design is balanced. Treatment coding would put the reference
    level's mean into the intercept and leave the effect matrices neither centred nor
    orthogonal.

    Parameters
    ----------
    factors : list[str]
        Column names in the design frame. Each is validated before it can reach patsy
        (SEC-14), since the name is interpolated into a formula string.
    model : str
        ``"main_effects"``, ``"interactions"`` (every two-way interaction as well), or a
        right-hand side to use verbatim.

    Returns
    -------
    rhs : str
        A patsy right-hand side, without the leading ``~``.
    """
    for name in factors:
        validate_identifier_is_safe(name)
    coded = [f"C({name}, Sum)" for name in factors]
    joined = " + ".join(coded)
    if model == "main_effects":
        return joined
    if model == "interactions":
        return f"({joined}) ** 2" if len(coded) > 1 else joined
    return model


def _pretty(term_name: str) -> str:
    """Strip patsy's coding wrapper, so ``C(A, Sum):C(B, Sum)`` reads as ``A:B``."""
    import re  # noqa: PLC0415 - local, used only for this one cosmetic substitution

    return re.sub(r"C\(([^,()]+), Sum\)", r"\1", term_name)


class ASCA(BaseEstimator):
    r"""ANOVA-Simultaneous Component Analysis: PCA of each design term's effect.

    The response matrix is written as

    .. math::

        X = \mathbf{1} m^T + \sum_f X_f + E

    with :math:`m` the grand mean, one effect matrix :math:`X_f` per design term, and
    :math:`E` the residual. Each :math:`X_f` then gets its own PCA, so a term's scores
    and loadings describe the multivariate structure of that factor's effect alone.

    Parameters
    ----------
    n_components : int, optional
        Components to extract per term. Capped per term at the rank its design columns
        can support, because an effect matrix for a two-level factor has rank 1 and
        asking for two components of it is asking for a component that does not exist.
        Default 2.
    model : str, optional
        ``"interactions"`` (the default) adds every two-way interaction to the main
        effects; ``"main_effects"`` fits main effects only. Anything else is taken as a
        patsy right-hand side and used verbatim, in which case the caller is responsible
        for the coding.
    add_residuals : bool, optional
        If True, add the residual matrix back onto each effect matrix before its PCA
        (APCA / ASCA+). The effect matrix alone has one distinct row per factor-level
        combination, so a score plot of it is a handful of points; adding the residuals
        back restores the scatter and shows whether the levels actually separate.
        Default False.
    scale : bool, optional
        If True, unit-variance scale the columns of ``X`` after centring, so a variable
        measured in large units does not dominate the decomposition. Default False,
        which is the right choice when the columns are already on one scale (a spectrum,
        say) and the wrong one when they are not.

    Attributes
    ----------
    terms_ : list[str]
        Design term names, in the order patsy resolved them, with the coding wrapper
        stripped: ``["A", "B", "A:B"]``.
    grand_mean_ : pd.Series
        Column means removed before the decomposition.
    column_scale_ : pd.Series
        Column scaling applied, all ones when ``scale=False``.
    effect_matrices_ : dict[str, pd.DataFrame]
        One ``N x K`` effect matrix per term.
    residuals_ : pd.DataFrame
        What no term explains.
    ssq_ : pd.Series
        Sum of squares per term, plus ``"residual"`` and ``"total"``.
    ssq_percent_ : pd.Series
        The same as a percentage of the total, which is the "factor effect" summary
        worth reading first.
    models_ : dict[str, PCA]
        The fitted per-term PCA, keyed by term.
    pvalues_ : pd.Series
        Permutation p-values per term. Absent until :meth:`permutation_test` is called.
    is_balanced_ : bool
        Whether every factor-level combination occurs equally often.

    Notes
    -----
    **Balance matters, and the model says so rather than assuming it.** On a balanced
    design the terms are orthogonal, the per-term sums of squares add up to the model
    sum of squares, and the decomposition is unique. On an unbalanced design they are
    not orthogonal: the split of the shared variation between correlated terms depends
    on how you choose to attribute it, which is the Type I / II / III question. This
    implementation fits every term simultaneously by least squares and reads each term's
    fitted contribution off that single fit, which is the usual ASCA treatment. It warns
    when the design is unbalanced, and :attr:`ssq_` then no longer partitions the total
    exactly; the gap is reported as ``ssq_["total"]`` minus the sum of the parts.

    Examples
    --------
    >>> model = ASCA(n_components=2).fit(X, design)            # doctest: +SKIP
    >>> model.ssq_percent_                                      # doctest: +SKIP
    >>> model.permutation_test(n_permutations=999, random_state=0)   # doctest: +SKIP
    >>> model.models_["A"].scores_                              # doctest: +SKIP

    References
    ----------
    Smilde, A. K., Jansen, J. J., Hoefsloot, H. C. J., Lamers, R.-J. A. N.,
    van der Greef, J., & Timmerman, M. E. (2005). ANOVA-simultaneous component
    analysis (ASCA): a new tool for analyzing designed metabolomics data.
    *Bioinformatics*, 21(13), 3043-3048.

    Zwanenburg, G., Hoefsloot, H. C. J., Westerhuis, J. A., Jansen, J. J., &
    Smilde, A. K. (2011). ANOVA-principal component analysis and
    ANOVA-simultaneous component analysis: a comparison. *J. Chemometrics*,
    25(10), 561-567.

    Camacho, J., Vitale, R., Morales-Jimenez, D., & Gomez-Llorente, C. (2022).
    Variable-selection ANOVA Simultaneous Component Analysis (VASCA).
    *Bioinformatics*, 38(1), 295-298.
    """

    # Bound like the PCA / PLS plot methods (ENG-05): a real method on the class, so
    # `help` and `inspect.signature` stay accurate and a subclass can override it.
    effect_summary_plot = _model_method(_effect_summary_plot)

    def __init__(
        self,
        n_components: int = 2,
        *,
        model: str = "interactions",
        add_residuals: bool = False,
        scale: bool = False,
    ):
        self.n_components = n_components
        self.model = model
        self.add_residuals = add_residuals
        self.scale = scale

    def fit(self, X: DataMatrix, design: pd.DataFrame, y: object = None) -> ASCA:  # noqa: ARG002
        """Decompose ``X`` by the design terms and fit a PCA to each term's effect.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The multivariate response. Must be complete: ASCA solves a least-squares
            problem over every column at once and has no missing-data path.
        design : pd.DataFrame of shape (n_samples, n_factors)
            One column per experimental factor, holding that factor's level for each
            row. Values may be strings or numbers; they are treated as categorical.
        y : object, optional
            Ignored, for sklearn API compatibility.

        Returns
        -------
        self : ASCA

        Raises
        ------
        ValueError
            If ``X`` and ``design`` disagree on the number of rows, if either holds
            missing values, or if a factor has only one level (it can carry no effect).
        """
        X, design = self._validated(X, design)
        self.feature_names_in_ = np.asarray(X.columns)
        self.grand_mean_ = X.mean(axis=0)
        centred = X - self.grand_mean_
        if self.scale:
            spread = X.std(axis=0, ddof=1)
            spread[spread < epsqrt] = 1.0
            self.column_scale_ = spread
            centred = centred / spread
        else:
            self.column_scale_ = pd.Series(1.0, index=X.columns)
        self._centred = centred

        rhs = _sum_coded_rhs(list(design.columns), self.model)
        design_matrix = dmatrix(rhs, design, return_type="dataframe")
        slices = design_matrix.design_info.term_slices
        self.is_balanced_ = self._check_balance(design)

        design_array = design_matrix.to_numpy()
        coefficients, *_ = np.linalg.lstsq(design_array, centred.to_numpy(), rcond=None)
        # Kept for the permutation tests, which refit this same design against shuffled
        # responses; rebuilding it per permutation would dominate their runtime.
        self._design_array = design_array
        self._term_columns: dict[str, slice] = {}

        self.terms_ = []
        self.effect_matrices_ = {}
        term_ranks: dict[str, int] = {}
        explained = np.zeros_like(centred.to_numpy())
        for term, columns in slices.items():
            name = _pretty(term.name())
            if name == "Intercept":
                # X is already centred, so the intercept is numerically zero. It stays in
                # the design matrix because dropping it would change the coding, but it
                # is not a term anyone wants a PCA of.
                continue
            self._term_columns[name] = columns
            block = design_array[:, columns]
            effect = block @ coefficients[columns, :]
            explained += effect
            self.terms_.append(name)
            self.effect_matrices_[name] = pd.DataFrame(effect, index=X.index, columns=X.columns)
            term_ranks[name] = int(np.linalg.matrix_rank(block))

        self.residuals_ = centred - pd.DataFrame(explained, index=X.index, columns=X.columns)
        self._term_ranks = term_ranks

        self._summarise_sums_of_squares(float((centred.to_numpy() ** 2).sum()))
        self.models_ = {name: self._fit_term_pca(name) for name in self.terms_}
        self.is_fitted_ = True
        return self

    @staticmethod
    def _validated(X: DataMatrix, design: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Coerce both inputs to frames and reject what the decomposition cannot use."""
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if not isinstance(design, pd.DataFrame):
            design = pd.DataFrame(design)
        if X.shape[0] != design.shape[0]:
            msg = f"X has {X.shape[0]} rows but design has {design.shape[0]}."
            raise ValueError(msg)
        if X.isna().to_numpy().any():
            msg = "ASCA needs a complete X: it solves one least-squares problem over every column."
            raise ValueError(msg)
        if design.isna().to_numpy().any():
            msg = "The design frame holds missing values; every row must carry a level for every factor."
            raise ValueError(msg)
        # `is_unique`/`nunique` on the column itself is the readable test; ruff prefers the
        # comparison form, and a factor with one level genuinely cannot carry an effect.
        single_level = [name for name in design.columns if (design[name] == design[name].iloc[0]).all()]
        if single_level:
            msg = (
                f"These factors have a single level and can carry no effect: {single_level}. Drop them from the design."
            )
            raise ValueError(msg)
        return X, design

    @staticmethod
    def _check_balance(design: pd.DataFrame) -> bool:
        """Report whether every observed combination of factor levels occurs equally often."""
        counts = design.groupby(list(design.columns), observed=True).size().to_numpy()
        return bool(counts.size and (counts == counts[0]).all())

    def _summarise_sums_of_squares(self, total: float) -> None:
        """Build `ssq_` / `ssq_percent_`, and say so when the design does not partition."""
        parts = {name: float((matrix.to_numpy() ** 2).sum()) for name, matrix in self.effect_matrices_.items()}
        parts[RESIDUAL] = float((self.residuals_.to_numpy() ** 2).sum())
        parts["total"] = total
        self.ssq_ = pd.Series(parts, name="sum of squares")
        self.ssq_percent_ = (self.ssq_ / total * 100.0).rename("percent of total")
        if self.is_balanced_:
            return
        accounted = sum(parts[name] for name in [*self.terms_, RESIDUAL])
        warnings.warn(
            f"The design is unbalanced, so the terms are not orthogonal and the sums of squares do not "
            f"partition the total exactly: the parts add to {accounted:.6g} against a total of {total:.6g}. "
            "Every term is still fitted simultaneously by least squares, which is the usual ASCA treatment, "
            "but a term's share depends on how the shared variation is attributed (the Type I / II / III "
            "question). Read the percentages as indicative rather than as a partition.",
            UserWarning,
            stacklevel=3,
        )

    def _fit_term_pca(self, term: str) -> PCA:
        """PCA of one term's effect matrix, with the component count capped at its rank.

        An effect matrix for a two-level factor has rank 1: there is one direction in
        which the two levels differ and nothing else. Asking `PCA` for more components
        than that raises, so the cap is applied here rather than left to the caller.
        """
        block = self.effect_matrices_[term]
        if self.add_residuals:
            block = block + self.residuals_
        # With residuals added the matrix is full rank again, so only the pure effect
        # matrix needs its rank ceiling.
        ceiling = block.shape[1] if self.add_residuals else max(1, self._term_ranks[term])
        return PCA(n_components=min(self.n_components, ceiling)).fit(block)

    def permutation_test(
        self,
        *,
        n_permutations: int = 999,
        random_state: int | np.random.Generator | None = None,
    ) -> pd.Series:
        """Test each term's effect against the null of exchangeable rows.

        For every term the rows of the response are permuted, the decomposition is
        refitted, and the term's sum of squares is recomputed. A term whose observed sum
        of squares sits in the upper tail of that null is carrying more variation than
        the design's shape alone would produce.

        Parameters
        ----------
        n_permutations : int, optional
            Number of permutations. Default 999, which puts the smallest attainable
            p-value at ``1 / 1000``.
        random_state : int, np.random.Generator, or None, optional
            Seeds the permutations, per the reproducibility contract.

        Returns
        -------
        pvalues : pd.Series
            One p-value per term, also stored as :attr:`pvalues_`. Each is
            ``(1 + #{null >= observed}) / (1 + n_permutations)``: the observed statistic
            counts itself among the permutations, because no finite set of shuffles
            licenses a claim of exactly zero. The same convention is used by the Van der
            Voet test, the multiblock randomization test and `PLSDA.permutation_test`.

        Notes
        -----
        One permutation refits every term at once, so the whole test costs
        ``n_permutations`` least-squares solves rather than one per term. The PCA step is
        not repeated: the statistic is the sum of squares of the effect matrix, which the
        decomposition produces directly.
        """
        rng = check_random_state(random_state)
        design_array = self._design_array
        pvalues = {}
        for name in self.terms_:
            observed = float(self.ssq_[name])
            reduced = (self.effect_matrices_[name] + self.residuals_).to_numpy()
            null = self._null_ssq(reduced, self._term_columns[name], int(n_permutations), rng, design_array)
            pvalues[name] = (1 + int(np.sum(null >= observed))) / (1 + int(n_permutations))
        self.pvalues_ = pd.Series(pvalues, name="p-value")
        return self.pvalues_

    @staticmethod
    def _null_ssq(
        reduced: np.ndarray,
        columns: slice,
        n_permutations: int,
        rng: np.random.Generator,
        design_array: np.ndarray,
    ) -> np.ndarray:
        """Null sums of squares for one term, by permuting the reduced-model response.

        The obvious null, permuting the rows of the whole response, is wrong when another
        term is large: it leaves that term's variation in the data, so the term under test
        inherits a share of it and its null is far too high. On a two-factor fixture where
        A carries 84 percent of the variation and B a real 11 percent, the whole-response
        null put B at p = 0.13 and hid a genuine effect.

        Permuting the *reduced* response instead, this term's own effect plus the residual
        and nothing else, is the Freedman-Lane / ter Braak construction: the exchangeable
        units are the rows once the other terms have been removed. The same fixture then
        puts B where it belongs.
        """
        null = np.empty(n_permutations)
        rows = np.arange(reduced.shape[0])
        block = design_array[:, columns]
        for index in range(n_permutations):
            shuffled = reduced[rng.permutation(rows), :]
            coefficients, *_ = np.linalg.lstsq(design_array, shuffled, rcond=None)
            effect = block @ coefficients[columns, :]
            null[index] = float((effect**2).sum())
        return null

    def vasca(
        self,
        term: str,
        *,
        n_permutations: int = 999,
        alpha: float = 0.05,
        random_state: int | np.random.Generator | None = None,
    ) -> Bunch:
        """Variable-selection ASCA: which variables carry this term's effect.

        The ASCA permutation test asks one question of the whole response matrix, so an
        effect that lives in three variables out of two hundred is diluted by the other
        hundred and ninety-seven and can fail to register at all. VASCA ranks the
        variables by their contribution to the term, then tests each nested subset of the
        top-ranked ones. A subset that contains the effect and little else gives a far
        smaller p-value than the whole matrix does.

        Parameters
        ----------
        term : str
            Which design term to examine; one of :attr:`terms_`.
        n_permutations : int, optional
            Permutations used to build the null for every subset at once. Default 999.
        alpha : float, optional
            Target false-discovery rate for the Benjamini-Hochberg step. Default 0.05.
        random_state : int, np.random.Generator, or None, optional
            Seeds the permutations.

        Returns
        -------
        result : sklearn.utils.Bunch
            ``table`` (pd.DataFrame, one row per subset size: the variable added at that
            step, the subset's cumulative sum of squares, how many standard deviations it
            sits above its own null, and its raw and FDR-corrected p-values),
            ``selected`` (list of variable names), ``ranking`` (the variables in
            contribution order) and ``p_value`` (the smallest corrected p-value found).

            ``selected`` is the subset that clears ``alpha`` and stands furthest above its
            own null. The second half of that matters: with a few hundred permutations the
            smallest attainable p-value is ``1 / (1 + n_permutations)`` and many subset
            sizes reach it at once, so choosing by p-value alone would return every
            variable that happened to tie at the floor. The z-score does not tie, and it
            peaks where the effect is concentrated.

            ``selected`` is empty when no subset clears ``alpha``, which is the honest
            answer for a term that carries nothing.

        Raises
        ------
        ValueError
            If ``term`` is not one of the fitted terms.

        Notes
        -----
        The permutations are shared across subset sizes: one shuffle produces a
        per-variable sum of squares vector, and every subset's null statistic is a partial
        sum of it. The whole walk therefore costs the same ``n_permutations`` solves as
        the single-term test, rather than one run per subset size.

        Because one test is made per subset size, the raw p-values are corrected across
        those tests with :func:`~process_improve.univariate.metrics.benjamini_hochberg`,
        which controls the false-discovery rate rather than the family-wise error rate.
        """
        if term not in self.terms_:
            msg = f"{term!r} is not a fitted term; choose one of {self.terms_}."
            raise ValueError(msg)

        rng = check_random_state(random_state)
        per_variable = (self.effect_matrices_[term].to_numpy() ** 2).sum(axis=0)
        order = np.argsort(per_variable)[::-1]
        names = [str(self.feature_names_in_[index]) for index in order]
        observed = np.cumsum(per_variable[order])

        # Same reduced-model null as the per-term test: this term's effect plus the
        # residual, nothing else, so a large neighbouring term cannot inflate it.
        reduced = (self.effect_matrices_[term] + self.residuals_).to_numpy()
        design_array = self._design_array
        columns = self._term_columns[term]
        block = design_array[:, columns]
        rows = np.arange(reduced.shape[0])

        draws = np.empty((int(n_permutations), observed.size))
        for index in range(int(n_permutations)):
            shuffled = reduced[rng.permutation(rows), :]
            coefficients, *_ = np.linalg.lstsq(design_array, shuffled, rcond=None)
            effect = block @ coefficients[columns, :]
            # The null subset is the top-m of *this* permutation, not of the observed
            # ranking: taking the observed ordering would let the null inherit it and
            # rig the comparison in the observed statistic's favour.
            draws[index, :] = np.cumsum(np.sort((effect**2).sum(axis=0))[::-1])

        raw = (1 + (draws >= observed).sum(axis=0)) / (1 + int(n_permutations))
        corrected = np.asarray(benjamini_hochberg(raw, alpha=alpha).p_adjusted, dtype=float)
        # How far above its own null each subset sits. With a few hundred permutations the
        # smallest attainable p-value is reached by many subset sizes at once, so the
        # p-value alone cannot say which subset is best; this can, and it is the quantity
        # the p-value is a coarse rounding of.
        spread = draws.std(axis=0, ddof=1)
        z_score = np.divide(observed - draws.mean(axis=0), spread, out=np.zeros_like(observed), where=spread > epsqrt)

        passing = np.flatnonzero(corrected <= alpha)
        if passing.size:
            # Among the subsets that clear alpha, take the one standing furthest above its
            # own null. On ties at the p-value floor that is what separates "the effect is
            # in these two variables" from "the effect is in these two plus six passengers".
            best = int(passing[np.argmax(z_score[passing])])
            selected = names[: best + 1]
        else:
            selected = []

        table = pd.DataFrame(
            {
                "variable": names,
                "ssq_cumulative": observed,
                "z_score": z_score,
                "p_value": raw,
                "p_value_fdr": corrected,
            },
            index=pd.RangeIndex(1, observed.size + 1, name="n_variables"),
        )
        return Bunch(
            table=table,
            selected=selected,
            ranking=names,
            p_value=float(np.min(corrected)),
        )
