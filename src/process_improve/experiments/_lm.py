# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.

from __future__ import annotations

import ast
import re
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import ModelDesc
from statsmodels.regression.linear_model import OLS

if TYPE_CHECKING:
    from process_improve.experiments.structures import Expt


class UnsafeFormulaError(ValueError):
    """Raised when a model formula contains tokens outside the safe Wilkinson subset.

    Patsy and statsmodels evaluate each formula term as a Python expression, so a
    formula coming from an untrusted source (for example the ``fit_linear_model``
    MCP tool) is a code-execution vector. :func:`validate_formula_is_safe` rejects
    anything that is not a plain Wilkinson formula over known data columns before it
    ever reaches patsy.
    """


# Characters permitted in a safe Wilkinson formula: column-name characters,
# whitespace, and the structural operators (~ + - * : ^) plus grouping parens.
# Notably excluded: quotes, '.', ',', '[', ']', '=', '!', '@', '%', backslash -
# i.e. everything needed to build a Python expression, attribute access, string
# literal, or function-call argument list.
_FORMULA_ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9_ \t\r\n~+\-*:^()]*$")
_FORMULA_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")

# Patsy transform helpers that wrap an arithmetic expression without naming a
# data column: ``I(...)`` (identity / "as-is") and ``Q(...)`` (quote a name).
_TRANSFORM_FUNCS = frozenset({"I", "Q"})

# Patsy categorical-contrast helpers. These are pure (they only re-code a
# categorical column into contrast columns), so they are safe to allow inside a
# formula. ``C(col)`` and ``C(col, Sum)`` / ``Treatment`` / ``Poly`` etc. let a
# caller specify explicit contrasts for a categorical factor. Their arguments
# are restricted (see ``_check_categorical_arg``) to data columns, other
# contrast helpers, and literals; never arbitrary calls.
_CATEGORICAL_FUNCS = frozenset({"C", "Treatment", "Sum", "Diff", "Helmert", "Poly"})

# Curated allowlist of numpy callables permitted inside a formula when
# ``allow_numpy=True``. These are pure, element-wise math transforms. We do NOT
# allow arbitrary ``np.<anything>`` because numpy also exposes dangerous I/O such
# as ``np.load`` (which deserialises pickles) and ``np.fromfile``.
_NUMPY_ALLOWED_FUNCS = frozenset(
    {
        "log",
        "log10",
        "log2",
        "log1p",
        "exp",
        "expm1",
        "sqrt",
        "cbrt",
        "square",
        "power",
        "reciprocal",
        "sign",
        "abs",
        "absolute",
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
    }
)


def validate_identifier_is_safe(name: object) -> None:
    """Reject a column / response name that is not a plain Python identifier.

    User-supplied names (``design_matrix`` dict keys, ``response_column``) are
    interpolated into a patsy formula, so a name such as ``"A); import os; ("``
    is an injection vector. We require a bare identifier and forbid dunders.

    Parameters
    ----------
    name:
        The candidate column or response name.

    Raises
    ------
    UnsafeFormulaError
        If *name* is not a string, contains ``__``, or is not a plain identifier.
    """
    if not isinstance(name, str):
        raise UnsafeFormulaError(f"name must be a string, got {type(name).__name__}.")
    if "__" in name:
        raise UnsafeFormulaError(f"name {name!r} may not contain '__' (dunder access is forbidden).")
    if not _IDENTIFIER_RE.match(name):
        raise UnsafeFormulaError(
            f"name {name!r} is not a plain identifier; only letters, digits and '_' are allowed "
            f"(and it may not start with a digit)."
        )


def validate_formula_is_safe(
    formula: str,
    allowed_names: Iterable[str],
    *,
    allow_transforms: bool = False,
    allow_numpy: bool = False,
) -> None:
    """Reject a model ``formula`` that is not a safe Wilkinson formula over *allowed_names*.

    This is the guard for untrusted callers (e.g. the ``fit_linear_model`` tool).
    Patsy evaluates every formula term as a Python expression with builtins and
    numpy in scope, so a string such as ``y ~ I(__import__('os').system('id'))``
    would execute arbitrary code.

    By default only a plain Wilkinson formula is allowed:

    * identifiers that name an actual data column,
    * the operators ``~ + - * : ^`` and grouping parentheses,
    * integer literals (for powers like ``(A + B)**2``) and whitespace.

    Any quote, dot, comma, dunder, or unknown identifier (``np``, ``I``,
    ``__import__``, ...) is rejected.

    The optional flags relax this for trusted-but-still-validated callers. They
    switch on an AST-based check that admits a curated set of transforms while
    still rejecting attribute access, string literals, dunders, and any call
    other than the allowlisted ones:

    * ``allow_transforms`` - permit ``I(...)`` / ``Q(...)`` wrapping arithmetic
      over data columns (e.g. the ``quadratic`` shorthand's ``I(A ** 2)``).
    * ``allow_numpy`` - additionally permit a curated allowlist of element-wise
      numpy calls such as ``np.log(A)`` or ``np.power(A, 2)``.

    Parameters
    ----------
    formula:
        The model formula in Wilkinson notation, e.g. ``"y ~ A*B"``.
    allowed_names:
        The legal identifier names, i.e. the columns present in the data.
    allow_transforms:
        If true, permit ``I(...)`` / ``Q(...)`` transforms of column arithmetic.
    allow_numpy:
        If true, permit a curated allowlist of element-wise ``np.<func>`` calls.

    Raises
    ------
    UnsafeFormulaError
        If *formula* is not a string, contains a ``__`` dunder, or references a
        token / construct outside the permitted subset.
    """
    if not isinstance(formula, str):
        raise UnsafeFormulaError(f"formula must be a string, got {type(formula).__name__}.")
    if "__" in formula:
        raise UnsafeFormulaError("formula may not contain '__' (dunder access is forbidden).")

    allowed = {str(name) for name in allowed_names}

    if not allow_transforms and not allow_numpy:
        # Strict plain-Wilkinson path (unchanged behaviour).
        if not _FORMULA_ALLOWED_CHARS.match(formula):
            forbidden = sorted({c for c in formula if not _FORMULA_ALLOWED_CHARS.match(c)})
            raise UnsafeFormulaError(f"formula contains forbidden characters: {forbidden}.")
        unknown = sorted({tok for tok in _FORMULA_IDENTIFIER.findall(formula) if tok not in allowed})
        if unknown:
            raise UnsafeFormulaError(
                f"formula references unknown name(s) {unknown}; only data columns are allowed: "
                f"{sorted(allowed)}."
            )
        return

    _validate_formula_ast(formula, allowed, allow_numpy=allow_numpy)


def _validate_formula_ast(formula: str, allowed: set[str], *, allow_numpy: bool) -> None:
    """AST-based validation of a formula that may contain ``I()``/``np`` transforms.

    Each side of the ``~`` is parsed as a Python expression and walked against a
    strict node allowlist. The interaction operator ``:`` is rewritten to ``*``
    so the side parses (the two are structurally equivalent for our purposes).
    """
    sides = formula.split("~")
    if len(sides) > 2:
        raise UnsafeFormulaError("formula may contain at most one '~'.")

    for side in sides:
        # ``:`` (interaction) and ``^`` are patsy structural operators; map ``:``
        # to ``*`` so the side is parseable Python. ``^`` already parses (BitXor).
        expr = side.replace(":", "*").strip()
        if not expr:
            continue
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise UnsafeFormulaError(f"formula side {side.strip()!r} is not a valid expression.") from exc
        _check_formula_node(tree.body, allowed, allow_numpy=allow_numpy)


def _check_formula_node(node: ast.AST, allowed: set[str], *, allow_numpy: bool) -> None:
    """Recursively validate a single AST node from a formula expression."""
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.BitXor)):
            raise UnsafeFormulaError(f"operator {type(node.op).__name__} is not allowed in a formula.")
        _check_formula_node(node.left, allowed, allow_numpy=allow_numpy)
        _check_formula_node(node.right, allowed, allow_numpy=allow_numpy)
        return

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise UnsafeFormulaError(f"unary operator {type(node.op).__name__} is not allowed in a formula.")
        _check_formula_node(node.operand, allowed, allow_numpy=allow_numpy)
        return

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeFormulaError(f"literal {node.value!r} is not allowed in a formula.")
        return

    if isinstance(node, ast.Name):
        if node.id not in allowed:
            raise UnsafeFormulaError(
                f"formula references unknown name(s) [{node.id!r}]; only data columns are allowed: "
                f"{sorted(allowed)}."
            )
        return

    if isinstance(node, ast.Call):
        _check_formula_call(node, allowed, allow_numpy=allow_numpy)
        return

    raise UnsafeFormulaError(f"construct {type(node).__name__} is not allowed in a formula.")


def _check_formula_call(node: ast.Call, allowed: set[str], *, allow_numpy: bool) -> None:
    """Validate a call node: ``I()``/``Q()``, a categorical contrast, or ``np.<func>``."""
    func = node.func

    # Categorical-contrast helpers (C, Treatment, Sum, ...) have their own arg
    # rules (bare contrast names, literals, keyword reference levels).
    if isinstance(func, ast.Name) and func.id in _CATEGORICAL_FUNCS:
        _check_categorical_call(node, allowed, allow_numpy=allow_numpy)
        return

    if node.keywords:
        raise UnsafeFormulaError("keyword arguments are not allowed in a formula call.")

    if isinstance(func, ast.Name) and func.id in _TRANSFORM_FUNCS:
        pass
    elif allow_numpy and isinstance(func, ast.Attribute):
        base = func.value
        if not (isinstance(base, ast.Name) and base.id == "np"):
            raise UnsafeFormulaError("only 'np.<func>' attribute calls are allowed in a formula.")
        if func.attr not in _NUMPY_ALLOWED_FUNCS:
            raise UnsafeFormulaError(
                f"numpy function 'np.{func.attr}' is not in the allowed set {sorted(_NUMPY_ALLOWED_FUNCS)}."
            )
    else:
        raise UnsafeFormulaError(
            "only I()/Q(), categorical contrasts C()/Treatment()/Sum()/Diff()/Helmert()/Poly() "
            "(and, when enabled, np.<func>()) calls are allowed in a formula."
        )

    for arg in node.args:
        if isinstance(arg, ast.Starred):
            raise UnsafeFormulaError("starred arguments are not allowed in a formula call.")
        _check_formula_node(arg, allowed, allow_numpy=allow_numpy)


def _check_categorical_call(node: ast.Call, allowed: set[str], *, allow_numpy: bool) -> None:
    """Validate a categorical-contrast call: ``C()``/``Treatment()``/``Sum()``/..."""
    for arg in node.args:
        if isinstance(arg, ast.Starred):
            raise UnsafeFormulaError("starred arguments are not allowed in a formula call.")
        _check_categorical_arg(arg, allowed, allow_numpy=allow_numpy)
    for kw in node.keywords:
        if kw.arg is None:
            raise UnsafeFormulaError("**kwargs are not allowed in a formula call.")
        _check_literal_or_container(kw.value)


def _check_categorical_arg(node: ast.AST, allowed: set[str], *, allow_numpy: bool) -> None:
    """Validate one argument of a categorical-contrast call (``C``/``Treatment``/...).

    Permitted: a data-column name, a bare contrast helper name (``Sum``), a
    nested contrast call (``Treatment(reference="A")``), or a literal / list of
    literals (reference levels, polynomial degree).
    """
    # A bare contrast-helper name used as the contrast argument, e.g. C(f, Sum).
    if isinstance(node, ast.Name) and node.id in _CATEGORICAL_FUNCS:
        return
    # A nested contrast call, e.g. C(f, Treatment(reference="A")).
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _CATEGORICAL_FUNCS:
        _check_formula_call(node, allowed, allow_numpy=allow_numpy)
        return
    # A literal reference level / degree, or a list/tuple of them.
    if isinstance(node, (ast.Constant, ast.List, ast.Tuple)):
        _check_literal_or_container(node)
        return
    # Otherwise it must be an ordinary allowed node (a data column, I()/Q(), ...).
    _check_formula_node(node, allowed, allow_numpy=allow_numpy)


def _check_literal_or_container(node: ast.AST) -> None:
    """Allow only string / numeric literals, or lists / tuples of them."""
    if isinstance(node, (ast.List, ast.Tuple)):
        for element in node.elts:
            _check_literal_or_container(element)
        return
    if isinstance(node, ast.Constant):
        value = node.value
        if not isinstance(value, bool) and isinstance(value, (str, int, float)):
            return
    raise UnsafeFormulaError("categorical-contrast arguments must be column names, contrast helpers, or literals.")


def forg(x: float, prec: int = 3) -> str:
    """Yanked from the code for Statsmodels / iolib / summary.py and adjusted.

    Formats ``x`` with ``prec`` significant/decimal digits, switching to the
    ``g`` format for very large or very small magnitudes. Any positive ``prec``
    is supported; ``prec=3`` and ``prec=4`` reproduce the original widths.
    """
    width = prec + 6
    if (abs(x) >= 1e4) or (abs(x) < 1e-4):
        return f"{x:{width}.{prec}g}"
    return f"{x:{width}.{prec}f}"


class Model(OLS):
    """Just a thin wrapper around the OLS class from Statsmodels."""

    # Declared for static typing. ``data`` starts as ``None`` and is replaced by
    # the fitted :class:`~process_improve.experiments.structures.Expt` in ``lm()``.
    data: Expt | None
    aliasing: dict | None
    name: str | None

    def __init__(
        self,
        OLS_instance: Any,  # noqa: ANN401
        model_spec: str,
        aliasing: dict | None = None,
        name: str | None = None,
    ) -> None:
        self._OLS = OLS_instance
        self._model_spec = model_spec
        self.name = name

        # Standard error
        self.df_resid = self._OLS.df_resid
        self.df_model = self._OLS.df_model
        self.nobs = self._OLS.nobs
        # Leads to errors for size inconsistency if the data frames have
        # missing data?
        # self.rsquared = self.R2 = self._OLS.rsquared
        self.residuals = self._OLS.resid

        # Will be replaced by the "lm()" function
        self.data = None
        self.aliasing = aliasing

    def __str__(self) -> str:
        """Return the model specification as a string."""
        spec = ModelDesc.from_formula(self._model_spec)
        return spec.describe()

    def summary(self, alpha: float = 0.05, print_to_screen: bool = True) -> Any:  # noqa: ARG002, ANN401
        """Build the OLS summary table for this model and return it.

        The returned object is the statsmodels summary instance, with the
        underlying ``self._OLS.summary()`` adjusted to label the residual
        standard error row. The method does NOT print anything by itself;
        the top-level :func:`summary` wrapper handles screen output via its
        own ``show`` flag. The ``alpha`` and ``print_to_screen`` arguments
        are unused and kept for backwards compatibility.
        """
        # Taken from statsmodels.regression.linear_model.py
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            main = "OLS Regression Results"
            if self.name:
                main += ": " + str(self.name)
            elif self.data is not None and self.data.pi_title:
                main += ": " + str(self.data.pi_title)

            smry = self._OLS.summary(title=main)
            # print(smry)
            # Call this directly and modify the result to suppress what we
            # don't really care to show:

            smry.tables[0].pop(8)

            se = "---"
            if not (np.isinf(self._OLS.scale)):
                se = f"{np.sqrt(self._OLS.scale):.3f}"

            # Residual standard error
            smry.tables[0][7][0].data = "Residual std error"
            smry.tables[0][7][1].data = se
            # smry.tables[0][7][0].data = se
            # smry.tables[0][7][1].data = se

        return smry

    def get_parameters(self, drop_intercept: bool = True) -> pd.DataFrame:
        """Get the parameter values; return them in a Pandas dataframe."""

        params = self._OLS.params.copy()
        try:
            if drop_intercept:
                params = params.drop("Intercept")
        except KeyError:
            # Some models (e.g. ``y ~ 0 + ...``) have no Intercept term; the
            # drop is a no-op in that case.
            pass

        return params.dropna()

    def get_factor_names(self, level: int = 1) -> list[str]:
        """
        Get the factors in a model which correspond to a certain level.

        1 : pure factors
        2 : 2-factor interactions and quadratic terms
        3 : 3-factor interactions and cubic terms
        4 : etc
        """
        spec = ModelDesc.from_formula(self._model_spec)
        return [term.name() for term in spec.rhs_termlist if len(term.factors) == level]

    def get_response_name(self) -> str:
        """Get the name of the response variable from the model specification."""
        spec = ModelDesc.from_formula(self._model_spec)
        return spec.lhs_termlist[0].name()

    def get_title(self) -> str:
        """Get the model's title, if it has one. Always returns a string."""
        if self.data is None:
            return ""
        return self.data.get_title()

    def get_aliases(
        self,
        aliasing_up_to_level: int = 2,
        drop_intercept: bool | None = True,
        websafe: bool | None = False,
    ) -> list:
        """
        Return a list, containing strings, representing the aliases
        of the fitted effects.

        aliasing_up_to_level: up to which level of interactions shown

        drop_intercept: default is True, but sometimes it is interesting to
            know which effects are aliased with the intercept

        websafe: default is False; if True, will print the first term
            in the aliasing in bold, since that is the nominally estimated
            effect.
        """
        alias_strings: list[Any] = []
        if self.aliasing is None or len(self.aliasing.keys()) == 0:
            return alias_strings

        params = self.get_parameters(drop_intercept=bool(drop_intercept))
        for p_name in params.index.values:
            aliasing = f'<span style="font-size: 130%; font-weight: 700">{p_name}</span>' if websafe else p_name
            suffix = ""
            for alias in self.aliasing[(p_name,)]:
                # Subtract "-1" because the first list entry tracks the sign
                if (len(alias) - 1) <= aliasing_up_to_level:
                    aliasing += f" {alias[0]} {':'.join(alias[1:])}"
                if (len(alias) - 1) > aliasing_up_to_level:
                    suffix = r" + <i>higher interactions</i>" if websafe else " + higher interactions"

            # Finished with this parameter
            alias_strings.append(aliasing + suffix)

        # All done
        return alias_strings


# Model.__repr__ = Model.__str__


def predict(model: Model, **kwargs: Any) -> Any:  # noqa: ANN401
    """Make predictions from the model."""
    return model._OLS.predict(exog=dict(kwargs))


def lm(  # noqa: C901, PLR0915
    model_spec: str,
    data: Expt,
    name: str | None = None,
    alias_threshold: float | None = 0.995,
) -> Model:
    """Create a linear model."""

    def find_aliases(  # noqa: C901, PLR0912
        model: Any,  # noqa: ANN401
        model_desc: ModelDesc,
        threshold_correlation: float = 0.995,
    ) -> tuple[dict, list]:
        """
        Find columns which are exactly correlated, or up to at least a level
        of `threshold_correlation`.
        Return a dictionary of aliasing and a list of columns to keep.

        The columns to keep will be in the order checked. Perhaps this can be
        improved.
        For example if AB = CD, then return AB to keep.
        For example if A = BCD, then return A, and not the BCD column to keep.
        """
        has_variation = model.exog.std(axis=0) > np.sqrt(np.finfo(float).eps)

        # Snippet of code here is from the NumPy "corrcoef" function. Adapted.
        c = np.cov(model.exog.T, None, rowvar=True)
        dot_product = model.exog.T @ model.exog
        try:
            d = np.diag(c)
        except ValueError:
            # scalar covariance
            # nan if incorrect value (nan, inf, 0), 1 otherwise
            return c / c  # type: ignore[return-value]  # degenerate scalar-covariance fallback; preserves original runtime behaviour
        stddev = np.sqrt(d.real)

        aliasing = defaultdict(list)
        terms = model_desc.rhs_termlist
        drop_columns: list[int] = []
        counter = -1
        corrcoef = c.copy()
        for idx, check in enumerate(has_variation):
            if check:
                counter += 1

                for j, stddev_value in enumerate(stddev):
                    if stddev_value == 0:
                        pass
                    else:
                        corrcoef[idx, j] = c[idx, j] / stddev[idx] / stddev_value

                # corrcoef = c / stddev[idx, None]
                # corrcoef = corrcoef / stddev[None, idx]

                candidates = [i for i, val in enumerate(np.abs(corrcoef[idx, :])) if (val > threshold_correlation)]
            else:
                # Columns with no variation
                candidates = [i for i, j in enumerate(has_variation) if (j <= threshold_correlation)]

            # Track the correlation signs (computed from the raw dot product so
            # the sign information matches the eventual alias decision below
            # regardless of which branch built ``candidates``).
            signs = [np.sign(j) for j in dot_product[idx, :]]

            # Now drop out the candidates with the longest word lengths
            alias_len = [(len(terms[i].factors), i) for i in candidates]
            alias_len.sort(reverse=True)
            drop_columns.extend(entry[1] for entry in alias_len[0:-1])

            for col in candidates:
                if col == idx:
                    # It is of course perfectly correlated with itself
                    pass
                else:
                    aliases = [t.name() for t in terms[col].factors]
                    if len(aliases) == 0:
                        aliases = ["Intercept"]

                    key = tuple([t.name() for t in terms[idx].factors])
                    if len(key) == 0:
                        key = ("Intercept",)

                    if signs[col] > 0:
                        aliases.insert(0, "+")
                    if signs[col] < 0:
                        aliases.insert(0, "-")
                    aliasing[key].append(aliases)

        # Sort the aliases in length:
        for key, val in aliasing.items():
            sorted_aliases = [(len(i), i) if i[1] != "Intercept" else (1e5, i) for i in val]
            sorted_aliases.sort()
            aliasing[key] = [i[1] for i in sorted_aliases]

        return aliasing, list(set(drop_columns))

    # Patsy evaluates each formula term as a Python expression, so an untrusted
    # ``model_spec`` is a code-execution vector. Allow only a safe Wilkinson
    # formula over the data columns, optionally with I()/Q() and a curated set of
    # element-wise numpy transforms (the public textbook API relies on these).
    validate_formula_is_safe(model_spec, data.columns, allow_transforms=True, allow_numpy=True)

    pre_model = smf.ols(model_spec, data=data)
    # SEC-19 (#268): a formula like ``y ~ (A+B+C+D+E)**5`` expands to
    # 2**5 terms; combined with a wide ``data`` this is a CPU sink.
    # Cap the expanded term count after patsy parses the RHS.
    from process_improve.config import settings  # noqa: PLC0415

    n_terms = len(pre_model.data.xnames)
    if n_terms > settings.max_formula_terms:
        raise ValueError(
            f"formula {model_spec!r} expanded to {n_terms} terms; "
            f"the SEC-19 cap is settings.max_formula_terms="
            f"{settings.max_formula_terms}."
        )
    model_description = ModelDesc.from_formula(model_spec)
    # ``alias_threshold`` is ``float | None`` at the public boundary; the inner
    # ``find_aliases`` uses it in numeric comparisons. The cast is a no-op at
    # runtime (preserving the original behaviour for any value, including None).
    aliasing, drop_columns = find_aliases(
        pre_model, model_description, threshold_correlation=cast("float", alias_threshold)
    )
    drop_column_names = [pre_model.data.xnames[i] for i in drop_columns]

    post_model = smf.ols(model_spec, data=data, drop_cols=drop_column_names)

    name = name or data.pi_title
    out = Model(
        OLS_instance=post_model.fit(),
        model_spec=model_spec,
        aliasing=aliasing,
        name=name,
    )
    out.data = data

    return out


def summary(
    model: Model,
    show: bool | None = True,
    aliasing_up_to_level: int = 3,
) -> Any:  # noqa: ANN401
    """
    Print a summary to the screen of the model.

    Appends, if there is any aliasing, a summary of those aliases,
    up to the (integer) level of interaction: `aliasing_up_to_level`.
    """
    out = model.summary()
    extra = []
    aliases = model.get_aliases(aliasing_up_to_level, drop_intercept=False)
    values = model.get_parameters(drop_intercept=False).values
    if len(aliases):
        extra.append("Aliasing pattern")
        for value, alias in zip(values, aliases, strict=False):
            extra.append(f" {forg(float(value), 4)} = {alias}")

    out.add_extra_txt(extra)
    if show:
        print(out)  # noqa: T201
    return out



# ENG-23 (#305): explicit ``__all__`` so the thin re-exporter ``models.py``
# can do ``from ._lm import *`` without triggering CodeQL's
# py/polluting-import warning.
__all__ = [
    "Model",
    "UnsafeFormulaError",
    "forg",
    "lm",
    "predict",
    "summary",
    "validate_formula_is_safe",
    "validate_identifier_is_safe",
]
