# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.

from __future__ import annotations

import ast
import itertools
import keyword
import re
import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

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
        raise UnsafeFormulaError(f"name must be a string, got {type(name).__name__}. Try {suggest_identifier(name)!r}.")
    if "__" in name:
        raise UnsafeFormulaError(
            f"name {name!r} may not contain '__' (dunder access is forbidden). Try {suggest_identifier(name)!r}."
        )
    if not _IDENTIFIER_RE.match(name):
        raise UnsafeFormulaError(
            f"name {name!r} is not a plain identifier; only letters, digits and '_' are allowed "
            f"(and it may not start with a digit). Try {suggest_identifier(name)!r}."
        )


def validate_identifiers_are_safe(names: Iterable[object]) -> None:
    """Apply :func:`validate_identifier_is_safe` to each name, and refuse all the unsafe ones in one error.

    The error maps each refused name to a :func:`suggest_identifier` suggestion. A suggestion that another
    name already has gains ``_2``, ``_3``, ..., so the map can go straight into ``DataFrame.rename``.

    Parameters
    ----------
    names : Iterable[object]
        The candidate column or factor names.

    Raises
    ------
    UnsafeFormulaError
        If any name is not a plain identifier.
    """
    names = list(names)
    refused = []
    for name in names:
        try:
            validate_identifier_is_safe(name)
        except UnsafeFormulaError:
            refused.append(name)
    if not refused:
        return
    taken = {str(name) for name in names}
    renames = {}
    for name in refused:
        renames[name] = _unused_name(suggest_identifier(name), taken)
        taken.add(renames[name])
    raise UnsafeFormulaError(
        f"{refused!r} cannot go in a model formula: a name there is letters, digits and '_', does not start "
        f"with a digit and does not hold '__'. Rename them, for example with .rename(columns={renames!r})."
    )


def _unused_name(stem: str, taken: set[str]) -> str:
    """Return *stem*, or the first of ``stem_2``, ``stem_3``, ... that is not in *taken*."""
    new, count = stem, 1
    while new in taken:
        count += 1
        new = f"{stem.rstrip('_')}_{count}"
    return new


#: Symbols that column names often hold, as words, so that a suggested name keeps their meaning.
_SYMBOL_WORDS = {"%": "pct", "°": "deg", "#": "num"}


def suggest_identifier(name: object) -> str:
    """Suggest a name close to *name* that :func:`validate_identifier_is_safe` and Python both accept.

    ``%``, ``°`` and ``#`` become ``pct``, ``deg`` and ``num``. Each run of other characters that a name
    cannot hold becomes one ``_``, and an ``_`` at either end is removed. A name that then does not start
    with a letter gets an ``x`` in front, and a Python keyword gets an ``_`` after it.

    Parameters
    ----------
    name : object
        The refused name. A name that is not a string is read as its ``str``.

    Returns
    -------
    str
        The suggested name.

    Examples
    --------
    >>> [suggest_identifier(name) for name in ["Water %", "Temp (°C)", "2nd stage", "lambda", 0]]
    ['Water_pct', 'Temp_deg_C', 'x2nd_stage', 'lambda_', 'x0']
    """
    text = str(name)
    for symbol, word in _SYMBOL_WORDS.items():
        text = text.replace(symbol, f" {word} ")
    # The guard's \w takes "²", which Python does not, and Python takes "·", which \w does not: keep neither.
    kept = "".join(char if _IDENTIFIER_RE.match(f"x{char}") and f"x{char}".isidentifier() else " " for char in text)
    text = re.sub(r"[ _]+", "_", kept).strip("_")
    if not _IDENTIFIER_RE.match(text):  # empty, or it starts with a digit or a letter outside A-Z
        text = f"x{text}"
    return f"{text}_" if keyword.iskeyword(text) else text


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
        token / construct outside the permitted subset; or if a column has the name
        of a function the formula calls, such as ``I`` in ``I(P ** 2)``.
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
                f"formula references unknown name(s) {unknown}; only data columns are allowed: {sorted(allowed)}."
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
        _check_no_column_hides_a_call(tree, allowed)


def _called_names(tree: ast.AST) -> list[ast.Name]:
    """Return the names a parsed formula side calls: ``I`` in ``I(A ** 2)``, ``np``, and ``Sum`` in ``C(A, Sum)``."""
    called = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func.value if isinstance(node.func, ast.Attribute) else node.func  # np.log(A) calls np
        if isinstance(func, ast.Name):
            called.append(func)
        if isinstance(node.func, ast.Name) and node.func.id == "C":  # the contrast in C(A, Sum) is called too
            called += [arg for arg in node.args[1:] if isinstance(arg, ast.Name)]
    return called


def _check_no_column_hides_a_call(tree: ast.AST, columns: set[str]) -> None:
    """Refuse a column that has the name of a function the formula calls.

    Patsy looks a name up in the data before its own functions, so a column named ``I``
    turns every ``I(P ** 2)`` into a call of the column, and a column named ``C`` does the
    same to ``C(A, Sum)``. Each fails with "'Series' object is not callable". The library's
    own entry points fit such a column under a stand-in name (:func:`stand_ins_for`).
    """
    hidden = sorted({node.id for node in _called_names(tree)} & columns)
    if hidden:
        renames = {name: _unused_name(f"{name}_", columns) for name in hidden}
        raise UnsafeFormulaError(
            f"The formula calls {', '.join(f'{name}()' for name in hidden)}, and a column has the same name, "
            f"which patsy finds first and tries to call. Rename the column, for example with "
            f".rename(columns={renames!r})."
        )


def _parsed_side(side: str) -> ast.Expression | None:
    """Parse one side of a formula as Python, or return None when it does not parse.

    ``:`` becomes ``*`` as in :func:`_validate_formula_ast`, and the side is parsed in parentheses, so it may
    start with a space or run over lines. Every name keeps its place in ``side``, one byte on in its first line.
    """
    try:
        return ast.parse(f"({side.replace(':', '*')})".encode(), mode="eval")
    except (SyntaxError, ValueError):
        return None


def stand_ins_for(formula: str, columns: Iterable[str]) -> dict[str, str]:
    """Map each column whose name *formula* calls to a name no column has: ``I`` to ``I_``, or ``I_2`` if that is taken.

    Patsy finds a column before its own function of the same name (see :func:`_check_no_column_hides_a_call`).
    The model is fitted with the column under its stand-in (:func:`with_stand_ins`), and the results are
    reported under its real name (:func:`real_names`), so a factor can be named ``I`` in a model with squares,
    or ``C`` in one with contrasts. A column the formula only names, as ``I`` in ``y ~ I + P``, keeps its name.

    Parameters
    ----------
    formula : str
        The model formula, or one side of it. One that does not parse maps nothing, and
        :func:`validate_formula_is_safe` then refuses it.
    columns : Iterable[str]
        The names of the data columns.

    Returns
    -------
    dict[str, str]
        Each hidden column's name, mapped to its stand-in; empty when the formula hides none.
    """
    if not isinstance(formula, str):
        return {}
    taken = {str(name) for name in columns}
    called = {node.id for side in formula.split("~") if (tree := _parsed_side(side)) for node in _called_names(tree)}
    stand_ins = {}
    for name in sorted(called & taken):
        stand_ins[name] = _unused_name(f"{name}_", taken)
        taken.add(stand_ins[name])
    return stand_ins


def with_stand_ins(formula: str, stand_ins: Mapping[str, str]) -> str:
    """Write *formula* with each column in *stand_ins* under its stand-in; the functions it calls keep their names.

    ``y ~ I + I(I ** 2)`` with ``{"I": "I_"}`` becomes ``y ~ I_ + I(I_ ** 2)``.
    """
    if not stand_ins:
        return formula
    return "~".join(_side_with_stand_ins(side, stand_ins) for side in formula.split("~"))


def _side_with_stand_ins(side: str, stand_ins: Mapping[str, str]) -> str:
    """Write one side of a formula with each column in *stand_ins* under its stand-in (see :func:`with_stand_ins`)."""
    tree = _parsed_side(side)
    if tree is None:
        return side
    called = {id(node) for node in _called_names(tree)}
    text = side.encode()  # the parser counts columns in bytes
    starts = list(itertools.accumulate((len(line) + 1 for line in text.split(b"\n")), initial=0))
    columns = [node for node in ast.walk(tree) if isinstance(node, ast.Name) and id(node) not in called]
    for node in sorted(columns, key=lambda node: (node.lineno, node.col_offset), reverse=True):
        if node.id in stand_ins and node.end_lineno is not None and node.end_col_offset is not None:
            start = starts[node.lineno - 1] + node.col_offset - (node.lineno == 1)  # less the "(" around the side
            end = starts[node.end_lineno - 1] + node.end_col_offset - (node.end_lineno == 1)
            text = text[:start] + stand_ins[node.id].encode() + text[end:]
    return text.decode()


_Value = TypeVar("_Value")


def real_names(value: _Value, stand_ins: Mapping[str, str]) -> _Value:
    """Return *value* with each stand-in of *stand_ins* read as the name it stands for.

    A stand-in is replaced where it stands as a name of its own, so ``I(I_ ** 2)`` reads
    ``I(I ** 2)`` and ``I_:P`` reads ``I:P``. A column really named ``I_`` keeps its name, since a
    stand-in is a name that no column has (:func:`stand_ins_for`), and so does a level label
    after patsy's ``[T.``.

    Parameters
    ----------
    value : str, dict, list or tuple
        The results: strings, and dicts, lists and tuples of them, nested to any depth. Keys are
        read as well as values; anything else is returned as it is.
    stand_ins : Mapping[str, str]
        Each real name, mapped to its stand-in.

    Returns
    -------
    str, dict, list or tuple
        The same structure, under the real names.
    """
    if not stand_ins:
        return value
    real = {stand_in: name for name, stand_in in stand_ins.items()}
    pattern = re.compile(rf"(?<![\w.])({'|'.join(map(re.escape, real))})(?!\w)")

    def read(item: object) -> object:
        if isinstance(item, str):
            return pattern.sub(lambda match: real[match.group(1)], item)
        if isinstance(item, dict):
            return {read(key): read(entry) for key, entry in item.items()}
        if type(item) is list:
            return [read(entry) for entry in item]
        if type(item) is tuple:
            return tuple(read(entry) for entry in item)
        return item

    return cast("_Value", read(value))


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
                f"formula references unknown name(s) [{node.id!r}]; only data columns are allowed: {sorted(allowed)}."
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
    #: Each column fitted under a stand-in name, mapped to it (see :func:`stand_ins_for`); set by ``lm()``.
    _stand_ins: dict[str, str]

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
        self._stand_ins = {}

    def __str__(self) -> str:
        """Return the model specification as a string."""
        spec = ModelDesc.from_formula(self._model_spec)
        return spec.describe()

    def summary(self, alpha: float = 0.05, print_to_screen: bool = True) -> Any:  # noqa: ARG002, ANN401
        """Build the OLS summary table for this model and return it.

        The returned object is the statsmodels summary instance, with the
        underlying ``self._OLS.summary()`` adjusted to label the residual
        standard error row and to leave out the date and time of the call,
        so that the same fit gives the same summary. The method does NOT
        print anything by itself;
        the top-level :func:`summary` wrapper handles screen output via its
        own ``show`` flag. The ``alpha`` and ``print_to_screen`` arguments
        are unused and kept for backwards compatibility.
        """
        # Taken from statsmodels.regression.linear_model.py
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            main = "OLS Regression Results"
            title = self.name or self.get_title()
            if title:
                main += ": " + str(title)

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

            # The time of the call is no fact about the model, and it made two summaries of one fit differ
            # (#952): leave out the Date and Time cells, move the left-hand cells below them up, and drop
            # the row that leaves empty.
            top = smry.tables[0]
            left = [(row[0].data, row[1].data) for row in top if row[0].data not in ("Date:", "Time:")]
            for row, (label, value) in zip(top, left + [("", "")] * (len(top) - len(left)), strict=True):
                row[0].data, row[1].data = label, value
            while not "".join(str(cell.data) for cell in top[-1]).strip():
                top.pop()

        return smry

    def get_parameters(self, drop_intercept: bool = True) -> pd.Series:
        """Get the parameter values; return them in a pandas Series indexed by term."""

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
        """Get the model's title, if it has one. Always returns a string.

        The title is the data's ``pi_title``. ``lm`` also accepts a plain DataFrame,
        which has none, so a missing attribute is an empty title rather than an error.
        """
        return str(getattr(self.data, "pi_title", None) or "")

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
            # Stored under the term's factors, ("A", "B") for "A:B"; ``get`` rather than
            # indexing, which on the defaultdict would add an empty entry per lookup.
            for alias in self.aliasing.get(tuple(p_name.split(":")), []):
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
    return model._OLS.predict(exog={model._stand_ins.get(name, name): value for name, value in kwargs.items()})


#: Correlation beyond which two model columns are exact aliases: one, up to rounding.
_EXACT_ALIAS = 1.0 - 1e-9


def _alias_word(term: Any) -> list[str]:  # noqa: ANN401
    """Return the factor names of a patsy term, ``["Intercept"]`` for the intercept."""
    return [factor.name() for factor in term.factors] or ["Intercept"]


def _find_aliases(exog: np.ndarray, terms: list[Any], threshold: float) -> tuple[dict, list[int]]:
    """Find the model columns that are aliases of one another, and the ones to drop.

    Two columns that vary are aliases when the absolute correlation of the centred
    columns exceeds ``threshold`` (a threshold of one means exactly, up to rounding);
    the sign of an alias is the sign of that correlation, so it holds in actual units
    too (``E = 30 - A`` is ``-A``). Columns that do not vary (the intercept, or a word
    that is constant across a fraction, ``ABC = -1``) are aliases of one another,
    signed by their levels. In each group the shortest word is kept, the first one on a
    tie, and the others are dropped. ``terms`` holds each column's patsy term.

    Returns ``{word: [[sign, *alias word], ...]}``, each word a tuple of factor names
    with its aliases shortest first and the intercept last, and the columns to drop.
    """
    exog = np.asarray(exog, dtype=float)
    centred = exog - exog.mean(axis=0)
    spread = np.linalg.norm(centred, axis=0)
    varies = spread > np.sqrt(np.finfo(float).eps) * np.maximum(np.linalg.norm(exog, axis=0), 1.0)
    unit = np.divide(centred, spread, out=np.zeros_like(centred), where=varies)
    correlation = unit.T @ unit
    limit = min(threshold, _EXACT_ALIAS)

    aliasing: dict = defaultdict(list)
    drop: set[int] = set()
    for idx in range(exog.shape[1]):
        if varies[idx]:
            group = np.flatnonzero(varies & (np.abs(correlation[idx]) > limit))
            signs = np.sign(correlation[idx])
        else:
            group = np.flatnonzero(~varies)
            signs = np.sign(exog[0, idx] * exog[0])
        ordered = sorted(group.tolist(), key=lambda col: (len(terms[col].factors), col))
        drop.update(ordered[1:])
        aliases = [["-" if signs[col] < 0 else "+", *_alias_word(terms[col])] for col in ordered if col != idx]
        if aliases:
            aliasing[tuple(_alias_word(terms[idx]))] = sorted(
                aliases, key=lambda alias: (1e5 if alias[1] == "Intercept" else len(alias), alias)
            )
    return aliasing, sorted(drop)


def lm(
    model_spec: str,
    data: Expt,
    name: str | None = None,
    alias_threshold: float | None = _EXACT_ALIAS,
) -> Model:
    """Create a linear model.

    Parameters
    ----------
    model_spec : str
        The model formula, e.g. ``"y ~ A * B"``. A column named like a function the formula
        calls, as ``I`` in ``I(P ** 2)``, is fitted under a stand-in name and reported under its own.
    data : Expt
        The data; a plain ``Expt`` or DataFrame works too.
    name : str or None
        The model's name; defaults to the data's ``pi_title``, if it has one.
    alias_threshold : float or None
        Two model columns whose absolute correlation exceeds this are treated as
        aliases, and the longer term is dropped from the fit and reported as an alias.
        The default detects only exact aliasing (a correlation of one up to rounding),
        as in a fractional factorial; a column that is merely highly correlated, such
        as a square in actual units, is kept. ``None`` drops nothing.

    Returns
    -------
    Model
        The fitted model.
    """

    # Patsy evaluates each formula term as a Python expression, so an untrusted
    # ``model_spec`` is a code-execution vector. Allow only a safe Wilkinson
    # formula over the data columns, optionally with I()/Q() and a curated set of
    # element-wise numpy transforms (the public textbook API relies on these).
    stand_ins = stand_ins_for(model_spec, data.columns)
    fit_spec = with_stand_ins(model_spec, stand_ins)
    fit_data = data.rename(columns=stand_ins) if stand_ins else data
    validate_formula_is_safe(fit_spec, fit_data.columns, allow_transforms=True, allow_numpy=True)

    pre_model = smf.ols(fit_spec, data=fit_data)
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
    if alias_threshold is None:
        aliasing: dict = {}
        drop_column_names: list[str] = []
    else:
        # Map each model column to its patsy term: a categorical term spans several
        # columns, and patsy orders the terms in the model matrix its own way.
        spec: Any = getattr(pre_model.data, "model_spec", None) or pre_model.data.design_info
        column_terms: list[Any] = [None] * n_terms
        for term, columns in spec.term_slices.items():
            column_terms[columns] = [term] * (columns.stop - columns.start)
        aliasing, drop_columns = _find_aliases(pre_model.exog, column_terms, alias_threshold)
        drop_column_names = [pre_model.data.xnames[i] for i in drop_columns]

    post_model = smf.ols(fit_spec, data=fit_data, drop_cols=drop_column_names)
    if stand_ins:  # the coefficients, the summary and the aliases are read under the real names
        post_model.data.xnames = real_names(post_model.data.xnames, stand_ins)
        post_model.data.ynames = real_names(post_model.data.ynames, stand_ins)
        aliasing = real_names(aliasing, stand_ins)

    name = name or getattr(data, "pi_title", None)
    out = Model(
        OLS_instance=post_model.fit(),
        model_spec=model_spec,
        aliasing=aliasing,
        name=name,
    )
    out.data = data
    out._stand_ins = stand_ins

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
    "suggest_identifier",
    "summary",
    "validate_formula_is_safe",
    "validate_identifier_is_safe",
    "validate_identifiers_are_safe",
]
