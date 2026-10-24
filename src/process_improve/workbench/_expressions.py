# (c) Kevin Dunn, 2010-2026. MIT License.
"""The expression language of a recipe's computed columns and row filters.

An expression is parsed with :mod:`ast` and walked, never evaluated as Python, so an
expression read from someone else's project file is data. The same walk gives the
value (:func:`evaluate`) and the pandas code that computes it (:func:`to_code`), so the
code a recipe writes does exactly what the recipe did, operation for operation.

Accepted: numbers, column names, ``+ - * / **``, one comparison (``< <= > >= == !=``)
against a number or a quoted text, ``&``, ``|`` and ``~`` to combine conditions, and
the functions in :data:`FUNCTIONS`. ``and``, ``or`` and ``not`` combine conditions as a
reader expects (``Tmax1 > 290 and Mn < 30000``) and are written as pandas' ``&``, ``|``
and ``~``. A column name that is not a Python identifier is
written in back-ticks, as in :meth:`pandas.DataFrame.query`: ``` `Real coffee` * 2 ```.
"""

from __future__ import annotations

import ast
import functools
import io
import keyword
import operator
import re
import tokenize
import unicodedata
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import pandas as pd

#: Longest expression accepted, which also bounds the depth of the parse.
MAX_EXPRESSION_LENGTH = 500

#: The functions an expression may call, each with one argument, and the code that calls it.
FUNCTIONS: dict[str, tuple[Callable[[Any], Any], str]] = {
    "abs": (np.abs, "np.abs"),
    "exp": (np.exp, "np.exp"),
    "log": (np.log, "np.log"),
    "log10": (np.log10, "np.log10"),
    "log1p": (np.log1p, "np.log1p"),
    "sign": (np.sign, "np.sign"),
    "sqrt": (np.sqrt, "np.sqrt"),
}

_BINARY: dict[type[ast.operator], tuple[Callable[[Any, Any], Any], str]] = {
    ast.Add: (operator.add, "+"),
    ast.Sub: (operator.sub, "-"),
    ast.Mult: (operator.mul, "*"),
    ast.Div: (operator.truediv, "/"),
    ast.Pow: (operator.pow, "**"),
    ast.BitAnd: (operator.and_, "&"),
    ast.BitOr: (operator.or_, "|"),
}
_UNARY: dict[type[ast.unaryop], tuple[Callable[[Any], Any], str]] = {
    ast.USub: (operator.neg, "-"),
    ast.UAdd: (operator.pos, "+"),
    ast.Invert: (operator.invert, "~"),
    ast.Not: (operator.invert, "~"),
}
# ``and`` / ``or`` bind after the comparisons, as a reader expects, and become pandas' ``&`` / ``|``.
_BOOLEAN: dict[type[ast.boolop], tuple[Callable[[Any, Any], Any], str]] = {
    ast.And: (operator.and_, "&"),
    ast.Or: (operator.or_, "|"),
}
_COMPARE: dict[type[ast.cmpop], tuple[Callable[[Any, Any], Any], str]] = {
    ast.Lt: (operator.lt, "<"),
    ast.LtE: (operator.le, "<="),
    ast.Gt: (operator.gt, ">"),
    ast.GtE: (operator.ge, ">="),
    ast.Eq: (operator.eq, "=="),
    ast.NotEq: (operator.ne, "!="),
}
_QUOTED = re.compile(r"`([^`]*)`")
_PLACEHOLDER = "__column_{}__"


class Expression:
    """A parsed, checked expression over the columns of a table.

    Parameters
    ----------
    text : str
        The expression as typed.
    columns : Iterable[str] or None
        The columns it may name. ``None`` takes every name in the text as a column, as
        typed: what writing the code of an expression already checked against its
        table needs.

    Raises
    ------
    ValueError
        If the text is too long, does not parse, uses syntax outside the language, or
        names a column that is not in ``columns``.
    """

    def __init__(self, text: str, columns: Iterable[str] | None) -> None:
        if len(text) > MAX_EXPRESSION_LENGTH:
            raise ValueError(f"An expression may be at most {MAX_EXPRESSION_LENGTH} characters long.")
        self.text = text
        quoted: list[str] = []

        def placeholder(match: re.Match[str]) -> str:
            quoted.append(match.group(1))
            return _PLACEHOLDER.format(len(quoted) - 1)

        source = _QUOTED.sub(placeholder, text)
        try:
            self._tree = ast.parse(source.strip(), mode="eval").body
        except SyntaxError as error:
            raise ValueError(f"The expression {text!r} does not parse: {error.msg}.") from None
        # Python's parser writes every identifier in its NFKC form, so a plain name is
        # matched on that form; a back-ticked name is matched exactly as written.
        known = list(columns) if columns is not None else [*_typed_names(source), *quoted]
        lookup = {unicodedata.normalize("NFKC", name): name for name in known}
        lookup.update({_PLACEHOLDER.format(i): name for i, name in enumerate(quoted)})
        self._names = lookup
        self._quoted = quoted
        self._known = set(known)
        self.columns = self._check(self._tree)

    def _check(self, node: ast.expr, *, text_allowed: bool = False) -> list[str]:
        """Return the columns ``node`` names, in order, refusing anything outside the language.

        ``text_allowed`` admits a quoted text, which only the right side of a comparison may be.
        """
        if isinstance(node, ast.Constant) and (
            _is_number(node.value) or (text_allowed and isinstance(node.value, str))
        ):
            return []
        if isinstance(node, ast.Name):
            return [self._column(node.id)]
        if isinstance(node, ast.Compare) and len(node.ops) > 1:
            raise ValueError(
                f"{self.text!r} chains comparisons. Join them with 'and' or 'or', for example "
                "'Tmax1 > 290 and Mn < 30000', or put each one in parentheses."
            )
        operation = _operation(node)
        if operation is None:
            shown = re.sub(r"__column_(\d+)__", lambda m: f"`{self._quoted[int(m.group(1))]}`", ast.unparse(node))
            raise ValueError(f"{shown!r} is not allowed in an expression; see the expression language.")
        children = operation[2]
        compared = isinstance(node, ast.Compare)
        return [
            name for i, child in enumerate(children) for name in self._check(child, text_allowed=compared and i == 1)
        ]

    def _column(self, parsed: str) -> str:
        name = self._names.get(parsed)
        if name is None or name not in self._known:
            missing = name if name is not None else parsed
            raise ValueError(f"The expression names {missing!r}, which is not a column.")
        return name

    def evaluate(self, frame: pd.DataFrame) -> Any:  # noqa: ANN401 - a Series, or a number for a constant expression
        """Return the expression's value over ``frame``: a Series, one value per row."""
        return self._value(self._tree, frame)

    def _value(self, node: ast.expr, frame: pd.DataFrame) -> Any:  # noqa: ANN401
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return frame[self._names[node.id]]
        function, _, children = cast("_Operation", _operation(node))  # _check admitted it
        return function(*(self._value(child, frame) for child in children))

    def to_code(self, frame: str = "df") -> str:
        """Return the pandas code that computes the expression over the table named ``frame``.

        Every operation is parenthesised, so the code groups exactly as the parse did.
        """
        return self._code(self._tree, frame)

    def _code(self, node: ast.expr, frame: str) -> str:
        if isinstance(node, ast.Constant):
            return python_literal(node.value)
        if isinstance(node, ast.Name):
            return f"{frame}[{python_literal(self._names[node.id])}]"
        _, write, children = cast("_Operation", _operation(node))  # _check admitted it
        return write([self._code(child, frame) for child in children])


#: An operation: the function that computes it, how its code joins its operands' code, and its operands.
_Operation = tuple["Callable[..., Any]", "Callable[[list[str]], str]", list[ast.expr]]


def _operation(node: ast.expr) -> _Operation | None:
    """Return the operation at ``node``, or None when the language has no such operation."""
    fn: Callable[..., Any]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        fn, symbol = _BINARY[type(node.op)]
        return fn, lambda c: f"({c[0]} {symbol} {c[1]})", [node.left, node.right]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        fn, symbol = _UNARY[type(node.op)]
        return fn, lambda c: f"({symbol}{c[0]})", [node.operand]
    if isinstance(node, ast.BoolOp) and type(node.op) in _BOOLEAN:
        fn, symbol = _BOOLEAN[type(node.op)]
        return lambda *v: functools.reduce(fn, v), lambda c: "(" + f" {symbol} ".join(c) + ")", list(node.values)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE:
        fn, symbol = _COMPARE[type(node.ops[0])]
        return fn, lambda c: f"({c[0]} {symbol} {c[1]})", [node.left, node.comparators[0]]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in FUNCTIONS
        and len(node.args) == 1
        and not node.keywords
    ):
        fn, name = FUNCTIONS[node.func.id]
        return fn, lambda c: f"{name}({c[0]})", [node.args[0]]
    return None


def _typed_names(source: str) -> list[str]:
    """Return the names in ``source`` as typed, before the parser rewrites them in NFKC form."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return []  # the parse that follows reports the error
    return [t.string for t in tokens if t.type == tokenize.NAME and not keyword.iskeyword(t.string)]


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def python_literal(value: object) -> str:
    """Return ``value`` as Python source: text in double quotes where it can be, numbers exactly."""
    if isinstance(value, str):
        text = repr(value)
        if text.startswith("'") and '"' not in value:
            return f'"{text[1:-1]}"'
        return text
    if isinstance(value, list | tuple):
        items = ", ".join(python_literal(v) for v in value)
        return f"({items},)" if isinstance(value, tuple) and len(value) == 1 else f"[{items}]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{python_literal(k)}: {python_literal(v)}" for k, v in value.items()) + "}"
    if isinstance(value, float) and not np.isfinite(value):
        return f"float({str(value)!r})"
    return repr(value)
