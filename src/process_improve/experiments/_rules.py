# (c) Kevin Dunn, 2010-2026. MIT License.

"""Constraints that name categorical factors: disallowed combinations and limits on some levels only.

A constraint is a statement every run must satisfy. Beside the inequalities over
continuous factors that :func:`~process_improve.experiments.designs_constrained.parse_constraint`
reads, a constraint may compare a categorical factor with its levels (``==``, ``!=``,
``in``, ``not in``) and join comparisons with ``and``, ``or`` and ``not``:

- ``"not (catalyst == 'B' and solvent == 'Y')"`` disallows one combination of levels;
- ``"not (catalyst == 'B' and T > 180)"`` limits ``T`` to 180 when the catalyst is B;
- ``"catalyst != 'C' or (T <= 170 and P <= 4)"`` keeps catalyst C in one corner.

Each such constraint is rewritten as an AND of OR-clauses (its conjunctive normal form),
and each clause is sorted by what it holds:

- level tests only: a combination of levels is allowed where at least one test holds;
- level tests and one inequality: the inequality applies where no test holds;
- one inequality only: an ordinary constraint, which applies everywhere;
- two or more inequalities: refused, because the settings they allow do not form a
  convex region, which the design engines need for each combination of levels.

So a region with rules is a finite union of ordinary regions: one per allowed
combination of the levels the rules name, each cut by the inequalities that apply there.
"""

from __future__ import annotations

import ast
import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from process_improve.experiments.designs_constrained import MAX_EXPRESSION_LENGTH, nfkc

if TYPE_CHECKING:
    from process_improve.experiments.factor import Constraint

#: Most OR-clauses one constraint may expand into; a longer expansion is refused rather than built.
MAX_CLAUSES = 64
#: Most combinations of the levels of the factors the rules name; every one is checked against the rules.
MAX_COMBINATIONS = 100_000

#: The comparison each comparison becomes under ``not``.
_NEGATED: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Lt: ast.GtE,
    ast.LtE: ast.Gt,
    ast.Gt: ast.LtE,
    ast.GtE: ast.Lt,
}

Labels = Mapping[str, np.ndarray]


@dataclass(frozen=True)
class LevelTest:
    """A test that holds where ``factor`` takes one of ``levels`` (each written as text)."""

    factor: str
    levels: frozenset[str]

    def holds(self, labels: Labels) -> np.ndarray:
        """Return, per run, whether the factor's level is one of :attr:`levels`."""
        return np.isin(labels[self.factor], list(self.levels))


@dataclass(frozen=True)
class Clause:
    """One OR-clause of a constraint: it holds where a level test holds, or else where its inequality does.

    Parameters
    ----------
    tests : tuple[LevelTest, ...]
        At most one test per factor.
    limit : str or None
        An inequality over continuous factors, as text
        :func:`~process_improve.experiments.designs_constrained.parse_constraint` reads;
        None when the clause only tests levels.
    source : int
        Index of the constraint the clause came from.
    """

    tests: tuple[LevelTest, ...]
    limit: str | None
    source: int

    def exempt(self, labels: Labels) -> np.ndarray:
        """Return, per run, whether a level test holds, so the clause holds whatever the continuous settings."""
        n = len(next(iter(labels.values()))) if labels else 0
        out = np.zeros(n, dtype=bool)
        for test in self.tests:
            out |= test.holds(labels)
        return out


@dataclass(frozen=True)
class Rules:
    """The clauses of a region's constraints that test levels of categorical factors.

    Parameters
    ----------
    levels : dict[str, tuple[str, ...]]
        Each categorical factor the rules name, with its levels written as text, in the
        order the factors were given.
    combinations : tuple[Clause, ...]
        Clauses of level tests only: a combination is allowed where each holds.
    limits : tuple[Clause, ...]
        Clauses of level tests and one inequality: the inequality applies on the runs
        where no test holds.
    """

    levels: dict[str, tuple[str, ...]]
    combinations: tuple[Clause, ...]
    limits: tuple[Clause, ...]

    @property
    def factors(self) -> list[str]:
        """Names of the categorical factors the rules name."""
        return list(self.levels)

    def allowed(self, labels: Labels) -> np.ndarray:
        """Return, per run, whether its combination of levels is allowed."""
        n = len(next(iter(labels.values()))) if labels else 0
        out = np.ones(n, dtype=bool)
        for clause in self.combinations:
            out &= clause.exempt(labels)
        return out

    def applies(self, labels: Labels) -> np.ndarray:
        """Return ``(runs, limits)``: whether each limit in :attr:`limits` applies to each run."""
        n = len(next(iter(labels.values()))) if labels else 0
        columns = [~clause.exempt(labels) for clause in self.limits]
        return np.column_stack(columns) if columns else np.zeros((n, 0), dtype=bool)

    def table(self) -> pd.DataFrame:
        """Return every combination of the named factors' levels (as text), one column per factor."""
        return pd.DataFrame(list(itertools.product(*self.levels.values())), columns=self.factors, dtype=object)

    def allowed_table(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Return the allowed combinations, and ``(combinations, limits)``: which limits apply on each."""
        table = self.table()
        labels = {name: table[name].to_numpy(dtype=str) for name in self.factors}
        keep = self.allowed(labels)
        return table[keep].reset_index(drop=True), self.applies(labels)[keep]


def text_labels(values: Iterable[Any]) -> np.ndarray:
    """Return the levels as the text the rules compare them by."""
    return np.array([str(v) for v in values], dtype=str)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def mentions(expression: str, names: Iterable[str]) -> bool:
    """Return whether ``expression`` names one of ``names`` (False for an expression that does not parse)."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False
    wanted = {nfkc(name) for name in names}
    return any(isinstance(node, ast.Name) and node.id in wanted for node in ast.walk(tree))


def names_categorical(expression: str, categorical: Iterable[str]) -> bool:
    """Return whether ``expression`` is a rule: it joins comparisons with ``and``/``or``/``not``, or names a level.

    An expression that does not parse, or is a single comparison over continuous
    factors, is not a rule; the ordinary parser then reads it, with its own messages.
    """
    try:
        tree = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return False
    if isinstance(tree, ast.BoolOp) or (isinstance(tree, ast.UnaryOp) and isinstance(tree.op, ast.Not)):
        return True
    return mentions(expression, categorical)


def compile_rules(
    constraints: Sequence[Constraint], continuous: Iterable[str], categorical: Mapping[str, Sequence[Any]]
) -> tuple[list[tuple[int, str]], Rules | None]:
    """Split constraints into ordinary inequalities and rules on categorical factors.

    Parameters
    ----------
    constraints : Sequence[Constraint]
        The constraints, in order.
    continuous : Iterable[str]
        Names of the continuous factors.
    categorical : Mapping[str, Sequence]
        Each categorical factor's levels.

    Returns
    -------
    tuple[list[tuple[int, str]], Rules or None]
        The inequalities that apply everywhere, each with the index of the constraint it
        came from (a constraint that is not a rule is passed on as written), and the
        rules, or None when no constraint is one.

    Raises
    ------
    ValueError
        For a rule that compares a categorical factor with something other than its
        levels, names a level the factor does not have, joins two inequalities with
        ``or``, expands into more than ``MAX_CLAUSES`` clauses, or leaves no allowed
        combination, or a level with no allowed combination.
    """
    lookup = {nfkc(name): name for name in categorical}
    levels = {name: tuple(str(v) for v in values) for name, values in categorical.items()}
    plain: list[tuple[int, str]] = []
    combination_clauses: list[Clause] = []
    limit_clauses: list[Clause] = []
    for index, constraint in enumerate(constraints):
        expression = constraint.expression
        if not names_categorical(expression, categorical):
            plain.append((index, expression))
            continue
        for clause in _clauses(expression, index, lookup, levels):
            if not clause.tests:
                plain.append((index, clause.limit or ""))
            elif clause.limit is None:
                combination_clauses.append(clause)
            else:
                limit_clauses.append(clause)
    named = {test.factor for clause in combination_clauses + limit_clauses for test in clause.tests}
    if not named:
        return plain, None
    rules = Rules(
        {name: levels[name] for name in categorical if name in named},
        tuple(combination_clauses),
        tuple(limit_clauses),
    )
    _check_combinations(rules)
    return plain, rules


def _check_combinations(rules: Rules) -> None:
    """Refuse rules that allow no combination, or leave a level of a named factor with none."""
    size = int(np.prod([len(levels) for levels in rules.levels.values()]))
    if size > MAX_COMBINATIONS:
        raise ValueError(
            f"The rules name categorical factors {rules.factors} with {size} combinations of levels, more than the "
            f"{MAX_COMBINATIONS} that can be checked."
        )
    allowed, _ = rules.allowed_table()
    if allowed.empty:
        raise ValueError(f"The constraints disallow every combination of the levels of {rules.factors}.")
    for name, levels in rules.levels.items():
        empty = [level for level in levels if level not in set(allowed[name])]
        if empty:
            shown = ", ".join(f"{name} = {level!r}" for level in empty)
            raise ValueError(
                f"The constraints leave no allowed run with {shown}. Remove that level from the factor, or "
                "loosen the constraints."
            )


def _clauses(
    expression: str, source: int, lookup: Mapping[str, str], levels: Mapping[str, tuple[str, ...]]
) -> list[Clause]:
    """Return the OR-clauses of one rule, each with merged level tests and at most one inequality."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"Constraint expression is longer than {MAX_EXPRESSION_LENGTH} characters.")
    try:
        tree = ast.parse(expression, mode="eval").body
    except SyntaxError as err:  # pragma: no cover - names_categorical has parsed it already
        raise ValueError(f"Constraint {expression!r} is not a valid expression: {err.msg}.") from err
    parsed = _conjunctive(_negation_normal(tree, expression), expression)
    out: list[Clause] = []
    for atoms in parsed:
        tests: dict[str, set[str]] = {}
        limits: list[ast.Compare] = []
        for atom in atoms:
            test = _level_test(atom, expression, lookup, levels)
            if test is None:
                limits.append(atom)
            else:
                tests.setdefault(test.factor, set()).update(test.levels)
        if any(len(chosen) == len(levels[name]) for name, chosen in tests.items()):
            continue  # a test that holds for every level: the clause always holds
        if len(limits) > 1:
            shown = " or ".join(repr(ast.unparse(limit)) for limit in limits)
            raise ValueError(
                f"Constraint {expression!r} allows a run where {shown} holds. Settings allowed by either of two "
                "inequalities do not form a convex region, which the design engines need for each combination of "
                "levels. Write the inequalities as constraints that each must hold."
            )
        level_tests = tuple(LevelTest(name, frozenset(chosen)) for name, chosen in tests.items() if chosen)
        if not level_tests and not limits:
            raise ValueError(f"Constraint {expression!r} can never hold: it names no level of its factors.")
        out.append(Clause(level_tests, ast.unparse(limits[0]) if limits else None, source))
    return out


def _negation_normal(node: ast.expr, expression: str, negate: bool = False) -> ast.expr:
    """Push every ``not`` down to the comparisons, splitting a chained comparison into its pairs."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _negation_normal(node.operand, expression, not negate)
    if isinstance(node, ast.BoolOp):
        op = node.op
        if negate:  # De Morgan: not (a and b) is (not a) or (not b), and the reverse
            op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        return ast.BoolOp(op=op, values=[_negation_normal(v, expression, negate) for v in node.values])
    if isinstance(node, ast.Compare):
        sides = [node.left, *node.comparators]
        pairs = [
            ast.Compare(left=left, ops=[_NEGATED[type(op)]() if negate else op], comparators=[right])
            for op, left, right in zip(node.ops, sides[:-1], sides[1:], strict=True)
        ]
        if len(pairs) == 1:
            return pairs[0]
        return ast.BoolOp(op=ast.Or() if negate else ast.And(), values=list(pairs))
    raise ValueError(
        f"Constraint {expression!r}: {ast.unparse(node)!r} is not a comparison. Join comparisons such as "
        "'T <= 180' or \"catalyst == 'B'\" with and, or, not."
    )


def _conjunctive(node: ast.expr, expression: str) -> list[list[ast.Compare]]:
    """Distribute ``or`` over ``and``: a list of clauses, each a list of comparisons any of which may hold."""
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        out = [clause for value in node.values for clause in _conjunctive(value, expression)]
    elif isinstance(node, ast.BoolOp):
        parts = [_conjunctive(value, expression) for value in node.values]
        size = int(np.prod([len(part) for part in parts]))
        if size > MAX_CLAUSES:
            raise ValueError(
                f"Constraint {expression!r} expands into {size} clauses, more than {MAX_CLAUSES}; split it into "
                "several constraints."
            )
        out = [[atom for clause in combo for atom in clause] for combo in itertools.product(*parts)]
    else:
        out = [[node]]  # type: ignore[list-item]  # _negation_normal leaves only comparisons at the leaves
    if len(out) > MAX_CLAUSES:
        raise ValueError(
            f"Constraint {expression!r} expands into {len(out)} clauses, more than {MAX_CLAUSES}; split it into "
            "several constraints."
        )
    return out


def _level_test(
    atom: ast.Compare, expression: str, lookup: Mapping[str, str], levels: Mapping[str, tuple[str, ...]]
) -> LevelTest | None:
    """Read a comparison of a categorical factor with its levels; None for a comparison of continuous factors."""
    named = [node.id for node in ast.walk(atom) if isinstance(node, ast.Name) and node.id in lookup]
    if not named:
        return None
    op, left, right = type(atom.ops[0]), atom.left, atom.comparators[0]
    if op in (ast.Eq, ast.NotEq) and isinstance(right, ast.Name) and not isinstance(left, ast.Name):
        left, right = right, left  # 'B' == catalyst
    name = lookup.get(left.id) if isinstance(left, ast.Name) else None
    values = _literal_levels(right, op)
    if name is None or values is None or op not in (ast.Eq, ast.NotEq, ast.In, ast.NotIn):
        factor = lookup[named[0]]
        example = levels[factor][0] if levels[factor] else "level"
        raise ValueError(
            f"Constraint {expression!r}: compare the categorical factor {factor!r} with its levels, as in "
            f'"{factor} == \'{example}\'" or "{factor} in ({", ".join(repr(v) for v in levels[factor][:2])})".'
        )
    unknown = sorted(set(values) - set(levels[name]))
    if unknown:
        raise ValueError(
            f"Constraint {expression!r}: {name!r} has no level {', '.join(repr(v) for v in unknown)}; its levels are "
            f"{list(levels[name])}."
        )
    chosen = set(values) if op in (ast.Eq, ast.In) else set(levels[name]) - set(values)
    return LevelTest(name, frozenset(chosen))


def _literal_levels(node: ast.expr, op: type[ast.cmpop]) -> list[str] | None:
    """Return the levels a comparison names, as text, or None when they are not plain numbers or text."""
    items = node.elts if op in (ast.In, ast.NotIn) and isinstance(node, ast.Tuple | ast.List | ast.Set) else [node]
    if op in (ast.In, ast.NotIn) and items == [node]:
        return None
    out = []
    for item in items:
        try:
            value = ast.literal_eval(item)  # a number or a string, a negative number included
        except ValueError:
            return None
        if not isinstance(value, str | int | float) or isinstance(value, bool):
            return None
        out.append(str(value))
    return out
