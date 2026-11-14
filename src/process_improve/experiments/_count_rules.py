# (c) Kevin Dunn, 2010-2026. MIT License.

"""Limits on how many of a set of factors a run may use: ``count_nonzero(A, B, C) <= 2``.

A factor that is left out of a run is at 0. Beside the inequalities over the factors'
values, a constraint may limit how many of the factors it names are above 0 in one run:

- ``"count_nonzero(A, B, C, D) <= 2"``: at most two of the four are used;
- ``"count_nonzero(A, B, C) >= 1"``: at least one of the three is used;
- ``"8 <= count_nonzero(A, B, C, ...) <= 14"``: between 8 and 14 of them are used;
- ``"count_nonzero(A, B, C) == 1"``: exactly one of the three is used.

Such a constraint is a comparison of one ``count_nonzero(...)`` with whole numbers, and
nothing else. The factors it names are continuous factors or mixture components whose
``low`` is 0 or more. One whose ``low`` is above 0 cannot be left out, so it always
counts. One with a ``min_when_used`` is left out or set at no less than that minimum;
one without it counts as used at any amount above 0, however small.

The settings such constraints allow do not form one convex region. They are a union of
pieces, one for each set of the factors in use: on a piece the factors left out are at
0, and the others range from their minimum (or from 0) to their upper limit, within the
other constraints. :class:`UsedSets` holds that structure for the design engines, the
sampler and the search of an optimum, which each work piece by piece.
"""

from __future__ import annotations

import ast
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from process_improve.experiments._expression_text import MAX_EXPRESSION_LENGTH, nfkc
from process_improve.experiments.factor import Constraint, Factor, FactorType

#: The function a count rule is written with.
COUNT = "count_nonzero"
#: How far above 0 (in the units the points are in) a factor counts as used.
_USED_TOL = 1e-9
#: With this many factors that can be left out, or fewer, every allowed set of them is listed; with more, sets
#: are drawn at random.
_LISTED_FACTORS = 10
#: Draws spent on each set wanted, when sets are drawn at random, before giving up on finding more.
_DRAWS_PER_SET = 20
#: Passes over the rules when a drawn set is changed until every rule holds.
_REPAIR_PASSES = 4

#: The comparison each comparison becomes when its two sides are exchanged.
_MIRRORED: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.Gt,
    ast.LtE: ast.GtE,
    ast.Gt: ast.Lt,
    ast.GtE: ast.LtE,
    ast.Eq: ast.Eq,
}


@dataclass(frozen=True)
class CountRule:
    """A limit on how many of ``names`` are above 0 in one run: at least ``at_least`` and at most ``at_most``."""

    names: tuple[str, ...]
    at_least: int
    at_most: int
    expression: str


def _calls_count(tree: ast.AST) -> bool:
    """Return whether ``tree`` calls ``count_nonzero`` anywhere."""
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == COUNT
        for node in ast.walk(tree)
    )


def is_count_rule(expression: str) -> bool:
    """Return whether ``expression`` uses ``count_nonzero`` (False for an expression that does not parse)."""
    try:
        return _calls_count(ast.parse(expression, mode="eval"))
    except SyntaxError:
        return False


def without_count_rules(constraints: Sequence[Constraint] | None) -> list[Constraint]:
    """Return the constraints that are not count rules, in order."""
    return [c for c in constraints or [] if not is_count_rule(c.expression)]


def _whole_number(node: ast.expr, expression: str) -> int:
    """Return the whole number ``node`` holds, or raise."""
    value = node.value if isinstance(node, ast.Constant) else None
    if isinstance(value, bool) or not isinstance(value, int | float) or value != int(value):
        raise ValueError(
            f"Constraint {expression!r}: {COUNT}(...) is compared with whole numbers, as in "
            f"'{COUNT}(A, B, C) <= 2' or '1 <= {COUNT}(A, B, C) <= 2'."
        )
    return int(value)


def count_rule_syntax(expression: str) -> tuple[list[str], int | None, int | None]:
    """Read a count rule: the names it counts (as parsed), and its lower and upper limit (None where it has none).

    Raises
    ------
    ValueError
        If ``expression`` is too long, does not parse, or is not a comparison of one
        ``count_nonzero(...)`` of factor names with whole numbers by ``<``, ``<=``, ``>``,
        ``>=`` or ``==``.
    """
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"Constraint expression is longer than {MAX_EXPRESSION_LENGTH} characters.")
    try:
        tree = ast.parse(expression, mode="eval").body
    except SyntaxError as err:
        raise ValueError(f"Constraint {expression!r} is not a valid expression: {err.msg}.") from err
    sides = [tree.left, *tree.comparators] if isinstance(tree, ast.Compare) else []
    calls = [
        side
        for side in sides
        if isinstance(side, ast.Call) and isinstance(side.func, ast.Name) and side.func.id == COUNT
    ]
    if len(calls) != 1 or sum(_calls_count(side) for side in sides) != 1:
        raise ValueError(
            f"Constraint {expression!r}: {COUNT}(...) stands alone on one side of a comparison with whole numbers, "
            f"as in '{COUNT}(A, B, C) <= 2'. It cannot be added to, multiplied or joined with 'and' or 'or'."
        )
    call = calls[0]
    if call.keywords or not call.args or not all(isinstance(arg, ast.Name) for arg in call.args):
        raise ValueError(f"Constraint {expression!r}: {COUNT}(...) takes factor names, as in '{COUNT}(A, B, C)'.")
    low: int | None = None
    high: int | None = None
    compare = tree  # an ast.Compare, since it has sides
    for op, left, right in zip(compare.ops, sides[:-1], sides[1:], strict=True):  # type: ignore[attr-defined]
        if left is not call and right is not call:
            raise ValueError(f"Constraint {expression!r}: each comparison must have {COUNT}(...) on one side.")
        kind = type(op) if left is call else _MIRRORED.get(type(op))  # read as: count <op> number
        number = _whole_number(right if left is call else left, expression)
        if kind in (ast.LtE, ast.Lt, ast.Eq):
            limit = number - (kind is ast.Lt)
            high = limit if high is None else min(high, limit)
        if kind in (ast.GtE, ast.Gt, ast.Eq):
            limit = number + (kind is ast.Gt)
            low = limit if low is None else max(low, limit)
        if kind not in _MIRRORED:
            raise ValueError(f"Constraint {expression!r}: only <, <=, >, >= and == compare {COUNT}(...) with a number.")
    return [arg.id for arg in call.args], low, high  # type: ignore[attr-defined]


def _counted_factors(expression: str, parsed: list[str], factors: Sequence[Factor]) -> list[Factor]:
    """Return the factors a count rule names, or raise for a name it cannot count."""
    lookup = {nfkc(f.name): f for f in factors}
    unknown = [name for name in parsed if name not in lookup]
    if unknown:
        known = ", ".join(f.name for f in factors if f.type != FactorType.categorical)
        raise ValueError(f"Constraint {expression!r} counts unknown factor(s) {unknown}; the factors are [{known}].")
    if len(set(parsed)) < len(parsed):
        raise ValueError(f"Constraint {expression!r} names a factor more than once.")
    counted = [lookup[name] for name in parsed]
    for f in counted:
        if f.type == FactorType.categorical or f.low is None or f.low < 0:
            raise ValueError(
                f"Constraint {expression!r} counts {f.name!r}, which is not an amount: {COUNT}(...) counts "
                "continuous factors and mixture components whose low is 0 or more, 0 being a factor left out."
            )
    return counted


def count_rules(constraints: Sequence[Constraint] | None, factors: Sequence[Factor]) -> list[CountRule]:
    """Return the count rules among ``constraints`` that can bind, in order.

    A rule that every setting meets (at most 3 of 3 factors, at least 1 of a set with a
    factor that cannot be left out) is dropped, so it changes nothing.

    Raises
    ------
    ValueError
        For a count rule that is not written as :func:`count_rule_syntax` reads it, that
        counts an unknown, repeated or categorical factor or one whose range starts below
        0, or that no setting can meet.
    """
    rules = []
    for constraint in constraints or []:
        expression = constraint.expression
        if not is_count_rule(expression):
            continue
        parsed, low, high = count_rule_syntax(expression)
        counted = _counted_factors(expression, parsed, factors)
        always = sum(f.low > 0 for f in counted)  # type: ignore[operator]
        at_least = always if low is None else max(low, always)
        at_most = len(counted) if high is None else min(high, len(counted))
        if at_least > at_most:
            raise ValueError(
                f"No setting meets {expression!r}: it counts {len(counted)} factor(s), of which {always} are in "
                f"every run (their low is above 0), so between {always} and {len(counted)} of them are used."
            )
        if at_least > always or at_most < len(counted):
            rules.append(CountRule(tuple(f.name for f in counted), at_least, at_most, expression))
    return rules


@dataclass
class UsedSets:
    """Which factors a run may use together, in a region with count rules.

    Parameters
    ----------
    names : list[str]
        The name of every column of the points the region works on.
    optional : list[tuple[int, float, float]]
        ``(column, zero, minimum)`` for each factor that can be left out, in the points'
        units: where it sits when it is left out, and the least it takes when it is used
        (``zero`` again for a factor without a ``min_when_used``). A factor that a rule
        counts and that is not listed here is used in every run.
    rules : list[CountRule]
        The limits on how many factors are used.
    """

    names: list[str]
    optional: list[tuple[int, float, float]]
    rules: list[CountRule]
    _members: list[tuple[np.ndarray, int, CountRule]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Note, per rule, which optional factors it counts and how many of its factors are always used."""
        position = {self.names[column]: i for i, (column, _zero, _minimum) in enumerate(self.optional)}
        self._members = []
        for rule in self.rules:
            members = np.array([position[name] for name in rule.names if name in position], dtype=int)
            self._members.append((members, len(rule.names) - len(members), rule))

    def used(self, x: np.ndarray) -> np.ndarray:
        """Return ``(n, n_optional)``: whether each point uses each factor that can be left out."""
        columns = [column for column, _zero, _minimum in self.optional]
        zeros = np.array([zero for _column, zero, _minimum in self.optional], dtype=float)
        return np.atleast_2d(x)[:, columns] > zeros + _USED_TOL

    def _sets_allowed(self, used: np.ndarray) -> np.ndarray:
        """Return, per row of ``used``, whether the set of factors in use meets every rule."""
        ok = np.ones(len(used), dtype=bool)
        for members, always, rule in self._members:
            count = used[:, members].sum(axis=1) + always
            ok &= (count >= rule.at_least) & (count <= rule.at_most)
        return ok

    def allows(self, x: np.ndarray) -> np.ndarray:
        """Return, per point, whether the number of factors it uses meets every rule."""
        return self._sets_allowed(self.used(x))

    def broken(self, x: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Return each rule's expression with the points that break it."""
        used = self.used(x)
        found = []
        for members, always, rule in self._members:
            count = used[:, members].sum(axis=1) + always
            found.append((rule.expression, (count < rule.at_least) | (count > rule.at_most)))
        return found

    def bounds(self, used: np.ndarray, bounds: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        """Return the bounds of the piece on which the factors ``used`` marks are in use and the others left out."""
        piece = list(bounds)
        for (column, zero, minimum), in_use in zip(self.optional, used, strict=True):
            piece[column] = (minimum, piece[column][1]) if in_use else (zero, zero)
        return piece

    def sets(self, wanted: int) -> np.ndarray:
        """Return up to ``wanted`` allowed sets of the factors in use, as rows of ``(n_sets, n_optional)`` booleans.

        With ``_LISTED_FACTORS`` optional factors or fewer, every allowed set is found, and
        ``wanted`` of them are kept when there are more. With more factors, sets are drawn:
        each factor is used with a probability that is itself drawn, so that small and
        large sets both come up, and a draw that breaks a rule has factors of that rule
        brought in or left out, at random, until the rules hold. The generator is seeded
        here: the sets, and the candidates built from them, do not depend on the caller's
        random state.
        """
        n = len(self.optional)
        rng = np.random.default_rng(0)
        if n <= _LISTED_FACTORS:
            every = np.array(list(itertools.product((False, True), repeat=n)), dtype=bool).reshape(2**n, n)
            allowed = every[self._sets_allowed(every)]
            if len(allowed) > wanted:
                allowed = allowed[np.sort(rng.choice(len(allowed), size=wanted, replace=False))]
            return allowed
        found: dict[bytes, np.ndarray] = {}
        for _ in range(_DRAWS_PER_SET * wanted):
            drawn = self._repaired(rng.random(n) < rng.uniform(), rng)
            if drawn is not None:
                found.setdefault(drawn.tobytes(), drawn)
                if len(found) == wanted:
                    break
        return np.array(list(found.values()), dtype=bool).reshape(len(found), n)

    def _repaired(self, used: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
        """Bring factors in or leave them out, at random, until ``used`` meets every rule; None when it does not."""
        for _ in range(_REPAIR_PASSES):
            for members, always, rule in self._members:
                count = int(used[members].sum()) + always
                if count > rule.at_most:
                    used[rng.choice(members[used[members]], size=count - rule.at_most, replace=False)] = False
                elif count < rule.at_least:
                    used[rng.choice(members[~used[members]], size=rule.at_least - count, replace=False)] = True
            if self._sets_allowed(used[None, :])[0]:
                return used
        return None


def used_sets(
    names: list[str], factors: Sequence[Factor], rules: list[CountRule], coded: Mapping[str, bool]
) -> UsedSets | None:
    """Return the :class:`UsedSets` of a region whose numeric columns are ``factors``, or None without rules.

    A factor can be left out when its ``low`` is 0 and it has a ``min_when_used`` or a
    rule counts it. ``coded[name]`` says whether the column is in coded units, where 0 is
    -1 and an amount ``a`` is ``2 a / high - 1``; otherwise the column holds the amount
    itself, as a mixture component's proportion.
    """
    if not rules:
        return None
    counted = {name for rule in rules for name in rule.names}
    optional = []
    for column, f in enumerate(factors):
        if f.low != 0 or (f.min_when_used is None and f.name not in counted):
            continue
        minimum = 0.0 if f.min_when_used is None else f.min_when_used
        if coded[f.name]:
            optional.append((column, -1.0, 2.0 * minimum / f.high - 1.0))  # type: ignore[operator]
        else:
            optional.append((column, 0.0, minimum))
    return UsedSets(names, optional, rules)
