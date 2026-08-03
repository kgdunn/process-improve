"""(c) Kevin Dunn, 2010-2026. MIT License.

Designing a descriptive-panel **screening** study: which samples does each
assessor taste, in which order, and is the panel big enough to see the effect?

The rest of this subpackage analyses panel data that already exists. This module
covers the step before that: turning a long list of candidate samples into a
serving plan a sensory lab can execute. Three constraints make a panel screen
different from an ordinary designed experiment, and each one is handled here:

* **Session capacity.** An assessor can only judge a handful of samples in one
  sitting before fatigue and carry-over dominate. When there are more candidates
  than slots, every assessor sees only a *subset*, so the design is an
  **incomplete block design** with the assessor-session as the block.
  :func:`cyclic_block_design` builds the blocks so that replication is equal (or
  as near-equal as the numbers allow) and every pair of candidates is compared
  within a block about equally often - the pairwise **concurrence**, which is
  what decides how precisely two candidates can be compared against each other.

* **Carry-over and position.** The sample tasted before this one changes the
  score of this one, and the first sample in a session is scored differently
  from the last. :func:`williams_design` returns serving orders that are
  balanced for first-order carry-over: every ordered pair of samples occurs
  equally often, so the carry-over effect cannot bias any one sample's mean.

* **Sensitivity.** Panel scores are noisy, so a screen that is too small will
  return "no significant effect" whatever the truth is.
  :func:`detectable_difference` converts a residual standard deviation and a
  panel size into the smallest difference the screen can actually resolve, and
  :func:`required_panelists` inverts it.

:func:`sensory_screening_plan` assembles all three into one serving sheet, with
an optional reference (control) sample anchored in every block so drift between
sessions can be removed at the analysis stage.

The output is deliberately in the same long shape the rest of the subpackage
consumes: once the scores are filled in next to ``product``, the sheet is ready
for :func:`process_improve.sensory.compare_products`.

References
----------
Williams, E. J. (1949). Experimental designs balanced for the estimation of
residual effects of treatments. *Australian Journal of Scientific Research*,
2(2), 149-168.

Cochran, W. G. & Cox, G. M. (1957). *Experimental Designs* (2nd ed.). Wiley.
Chapters 9-11 on incomplete block designs.

Naes, T., Brockhoff, P. B. & Tomic, O. (2010). *Statistics for Sensory and
Consumer Science*. Wiley. Chapter 2 on designing sensory experiments.

MacFie, H. J., Bratchell, N., Greenhoff, K. & Vallis, L. V. (1989). Designs to
balance the effect of order of presentation and first-order carry-over effects
in hall tests. *Journal of Sensory Studies*, 4(2), 129-148.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as _student_t

#: Columns of the serving sheet returned by :func:`sensory_screening_plan`.
SERVING_PLAN_COLUMNS: tuple[str, ...] = (
    "panelist_id",
    "session",
    "position",
    "product",
    "role",
    "block",
)

_MIN_TREATMENTS = 2


@dataclass
class ScreeningPlan:
    """Outcome of :func:`sensory_screening_plan`.

    Attributes
    ----------
    plan : pandas.DataFrame
        The serving sheet: one row per serving, with the columns of
        :data:`SERVING_PLAN_COLUMNS`. ``block`` numbers the assessor-session
        blocks consecutively; ``role`` is ``"control"`` or ``"test"``.
    diagnostics : dict
        Balance read-outs for the plan - ``replication`` (min/max/mean servings
        per candidate), ``concurrence`` (min/max/mean number of blocks in which
        a pair of candidates meet), ``balanced`` (True only for an exact
        balanced incomplete block design), ``control_coverage`` (fraction of
        blocks containing the reference), ``position_balance`` (largest spread
        in how often a candidate lands in any one position) and
        ``n_servings`` / ``n_blocks``.
    config : dict
        The resolved call arguments, for provenance.
    warnings : list of str
        Practical problems that do not stop the plan being produced - most
        importantly a panel too small to cover every candidate.
    """

    plan: pd.DataFrame
    diagnostics: dict[str, Any]
    config: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Serving order: carry-over balance
# ---------------------------------------------------------------------------


def _williams_first_row(n_treatments: int) -> list[int]:
    """Return the generating row ``0, 1, t-1, 2, t-2, ...`` of a Williams square."""
    row = []
    for position in range(n_treatments):
        if position % 2 == 0:
            row.append(position // 2)
        else:
            row.append(n_treatments - (position + 1) // 2)
    return row


def williams_design(
    n_treatments: int,
    *,
    n_subjects: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Build serving orders balanced for first-order carry-over (a Williams design).

    Each returned sequence is a permutation of all ``n_treatments`` treatments.
    Across the full set of sequences, every treatment appears once in every
    position (a Latin square) and every *ordered* pair of treatments occurs
    equally often, so the effect of "what was tasted just before" is spread
    evenly over the treatments instead of favouring some of them.

    An even ``n_treatments`` achieves this with ``n_treatments`` sequences; an
    odd ``n_treatments`` needs a second, mirrored square, so ``2 * n_treatments``
    sequences are returned.

    Parameters
    ----------
    n_treatments : int
        Number of treatments to order, at least 2.
    n_subjects : int, optional
        Number of assessors to produce a sequence for. The balanced set of
        sequences is recycled in order until this many are produced; balance is
        exact only when ``n_subjects`` is a multiple of the number of sequences.
        Defaults to the natural size of the design.
    seed : int, optional
        When given, the *assignment* of sequences to subjects is shuffled. The
        balance properties are unaffected.

    Returns
    -------
    pandas.DataFrame
        Long format with columns ``sequence`` (0-based subject index),
        ``position`` (1-based serving order) and ``treatment`` (0-based
        treatment index).

    Examples
    --------
    >>> design = williams_design(4)
    >>> design.pivot(index="sequence", columns="position", values="treatment").to_numpy()
    array([[0, 3, 1, 2],
           [1, 0, 2, 3],
           [2, 1, 3, 0],
           [3, 2, 0, 1]])
    """
    if n_treatments < _MIN_TREATMENTS:
        msg = f"A carry-over balanced order needs at least 2 treatments; got {n_treatments}."
        raise ValueError(msg)

    first_row = _williams_first_row(n_treatments)
    square = [[(value + shift) % n_treatments for value in first_row] for shift in range(n_treatments)]
    if n_treatments % 2 == 1:
        square += [list(reversed(row)) for row in square]

    sequences = square
    if n_subjects is not None:
        if n_subjects < 1:
            msg = f"n_subjects must be at least 1; got {n_subjects}."
            raise ValueError(msg)
        sequences = [square[i % len(square)] for i in range(n_subjects)]

    if seed is not None:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(sequences))
        sequences = [sequences[i] for i in order]

    records = [
        {"sequence": subject, "position": position + 1, "treatment": treatment}
        for subject, sequence in enumerate(sequences)
        for position, treatment in enumerate(sequence)
    ]
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Incomplete blocks
# ---------------------------------------------------------------------------


def cyclic_block_design(
    n_treatments: int,
    *,
    block_size: int,
    n_blocks: int,
    seed: int | None = None,
) -> list[list[int]]:
    """
    Build ``n_blocks`` incomplete blocks of ``block_size`` treatments each.

    The blocks are filled greedily: each slot takes the treatment that has been
    used least so far, breaking ties in favour of the treatment that has met the
    treatments already in this block the fewest times. That drives the design
    towards equal replication and equal pairwise concurrence, which is what a
    balanced incomplete block design achieves exactly. When an exact BIBD exists
    for the given ``(n_treatments, block_size, n_blocks)`` this construction
    finds one; otherwise it returns the nearest thing the numbers allow, and
    :func:`plan_diagnostics` reports how close that is.

    Parameters
    ----------
    n_treatments : int
        Number of distinct treatments to spread over the blocks.
    block_size : int
        Treatments per block. Must not exceed ``n_treatments``: a block cannot
        contain a treatment twice.
    n_blocks : int
        Number of blocks to build.
    seed : int, optional
        Seed for the tie-breaking shuffle, making the result reproducible.

    Returns
    -------
    list of list of int
        One list of 0-based treatment indices per block.

    Examples
    --------
    >>> blocks = cyclic_block_design(7, block_size=3, n_blocks=7, seed=0)
    >>> plan_diagnostics(blocks, n_treatments=7)["balanced"]
    True
    """
    if n_treatments < _MIN_TREATMENTS:
        msg = f"Need at least 2 treatments to block; got {n_treatments}."
        raise ValueError(msg)
    if block_size < 1 or block_size > n_treatments:
        msg = f"block_size must be between 1 and n_treatments ({n_treatments}); got {block_size}."
        raise ValueError(msg)
    if n_blocks < 1:
        msg = f"n_blocks must be at least 1; got {n_blocks}."
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    replication = np.zeros(n_treatments, dtype=int)
    concurrence = np.zeros((n_treatments, n_treatments), dtype=int)

    blocks: list[list[int]] = []
    for _ in range(n_blocks):
        block: list[int] = []
        for _slot in range(block_size):
            candidates = [t for t in range(n_treatments) if t not in block]
            # Least-used first; then least already-met inside this block; then random.
            jitter = rng.random(n_treatments)
            best = min(
                candidates,
                key=lambda t: (replication[t], sum(concurrence[t, other] for other in block), jitter[t]),
            )
            block.append(best)
        for treatment in block:
            replication[treatment] += 1
        for i, first in enumerate(block):
            for second in block[i + 1 :]:
                concurrence[first, second] += 1
                concurrence[second, first] += 1
        blocks.append(block)

    _balance_by_swapping(blocks, n_treatments=n_treatments, replication=replication, concurrence=concurrence)
    return [sorted(block) for block in blocks]


def _imbalance_penalty(n_treatments: int) -> float:
    """Weight making equal replication strictly more important than equal concurrence."""
    return 10.0 * n_treatments


def _pair_changes(losers: list[tuple[int, list[int]]], gainers: list[tuple[int, list[int]]]) -> dict:
    """Aggregate per-pair concurrence increments for a candidate move.

    ``losers``/``gainers`` are ``(treatment, partners)`` pairs: the treatment
    stops (starts) sharing a block with each of its partners. Aggregating first
    is what makes overlapping blocks safe - a pair touched by both halves of an
    exchange nets out instead of being double-counted.
    """
    changes: dict[tuple[int, int], int] = {}
    for treatment, partners in losers:
        for partner in partners:
            key = (min(treatment, partner), max(treatment, partner))
            changes[key] = changes.get(key, 0) - 1
    for treatment, partners in gainers:
        for partner in partners:
            key = (min(treatment, partner), max(treatment, partner))
            changes[key] = changes.get(key, 0) + 1
    return changes


def _apply_move(
    pair_changes: dict,
    rep_changes: dict,
    *,
    replication: np.ndarray,
    concurrence: np.ndarray,
) -> None:
    """Commit the aggregated increments of a move to the running counters."""
    for treatment, change in rep_changes.items():
        replication[treatment] += change
    for (first, second), change in pair_changes.items():
        concurrence[first, second] += change
        concurrence[second, first] += change


def _move_delta(
    pair_changes: dict,
    rep_changes: dict,
    *,
    replication: np.ndarray,
    concurrence: np.ndarray,
    weight: float,
) -> float:
    """Change in the imbalance objective if this move were applied."""
    delta = 0.0
    for treatment, change in rep_changes.items():
        current = replication[treatment]
        delta += weight * (2 * current * change + change**2)
    for (first, second), change in pair_changes.items():
        current = concurrence[first, second]
        delta += 2 * current * change + change**2
    return delta


def _balance_by_swapping(
    blocks: list[list[int]],
    *,
    n_treatments: int,
    replication: np.ndarray,
    concurrence: np.ndarray,
    max_moves: int = 5_000,
) -> None:
    """
    Improve the greedy blocks in place by hill-climbing on the imbalance.

    The objective is ``w * sum (r_i - rbar)^2 + sum_{i<j} (lambda_ij - lambdabar)^2``:
    equal replication first (hence the large ``w``), then equal pairwise
    concurrence. Two move types are tried, and both are needed:

    * a **substitution** - one treatment in a block is replaced by one not in
      it. This changes replication, so it is the move that levels replication
      out; but once replication *is* level, every substitution makes it worse.
    * an **exchange** - two blocks trade one treatment each. Replication is
      untouched, so this is the move that goes on improving concurrence after
      replication has settled. Without it the search stalls one step short of an
      exact balanced incomplete block design.

    First improving move wins, and deltas are computed incrementally, so this is
    cheap even for a few hundred blocks.
    """
    weight = _imbalance_penalty(n_treatments)
    moves = 0
    while moves < max_moves:
        move = _find_improving_move(
            blocks,
            n_treatments=n_treatments,
            replication=replication,
            concurrence=concurrence,
            weight=weight,
        )
        if move is None:
            return
        pair_changes, rep_changes, commit = move
        _apply_move(pair_changes, rep_changes, replication=replication, concurrence=concurrence)
        commit()
        moves += 1


def _find_improving_move(  # noqa: C901 - the two move types share one scan
    blocks: list[list[int]],
    *,
    n_treatments: int,
    replication: np.ndarray,
    concurrence: np.ndarray,
    weight: float,
) -> tuple[dict, dict, Any] | None:
    """Return the first move that lowers the objective, or None when at a local optimum."""
    for index, block in enumerate(blocks):
        for out_treatment in block:
            others = [t for t in block if t != out_treatment]
            for in_treatment in range(n_treatments):
                if in_treatment in block:
                    continue
                pair_changes = _pair_changes([(out_treatment, others)], [(in_treatment, others)])
                rep_changes = {out_treatment: -1, in_treatment: +1}
                delta = _move_delta(
                    pair_changes,
                    rep_changes,
                    replication=replication,
                    concurrence=concurrence,
                    weight=weight,
                )
                if delta < 0:
                    target, leaving, entering = block, out_treatment, in_treatment

                    def commit(target: list[int] = target, leaving: int = leaving, entering: int = entering) -> None:
                        target[target.index(leaving)] = entering

                    return pair_changes, rep_changes, commit

        for other_index in range(index + 1, len(blocks)):
            other_block = blocks[other_index]
            for first in block:
                if first in other_block:
                    continue
                rest_here = [t for t in block if t != first]
                for second in other_block:
                    if second in block:
                        continue
                    rest_there = [t for t in other_block if t != second]
                    pair_changes = _pair_changes(
                        [(first, rest_here), (second, rest_there)],
                        [(second, rest_here), (first, rest_there)],
                    )
                    delta = _move_delta(
                        pair_changes,
                        {},
                        replication=replication,
                        concurrence=concurrence,
                        weight=weight,
                    )
                    if delta < 0:
                        here, there, a, b = block, other_block, first, second

                        def commit_exchange(
                            here: list[int] = here,
                            there: list[int] = there,
                            a: int = a,
                            b: int = b,
                        ) -> None:
                            here[here.index(a)] = b
                            there[there.index(b)] = a

                        return pair_changes, {}, commit_exchange
    return None


def plan_diagnostics(blocks: list[list[int]], *, n_treatments: int) -> dict[str, Any]:
    """
    Summarise how balanced a set of blocks is.

    Parameters
    ----------
    blocks : list of list of int
        The blocks, as 0-based treatment indices (the output of
        :func:`cyclic_block_design`).
    n_treatments : int
        Total number of treatments the blocks were drawn from. Treatments that
        appear in no block are counted with a replication of zero, so a panel
        too small to cover the candidate list shows up here.

    Returns
    -------
    dict
        ``replication`` and ``concurrence`` (each with ``min``, ``max``,
        ``mean``), ``balanced`` (True when replication and off-diagonal
        concurrence are both constant, i.e. an exact BIBD), ``n_blocks`` and
        ``n_servings``.
    """
    replication = np.zeros(n_treatments, dtype=int)
    concurrence = np.zeros((n_treatments, n_treatments), dtype=int)
    for block in blocks:
        for treatment in block:
            replication[treatment] += 1
        for i, first in enumerate(block):
            for second in block[i + 1 :]:
                concurrence[first, second] += 1
                concurrence[second, first] += 1

    off_diagonal = concurrence[~np.eye(n_treatments, dtype=bool)]
    balanced = bool(replication.min() == replication.max() and off_diagonal.min() == off_diagonal.max())
    return {
        "replication": {
            "min": int(replication.min()),
            "max": int(replication.max()),
            "mean": float(replication.mean()),
        },
        "concurrence": {
            "min": int(off_diagonal.min()),
            "max": int(off_diagonal.max()),
            "mean": float(off_diagonal.mean()),
        },
        "balanced": balanced,
        "n_blocks": len(blocks),
        "n_servings": int(replication.sum()),
    }


# ---------------------------------------------------------------------------
# The assembled serving plan
# ---------------------------------------------------------------------------


def _position_balance(plan: pd.DataFrame) -> float:
    """Largest spread (max - min) in how often any candidate lands in one position."""
    tests = plan.loc[plan["role"] == "test"]
    if tests.empty:
        return 0.0
    counts = tests.pivot_table(index="product", columns="position", aggfunc="size", values="block").fillna(0)
    return float((counts.max(axis=1) - counts.min(axis=1)).max())


def _validate_screening_inputs(  # noqa: PLR0913 - one guard per screening-plan knob
    products: list[str],
    *,
    n_panelists: int,
    samples_per_session: int,
    control: str | None,
    replicates: int,
    n_sessions: int | None,
) -> None:
    """Reject inputs that cannot produce a runnable serving plan."""
    if len(products) != len(set(products)):
        msg = "The candidate list contains duplicate labels; each sample must appear once."
        raise ValueError(msg)
    if len(products) < _MIN_TREATMENTS:
        msg = f"Screening needs at least 2 candidates; got {len(products)}."
        raise ValueError(msg)
    if control is not None and control in products:
        msg = f"The control {control!r} must not also appear in the candidate list."
        raise ValueError(msg)
    if n_panelists < 1:
        msg = f"n_panelists must be at least 1; got {n_panelists}."
        raise ValueError(msg)
    if replicates < 1:
        msg = f"replicates must be at least 1; got {replicates}."
        raise ValueError(msg)
    if n_sessions is not None and n_sessions < 1:
        msg = f"n_sessions must be at least 1; got {n_sessions}."
        raise ValueError(msg)
    if samples_per_session - (1 if control is not None else 0) < 1:
        msg = (
            f"samples_per_session={samples_per_session} leaves no room for a candidate once the "
            "control is anchored; increase it or drop the control."
        )
        raise ValueError(msg)


def _capacity_warnings(*, n_test: int, n_blocks: int, slots_per_block: int, replicates: int) -> list[str]:
    """Flag a panel too small to cover the candidate list at the requested replication."""
    capacity = n_blocks * slots_per_block
    if capacity < n_test:
        return [
            f"The panel cannot cover the candidate list: {n_blocks} blocks x {slots_per_block} slots = "
            f"{capacity} servings for {n_test} candidates. Add sessions or assessors, or shorten the list."
        ]
    if capacity < n_test * replicates:
        return [
            f"Capacity ({capacity} servings) is below the requested {n_test * replicates} "
            f"(= {n_test} candidates x {replicates} replicates); realised replication will be lower."
        ]
    return []


def _serving_records(
    blocks: list[list[int]],
    *,
    products: list[str],
    control: str | None,
    n_panelists: int,
    orders: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """Expand the blocks into one record per serving, in the order they are served."""
    records: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks):
        panelist = f"P{block_index % n_panelists + 1:02d}"
        session = block_index // n_panelists + 1
        if orders is None:
            ordered = list(block)
        else:
            sequence = orders.loc[orders["sequence"] == block_index].sort_values("position")["treatment"]
            ordered = [block[i] for i in sequence]

        offset = 0
        if control is not None:
            records.append(
                {
                    "panelist_id": panelist,
                    "session": session,
                    "position": 1,
                    "product": control,
                    "role": "control",
                    "block": block_index + 1,
                }
            )
            offset = 1
        records.extend(
            {
                "panelist_id": panelist,
                "session": session,
                "position": position + 1 + offset,
                "product": products[treatment],
                "role": "test",
                "block": block_index + 1,
            }
            for position, treatment in enumerate(ordered)
        )
    return records


def sensory_screening_plan(  # noqa: PLR0913 - each argument is a distinct, explicit design knob
    products: list[str],
    *,
    n_panelists: int,
    samples_per_session: int,
    control: str | None = None,
    replicates: int = 1,
    n_sessions: int | None = None,
    seed: int | None = None,
) -> ScreeningPlan:
    """
    Build a blocked, carry-over balanced serving plan for a panel screen.

    Each assessor-session is one **block**. When the candidate list is longer
    than a session can hold, the blocks are incomplete: an assessor sees a
    subset, chosen by :func:`cyclic_block_design` so that replication and
    pairwise concurrence stay as even as the arithmetic allows. Within a block
    the serving order comes from :func:`williams_design`, so the sample tasted
    beforehand does not systematically favour any candidate.

    Parameters
    ----------
    products : list of str
        The candidate samples to screen. Must be unique and must not include
        ``control``.
    n_panelists : int
        Number of assessors on the panel.
    samples_per_session : int
        How many samples one assessor can judge in one session. When a
        ``control`` is anchored it occupies one of these slots.
    control : str, optional
        Label of a reference sample served in every block, at the first
        position. Anchoring a reference lets the analysis remove session and
        assessor drift; leave it out to use the whole capacity for candidates.
    replicates : int
        Target number of times each candidate is served across the whole plan.
        More sessions are scheduled until this is met, so the realised
        replication is at least this and never more than one above the minimum.
    n_sessions : int, optional
        Fix the number of sessions per assessor instead of deriving it from
        ``replicates``. When the fixed number cannot cover every candidate this
        is reported in ``warnings`` rather than raising.
    seed : int, optional
        Seed making the plan reproducible.

    Returns
    -------
    ScreeningPlan
        The serving sheet plus balance diagnostics; see :class:`ScreeningPlan`.

    Examples
    --------
    >>> plan = sensory_screening_plan(
    ...     [f"C{i}" for i in range(1, 22)],
    ...     n_panelists=12,
    ...     samples_per_session=6,
    ...     control="Base",
    ...     replicates=2,
    ...     seed=0,
    ... )
    >>> plan.diagnostics["control_coverage"]
    1.0
    """
    _validate_screening_inputs(
        products,
        n_panelists=n_panelists,
        samples_per_session=samples_per_session,
        control=control,
        replicates=replicates,
        n_sessions=n_sessions,
    )

    n_test = len(products)
    test_slots = samples_per_session - (1 if control is not None else 0)
    slots_per_block = min(test_slots, n_test)

    if n_sessions is None:
        blocks_needed = int(np.ceil(n_test * replicates / slots_per_block))
        n_sessions = max(1, int(np.ceil(blocks_needed / n_panelists)))

    n_blocks = n_panelists * n_sessions
    warnings = _capacity_warnings(
        n_test=n_test, n_blocks=n_blocks, slots_per_block=slots_per_block, replicates=replicates
    )

    blocks = cyclic_block_design(n_test, block_size=slots_per_block, n_blocks=n_blocks, seed=seed)
    orders = williams_design(slots_per_block, n_subjects=n_blocks, seed=seed) if slots_per_block > 1 else None

    records = _serving_records(blocks, products=products, control=control, n_panelists=n_panelists, orders=orders)
    plan = pd.DataFrame.from_records(records, columns=list(SERVING_PLAN_COLUMNS))
    plan = plan.sort_values(["block", "position"], ignore_index=True)

    diagnostics = plan_diagnostics(blocks, n_treatments=n_test)
    diagnostics["control_coverage"] = 1.0 if control is not None else 0.0
    diagnostics["position_balance"] = _position_balance(plan)
    diagnostics["n_sessions"] = n_sessions
    diagnostics["slots_per_block"] = slots_per_block

    config = {
        "n_products": n_test,
        "n_panelists": n_panelists,
        "samples_per_session": samples_per_session,
        "control": control,
        "replicates": replicates,
        "n_sessions": n_sessions,
        "seed": seed,
    }
    return ScreeningPlan(plan=plan, diagnostics=diagnostics, config=config, warnings=warnings)


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------


def detectable_difference(
    *,
    sd: float,
    n_per_product: int,
    alpha: float = 0.05,
    power: float = 0.80,
    n_comparisons: int = 1,
) -> dict[str, Any]:
    """
    Smallest difference between two samples the screen can resolve.

    Uses the usual two-sample normal-theory approximation with Student-t
    quantiles, ``delta = (t_{1 - alpha', df} + t_{power, df}) * sd * sqrt(2 / n)``
    where ``df = 2 * (n - 1)`` and ``alpha' = alpha / (2 * n_comparisons)`` is
    the Bonferroni-adjusted two-sided level. Because it uses the *residual*
    standard deviation of the blocked model, the assessor-to-assessor variation
    that blocking removes is already excluded, as it should be.

    This is an approximation, and a deliberately honest one: quote it as the
    order of magnitude the screen can see, not a guarantee.

    Parameters
    ----------
    sd : float
        Residual standard deviation of a single score, on the scale the panel
        uses. Take it from a previous study on the same attribute and scale.
    n_per_product : int
        Number of independent scores contributing to each sample's mean
        (assessors x replicates), at least 2.
    alpha : float
        Two-sided significance level before any multiplicity adjustment.
    power : float
        Probability of detecting a difference of exactly this size.
    n_comparisons : int
        Size of the comparison family to Bonferroni-correct for; for example the
        number of candidates each being compared against a single control.

    Returns
    -------
    dict
        ``difference`` (the minimum detectable difference), plus the ``sd``,
        ``n_per_product``, ``alpha``, ``power`` and ``n_comparisons`` used, so
        the number can be quoted with its assumptions attached.

    Examples
    --------
    >>> round(detectable_difference(sd=1.5, n_per_product=12)["difference"], 3)
    1.796
    """
    if n_per_product < _MIN_TREATMENTS:
        msg = f"n_per_product must be at least 2 to leave error degrees of freedom; got {n_per_product}."
        raise ValueError(msg)
    if sd <= 0:
        msg = f"sd must be positive; got {sd}."
        raise ValueError(msg)
    if not 0 < alpha < 1:
        msg = f"alpha must be strictly between 0 and 1; got {alpha}."
        raise ValueError(msg)
    if not 0 < power < 1:
        msg = f"power must be strictly between 0 and 1; got {power}."
        raise ValueError(msg)
    if n_comparisons < 1:
        msg = f"n_comparisons must be at least 1; got {n_comparisons}."
        raise ValueError(msg)

    df = 2 * (n_per_product - 1)
    t_alpha = float(_student_t.ppf(1 - alpha / (2 * n_comparisons), df))
    t_beta = float(_student_t.ppf(power, df))
    difference = (t_alpha + t_beta) * sd * np.sqrt(2 / n_per_product)
    return {
        "difference": float(difference),
        "sd": float(sd),
        "n_per_product": int(n_per_product),
        "alpha": float(alpha),
        "power": float(power),
        "n_comparisons": int(n_comparisons),
    }


def required_panelists(  # noqa: PLR0913 - explicit sizing knobs, each distinct
    *,
    sd: float,
    difference: float,
    alpha: float = 0.05,
    power: float = 0.80,
    n_comparisons: int = 1,
    max_n: int = 10_000,
) -> dict[str, Any]:
    """
    Smallest ``n_per_product`` whose detectable difference reaches a target.

    Inverts :func:`detectable_difference` by search, so the two always agree.

    Parameters
    ----------
    sd : float
        Residual standard deviation of a single score.
    difference : float
        The difference that must be detectable, on the same scale as ``sd``.
    alpha, power, n_comparisons
        As in :func:`detectable_difference`.
    max_n : int
        Search ceiling; exceeding it raises rather than looping forever.

    Returns
    -------
    dict
        ``n_per_product`` (the answer) and ``achieved_difference`` (what that n
        actually resolves, which is at or below ``difference``), plus the
        inputs.

    Examples
    --------
    >>> required_panelists(sd=1.5, difference=1.0)["n_per_product"]
    37
    """
    if difference <= 0:
        msg = f"difference must be positive; got {difference}."
        raise ValueError(msg)

    for n in range(2, max_n + 1):
        achieved = detectable_difference(sd=sd, n_per_product=n, alpha=alpha, power=power, n_comparisons=n_comparisons)[
            "difference"
        ]
        if achieved <= difference:
            return {
                "n_per_product": n,
                "achieved_difference": achieved,
                "sd": float(sd),
                "difference": float(difference),
                "alpha": float(alpha),
                "power": float(power),
                "n_comparisons": int(n_comparisons),
            }

    msg = f"No n_per_product up to {max_n} reaches a detectable difference of {difference} with sd={sd}."
    raise ValueError(msg)
