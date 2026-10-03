"""Quality floor for the run-budget OMARS search, ``generate_design(design_type="omars_ilp", budget=N)``.

The thresholds below were measured once, from the search as released in 1.97.0
(50 randomized-objective restarts with the HiGHS defaults, ``random_state=42``),
for five and six factors at the budgets 17, 19 and 21 and with no budget.  That
search took about 150 s per six-factor case, so it is not re-run here: the
tests run only the current search, and each of its reported criteria must be
no worse than the recorded value (run count and A-optimality no larger,
D-efficiency no smaller, maximum second-order correlation no larger), within
1e-9.  The design itself may differ.
"""

from __future__ import annotations

import pytest

from process_improve.experiments import Factor, generate_design
from process_improve.experiments.designs_omars import is_omars

# (k, budget): (run count, D-efficiency, A-optimality, max second-order correlation),
# measured from release 1.97.0 with random_state=42.
_BASELINE: dict[tuple[int, int | None], tuple[int, float, float, float]] = {
    (5, 17): (17, 37.89263428733616, 2.675968992248061, 0.5142857142857145),
    (5, 19): (19, 40.98866771570662, 2.3931339977851596, 0.7367883976130072),
    (5, 21): (21, 40.514049041274305, 1.9402930402930394, 0.6546536707079772),
    (5, None): (31, 30.33668717531257, 27.81864842291465, 0.5913123959890828),
    (6, 17): (17, 35.880687192939185, 3.7056478405315607, 0.5142857142857145),
    (6, 19): (19, 35.24000872060751, 3.5583333333333327, 1.0),
    (6, 21): (21, 38.829603008525254, 2.716666666666666, 0.7567874686642697),
    (6, None): (43, 26.780664580077236, 86.14846567293829, 0.4980732105593747),
}
_TOL = 1e-9


def _factors(k: int) -> list[Factor]:
    return [Factor(name=chr(65 + i), low=-1, high=1) for i in range(k)]


@pytest.mark.slow
@pytest.mark.parametrize(("k", "budget"), list(_BASELINE))
def test_budget_search_is_no_worse_than_the_baseline(k: int, budget: int | None) -> None:
    """Every reported criterion of the chosen design is no worse than the 1.97.0 search's."""
    runs, d_efficiency, a_optimality, max_correlation = _BASELINE[k, budget]
    result = generate_design(_factors(k), design_type="omars_ilp", budget=budget, random_state=42)
    meta = result.metadata
    assert is_omars(result.design[result.factor_names].to_numpy(dtype=float))
    assert meta["model_rank"] == meta["model_params"]
    assert meta["n_runs_selected"] <= runs
    assert meta["d_efficiency"] >= d_efficiency - _TOL
    assert meta["a_optimality"] <= a_optimality + _TOL
    assert meta["max_second_order_correlation"] <= max_correlation + _TOL
