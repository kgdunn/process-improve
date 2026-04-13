# (c) Kevin Dunn, 2010-2026. MIT License.

"""Multi-stage experimental strategy recommender.

Given a DOE problem specification (factors, responses, budget, constraints,
domain, prior knowledge), recommend a multi-stage experimental strategy
using deterministic decision rules from Montgomery, NIST, and Stat-Ease SCOR.

Quick start::

    from process_improve.experiments.strategy import recommend_strategy

    result = recommend_strategy(
        factors=[Factor(name="A", low=0, high=100), ...],
        responses=[Response(name="Yield", goal="maximize")],
        budget=40,
        domain="fermentation",
    )
"""

from process_improve.experiments.strategy.engine import recommend_strategy

__all__ = ["recommend_strategy"]
