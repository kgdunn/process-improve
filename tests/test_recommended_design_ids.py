"""Every design the knowledge base or the strategy planner recommends is one generate_design can build.

A recommended id is either a ``generate_design`` design type (a key of ``_DESIGN_REGISTRY``)
or one of the ids that ``NON_GENERATE_DESIGN_IDS`` documents as something else.
"""

from __future__ import annotations

import pytest

from process_improve.experiments.designs import _DESIGN_REGISTRY
from process_improve.experiments.factor import Constraint, Factor, Response
from process_improve.experiments.knowledge.engine import reload_knowledge_graph
from process_improve.experiments.strategy.domain_templates import DOMAIN_TEMPLATES
from process_improve.experiments.strategy.engine import recommend_strategy
from process_improve.experiments.strategy.models import NON_GENERATE_DESIGN_IDS, DomainType

_KNOWN = set(_DESIGN_REGISTRY) | set(NON_GENERATE_DESIGN_IDS)


def _knowledge_base_ids() -> dict[str, str]:
    """Each design id in the knowledge base, mapped to where it appears."""
    graph = reload_knowledge_graph()
    ids: dict[str, str] = {}
    for design in graph.design_types.values():
        ids.setdefault(design.id, "design_types.yaml id")
        for target in design.can_augment_to:
            ids.setdefault(target["target"], f"design_types.yaml {design.id} can_augment_to")
    for rule in graph.decision_rules:
        ids.setdefault(rule.recommend["primary"], f"decision_rules.yaml {rule.id} primary")
        for alternative in rule.recommend.get("alternatives", []):
            ids.setdefault(alternative["design"], f"decision_rules.yaml {rule.id} alternative")
    return ids


def _continuous(k: int) -> list[Factor]:
    return [Factor(name=f"X{i}", low=0, high=1) for i in range(k)]


def _planner_scenarios() -> list[dict]:
    responses = [Response(name="y", goal="maximize")]
    scenarios: list[dict] = [
        {"factors": _continuous(k), "responses": responses, "budget": budget}
        for k in (1, 2, 3, 5, 6, 8, 12)
        for budget in (None, 6, 12, 20, 40)
    ]
    scenarios += [
        {"factors": _continuous(4), "responses": responses, "constraints": [Constraint(expression="X0 + X1 <= 1.5")]},
        {"factors": _continuous(4), "responses": responses, "hard_to_change_factors": ["X0"]},
        {"factors": [Factor(name="C", type="categorical", levels=["a", "b", "c"]), *_continuous(3)]},
        {"factors": [Factor(name=f"M{i}", type="mixture", low=0, high=1) for i in range(4)], "responses": responses},
        {"factors": _continuous(6), "responses": responses, "prior_knowledge": "Published: X0 and X1 are significant."},
    ]
    return scenarios


def test_the_knowledge_base_recommends_buildable_designs() -> None:
    unknown = {design_id: where for design_id, where in _knowledge_base_ids().items() if design_id not in _KNOWN}
    assert not unknown, f"Not generate_design design types: {unknown}"


def test_the_domain_preferences_name_buildable_designs() -> None:
    preferences = {
        (domain, key): template[key]
        for domain, template in DOMAIN_TEMPLATES.items()
        for key in ("screening_preference", "rsm_preference")
        if template[key] is not None
    }
    unknown = {where: value for where, value in preferences.items() if value not in _KNOWN}
    assert not unknown


@pytest.mark.parametrize("domain", [d.value for d in DomainType])
def test_the_strategy_planner_recommends_buildable_designs(domain: str) -> None:
    for inputs in _planner_scenarios():
        for stage in recommend_strategy(**inputs, domain=domain)["stages"]:
            assert stage["design_type"] in _KNOWN, (inputs, stage["design_type"])


def test_every_non_generate_design_id_is_documented() -> None:
    assert not set(NON_GENERATE_DESIGN_IDS) & set(_DESIGN_REGISTRY)
    assert all(description.strip() for description in NON_GENERATE_DESIGN_IDS.values())
