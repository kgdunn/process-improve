"""Tests for the reusable analysis-recipe framework and the sensory recipes."""

from __future__ import annotations

import pytest

from process_improve.recipes import (
    AnalysisRecipe,
    RecipeStep,
    get_recipe,
    list_recipes,
    register_recipe,
    select_recipe,
)
from process_improve.tool_spec import discover_tools, execute_tool_call, get_tool_specs


def test_to_payload_shape_and_conditional_keys():
    recipe = AnalysisRecipe(
        key="demo_recipe",
        title="Demo",
        summary="A demo.",
        domain="testing",
        cue_phrases=["demo cue"],
        inputs_needed=["nothing"],
        stages=[
            RecipeStep(order=1, directive="Do a thing.", tools=["some_tool"], arg_hints={"x": "1"}),
            RecipeStep(order=2, directive="Interpret it."),  # prose-only
        ],
    )
    payload = recipe.to_payload()
    assert payload["recipe_key"] == "demo_recipe"
    assert payload["status"] == "available"
    assert "note" not in payload  # only planned recipes carry a note
    first, second = payload["stages"]
    assert first["step"] == 1
    assert first["tools"] == ["some_tool"]
    assert first["arg_hints"] == {"x": "1"}
    # The prose-only step omits the empty tools / arg_hints keys entirely.
    assert "tools" not in second
    assert "arg_hints" not in second


def test_planned_recipe_payload_carries_note():
    planned = AnalysisRecipe(
        key="planned_demo",
        title="Planned",
        summary="Later.",
        domain="testing",
        cue_phrases=[],
        inputs_needed=[],
        stages=[],
        status="planned",
    )
    payload = planned.to_payload()
    assert payload["status"] == "planned"
    assert payload["stages"] == []
    assert "note" in payload


def test_register_recipe_rejects_duplicate_key():
    recipe = AnalysisRecipe(
        key="sensory_intake",  # already registered by the sensory recipes module
        title="x",
        summary="x",
        domain="testing",
        cue_phrases=[],
        inputs_needed=[],
        stages=[],
    )
    list_recipes()  # ensure discovery has registered the sensory recipes
    with pytest.raises(ValueError, match="already registered"):
        register_recipe(recipe)


def test_sensory_recipes_are_discovered():
    keys = {r.key for r in list_recipes()}
    assert {
        "sensory_intake",
        "sensory_panel_processing",
        "sensory_relate_covariates",
        "sensory_visualisation",
    } <= keys
    assert get_recipe("sensory_visualisation").status == "planned"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("I have a wide panel spreadsheet to load", "sensory_intake"),
        ("are my panelists consistent and using the scale the same way", "sensory_panel_processing"),
        ("what chemistry drives sweetness", "sensory_relate_covariates"),
        ("make me a perceptual map biplot", "sensory_visualisation"),
    ],
)
def test_select_recipe_matches(query: str, expected: str):
    matched = select_recipe(query)
    assert matched is not None
    assert matched.key == expected


def test_select_recipe_returns_none_for_unrelated_query():
    assert select_recipe("totally unrelated request about widgets") is None


def test_recipe_steps_only_reference_registered_tools():
    discover_tools()
    tool_names = {spec["name"] for spec in get_tool_specs()}
    for recipe in list_recipes():
        for step in recipe.stages:
            for tool in step.tools:
                assert tool in tool_names, f"{recipe.key} step {step.order} names unknown tool {tool!r}"


def test_select_analysis_recipe_tool():
    out = execute_tool_call("select_analysis_recipe", {"query": "what drives liking from the measurements"})
    assert out["ok"] is True
    assert out["matched"] is True
    assert out["recipe"]["recipe_key"] == "sensory_relate_covariates"
    keys = {entry["recipe_key"] for entry in out["available"]}
    assert "sensory_intake" in keys
    assert "sensory_visualisation" in keys

    miss = execute_tool_call("select_analysis_recipe", {"query": "completely off-topic widget chatter"})
    assert miss["matched"] is False
    assert miss["recipe"] is None
    assert miss["available"]  # the catalogue is always offered


def test_select_analysis_recipe_rejects_unknown_kwargs():
    from process_improve.tool_spec import ToolInputInvalidError

    with pytest.raises(ToolInputInvalidError):
        execute_tool_call("select_analysis_recipe", {"query": "x", "surprise": True})
