# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``search_design_catalog``, the design catalog's search."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec


class SearchDesignCatalogInput(BaseModel):
    """Input contract for ``search_design_catalog``."""

    model_config = ConfigDict(extra="forbid")

    min_factors: int | None = Field(default=None, ge=1, le=60, description="The fewest factors.")
    max_factors: int | None = Field(default=None, ge=1, le=60, description="The most factors.")
    min_runs: int | None = Field(default=None, ge=1, description="The fewest runs.")
    max_runs: int | None = Field(default=None, ge=1, description="The most runs: the user's budget.")
    model: Literal["main_effects", "interactions", "quadratic"] | None = Field(
        default=None,
        description=(
            "A model the design must estimate in full: 'main_effects', 'interactions' (main effects and "
            "two-factor interactions) or 'quadratic' (the full second-order model, with squares)."
        ),
    )
    min_categorical: int | None = Field(default=None, ge=0, description="The fewest categorical factors.")
    max_categorical: int | None = Field(
        default=None, ge=0, description="The most categorical factors; 0 for designs with numeric factors only."
    )
    families: list[str] | None = Field(
        default=None,
        description=(
            "Design families to keep, such as 'full_factorial', 'fractional_factorial', 'plackett_burman', "
            "'dsd', 'omars', 'ccd', 'box_behnken', 'optimal', 'mixture', 'orthogonal_array', 'supersaturated', "
            "'latin_square', 'block_design', 'split_plot'."
        ),
    )
    topics: list[str] | None = Field(
        default=None,
        description=(
            "Applications (industry sectors) to keep: a design passes when it lists any of them. Such as "
            "'food-beverages', 'semiconductors', 'chemical-processes', 'pharmaceuticals', 'bioprocessing', "
            "'plastics-rubber', 'metals', 'machining', 'automotive', 'aerospace-defence', 'environment', "
            "'measurement', 'business', or 'general' (no stated application); an unknown one is refused with "
            "the full list. A generated design has no application, so any application keeps published designs only."
        ),
    )
    collection: Literal["both", "published", "generated"] = Field(
        default="both",
        description="Designs printed in a published source, designs the library makes, or both.",
    )
    has_results: bool = Field(
        default=False, description="Keep only published experiments printed with their results, ready to analyse."
    )
    survives_run_loss: bool = Field(
        default=False,
        description="Keep only designs that still estimate their model if any one run is lost or fails.",
    )
    min_quadratic_projection: int | None = Field(
        default=None,
        ge=1,
        le=12,
        description=(
            "Keep only designs whose every projection onto this many factors estimates the full quadratic in "
            "them: if only that many factors matter, their response surface needs no more runs."
        ),
    )
    min_projectivity: int | None = Field(
        default=None, ge=1, le=12, description="Keep only two-level designs of at least this projectivity."
    )
    year_from: int | None = Field(default=None, description="The earliest year of the source.")
    year_to: int | None = Field(default=None, description="The latest year of the source.")
    text: str = Field(
        default="",
        max_length=200,
        description=(
            "Words to find in the design's title, family, source or applications, such as an author or 'pilot plant'."
        ),
    )
    sort: Literal["runs", "factors", "newest"] = Field(
        default="runs", description="The order: fewest runs, fewest factors, or the most recent source first."
    )
    limit: int = Field(default=10, ge=1, le=50, description="The most designs to return.")


@tool_spec(
    name="search_design_catalog",
    description=(
        "Search the design catalog: the designed experiments printed in textbooks, handbooks and papers "
        "(Box, Hunter and Hunter; the NIST e-Handbook; Jones and Nachtsheim; Nunez Ares and Goos; and others) "
        "and the designs process-improve makes, the same records the catalog at https://factori.al/catalog/ "
        "shows. Filter by factors, runs, the model a design estimates in full, categorical factors, family, "
        "application (industry sector), whether it survives losing a run, its projections, its source's year "
        "and words in its title. Each design returned carries its source and page (with the address of the "
        "section that prints it, for a source on the web), its applications, what it estimates and "
        "withstands, how many other entries are the same design, and the address of its page in the "
        "catalog, which opens it in the browser app. Use this tool to find a published design or a published "
        "experiment to learn from, or to compare the families that fit a user's factors and budget. To make "
        "a design, use generate_design; to measure one, use evaluate_design. The first search builds the "
        "catalog, which takes some seconds."
    ),
    input_model=SearchDesignCatalogInput,
    examples="""
    # "Which designs fit 6 factors in at most 17 runs and still give a response surface in any 3?"
        -> ``search_design_catalog(min_factors=6, max_factors=6, max_runs=17, min_quadratic_projection=3)``

    # "Show me a published response-surface experiment with real data I can analyse."
        -> ``search_design_catalog(model="quadratic", has_results=True, collection="published")``

    # "Which screening designs for 7 factors can lose a run and still estimate the main effects?"
        -> ``search_design_catalog(min_factors=7, max_factors=7, survives_run_loss=True, sort="runs")``

    # "Has anyone published a designed experiment on food, with the data?"
        -> ``search_design_catalog(topics=["food-beverages"], has_results=True)``
    """,
    category="experiments",
)
def search_design_catalog_tool(spec: SearchDesignCatalogInput) -> dict[str, Any]:
    """Return the designs of the catalog that pass every filter, the fewest runs first by default."""
    try:
        from process_improve.experiments.design_catalog import search_catalog  # noqa: PLC0415

        filters = spec.model_dump(exclude={"limit"})
        found = search_catalog(**filters)
        return clean({"count": len(found), "shown": min(len(found), spec.limit), "designs": found[: spec.limit]})
    except _TOOL_EXPECTED_EXCEPTIONS as exc:
        logger.exception("Tool search_design_catalog failed")
        return {"error": str(exc)}


_register("search_design_catalog")
