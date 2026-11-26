# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``query_concept_graph``, the book's graph of design terms, queried in Cypher."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec


class QueryConceptGraphInput(BaseModel):
    """Input contract for ``query_concept_graph``."""

    model_config = ConfigDict(extra="forbid")

    cypher: str = Field(
        "",
        max_length=2000,
        description=(
            "One read-only Cypher statement. Leave empty to get the schema first: the node labels, "
            "the link types, the properties of a term, and example queries."
        ),
    )
    parameters: dict[str, Any] | None = Field(
        None,
        description="Values for the statement's $name parameters, e.g. {'id': 'curvature'}.",
    )
    max_rows: int = Field(50, ge=1, le=500, description="The most rows to return.")


@tool_spec(
    name="query_concept_graph",
    description=(
        "Query the textbook's graph of design-of-experiments terms in Cypher: 237 terms (concepts, "
        "methods, design types, diagnostics, guidelines) and 490 links between them (USES, IS_A, "
        "REMEDIED_BY, CAUSES, REQUIRES, CONTRASTS_WITH, ...). Every term and link carries `evidence`, "
        "the line of the book it comes from, so an answer can cite its source. "
        "Use it for questions about how ideas connect: what remedies a symptom, what a method needs, "
        "how two terms are related, which designs have a property. "
        "Call it with no cypher first to read the schema and example queries. The graph is read-only."
    ),
    input_model=QueryConceptGraphInput,
    examples="""
    # "What does the graph hold?"
        -> ``query_concept_graph()``

    # "What fixes curvature, and where does the book say so?"
        -> ``query_concept_graph(
                cypher="MATCH (:Term {id: $id})-[link:REMEDIED_BY]->(fix:Term) "
                       "RETURN fix.name AS remedy, link.evidence AS book_line",
                parameters={"id": "curvature"})``

    # "Find the term for an outcome variable"
        -> ``query_concept_graph(
                cypher="CALL db.index.fulltext.queryNodes('term_text', 'outcome') YIELD node "
                       "RETURN node.id AS id, node.name AS name LIMIT 5")``
    """,
    category="experiments",
)
def query_concept_graph_tool(spec: QueryConceptGraphInput) -> dict[str, Any]:
    """Run one read-only Cypher statement on the book's graph of terms."""
    try:
        from process_improve.experiments.knowledge.concept_graph import ConceptGraph  # noqa: PLC0415

        # One graph per call: relate ties a connection to the thread that opened it, and a graph
        # that lives for one call cannot carry anything from one caller to the next.
        # shortcut: relate's Python API has no query timeout, so a statement that pairs every term
        # with every other several times over runs until it ends; a hosted server stops it with
        # tool_safety's timeout. Pass relate's own limit here once its Python API has one.
        with ConceptGraph() as graph:
            if not spec.cypher.strip():
                return clean({"schema": graph.schema(), "source": graph.source})
            rows = graph.query(spec.cypher, spec.parameters)
            return clean(
                {
                    "rows": rows[: spec.max_rows],
                    "n_rows": len(rows),
                    "truncated": len(rows) > spec.max_rows,
                    "source": graph.source,
                }
            )
    except ImportError as e:  # relate is not installed: say how to install it
        return {"error": str(e)}
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool query_concept_graph failed")
        return {"error": str(e)}


_register("query_concept_graph")
