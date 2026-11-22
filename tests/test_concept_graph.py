"""Tests of the book's graph of design terms in relate: ``ConceptGraph`` and the tool ``query_concept_graph``.

relate is not installed with the package, so the tests that open a graph skip without it; the
tests of the data file, of the terms behind each action and of the missing-package message do not.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from process_improve.experiments import Factor, recommend_next_step
from process_improve.experiments.knowledge import ConceptGraph
from process_improve.experiments.knowledge import concept_graph as module
from process_improve.experiments.next_step import ACTIONS
from process_improve.tool_spec import execute_tool_call, get_tool_specs

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads(module._DATA_FILE.read_text(encoding="utf-8"))
TERM_IDS = {term["id"] for term in DATA["nodes"]}

#: A reply of ``recommend_next_step`` cut down to what ``record_step`` reads.
CURVED = {
    "action": "augment_to_rsm",
    "headline": "Add 6 axial and centre runs.",
    "reasons": ["The centre runs are off the plane of the corners (curvature test p = 1.3e-08)."],
    "n_runs": 6,
    "evidence": {"curvature": {"significant": True}, "lack_of_fit": {"significant": False}, "aliasing": []},
}


@pytest.fixture
def graph() -> Iterator[ConceptGraph]:
    """Return the read-only graph of the book's terms."""
    pytest.importorskip("relate")
    with ConceptGraph() as opened:
        yield opened


@pytest.fixture
def campaign(tmp_path: Path) -> Iterator[ConceptGraph]:
    """Return a graph that keeps a study's steps in a file."""
    pytest.importorskip("relate")
    with ConceptGraph(tmp_path / "campaign.rl") as opened:
        yield opened


# --- the data the package ships ---------------------------------------------------------------


def test_the_packaged_graph_is_the_one_the_concept_map_draws() -> None:
    """One graph, built in pid-book, is in the wheel and in the browser app: the two copies never differ."""
    packaged = module._DATA_FILE.read_bytes()
    drawn = (ROOT / "web" / "map" / "graph.json").read_bytes()
    assert packaged == drawn, (
        "copy web/map/graph.json to src/process_improve/experiments/knowledge/data/concept_graph.json"
    )


def test_every_action_of_recommend_next_step_names_terms_the_book_has() -> None:
    """A new action, or a term the book renames, fails here rather than leaving a step with no link."""
    assert set(module.ACTION_TERMS) == set(ACTIONS)
    named = {term for terms in module.ACTION_TERMS.values() for term in terms}
    assert named | {"curvature", "lack_of_fit_test", "confounding"} <= TERM_IDS


def test_without_relate_the_graph_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing package is an ImportError with the command, and an error envelope from the tool."""
    monkeypatch.setitem(sys.modules, "relate", None)  # makes ``import relate`` raise ImportError
    with pytest.raises(ImportError, match="pip install relate-db"):
        ConceptGraph()
    assert "pip install relate-db" in execute_tool_call("query_concept_graph", {})["error"]


# --- the book's terms, read-only --------------------------------------------------------------


def test_the_graph_holds_every_term_and_link_of_the_file(graph: ConceptGraph) -> None:
    """Each term carries its type as a label too, and nothing in the file is left out."""
    schema = graph.schema()
    assert schema["labels"]["Term"] == len(DATA["nodes"])
    assert schema["labels"]["DesignType"] == sum(term["type"] == "DesignType" for term in DATA["nodes"])
    assert sum(schema["links"].values()) == len(DATA["edges"])
    assert schema["term_properties"] == list(module.TERM_PROPERTIES)
    assert graph.source["commit"] == DATA["source"]["commit"]


def test_a_link_carries_the_line_of_the_book_that_states_it(graph: ConceptGraph) -> None:
    """The answer to "what remedies curvature" cites where the book says so."""
    rows = graph.query(
        "MATCH (:Term {id: $id})-[link:REMEDIED_BY]->(fix:Term) RETURN fix.id AS remedy, link.evidence AS book_line",
        {"id": "curvature"},
    )
    stated = next(e for e in DATA["edges"] if (e["source"], e["type"]) == ("curvature", "REMEDIED_BY"))
    assert rows == [{"remedy": stated["target"], "book_line": stated["evidence"]}]


def test_a_term_is_found_by_another_of_its_names(graph: ConceptGraph) -> None:
    """The full-text index covers a term's aliases: the book also calls a response an outcome."""
    rows = graph.query(
        "CALL db.index.fulltext.queryNodes('term_text', 'outcome') YIELD node RETURN node.id AS id LIMIT 5"
    )
    assert "response" in [row["id"] for row in rows]
    assert "outcome" in next(term for term in DATA["nodes"] if term["id"] == "response")["aliases"]


@pytest.mark.parametrize("name", sorted(module.EXAMPLE_QUERIES))
def test_every_example_query_of_the_schema_returns_rows(graph: ConceptGraph, name: str) -> None:
    """An agent copies these, so each must run on the graph as it is."""
    assert graph.query(module.EXAMPLE_QUERIES[name])


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE (:Term {id: 'made_up'})",
        "MATCH (term:Term {id: 'curvature'}) SET term.definition = 'changed'",
        "MATCH (term:Term) DETACH DELETE term",
        "CREATE INDEX made_up IF NOT EXISTS FOR (term:Term) ON (term.name)",
        "UNWIND [1] AS x CALL (x) { CREATE (:Term {id: 'made_up'}) } IN TRANSACTIONS RETURN x",
    ],
)
def test_no_statement_changes_the_terms(graph: ConceptGraph, statement: str) -> None:
    """The graph an agent queries is opened read-only, and stays as the book has it."""
    with pytest.raises(ValueError, match="WriteOnReadOnlyAccessDatabase"):
        graph.query(statement)
    with ConceptGraph() as later:
        assert later.schema()["labels"] == graph.schema()["labels"]
        assert later.query("MATCH (term:Term {id: 'curvature'}) RETURN term.definition AS text")[0]["text"] != "changed"


def test_no_statement_reads_a_file(graph: ConceptGraph) -> None:
    """LOAD CSV is refused: relate reads a file only from a folder its host names, and the graph names none."""
    with pytest.raises(ValueError, match="LOAD CSV file unavailable"):
        graph.query(f"LOAD CSV FROM 'file://{Path(__file__).as_posix()}' AS line RETURN line")


def test_a_label_made_by_a_procedure_does_not_outlive_its_graph(graph: ConceptGraph) -> None:
    """A read-only connection may call ``db.createLabel``, but it changes nothing another graph sees."""
    graph.query("CALL db.createLabel('MadeUp')")
    with ConceptGraph() as later:
        assert "MadeUp" not in later.query("CALL db.labels() YIELD label RETURN collect(label) AS labels")[0]["labels"]


def test_a_statement_relate_cannot_parse_is_a_value_error(graph: ConceptGraph) -> None:
    """The caller gets one exception class for every query relate refuses, with relate's message."""
    with pytest.raises(ValueError, match="syntax error"):
        graph.query("MATCH (term RETURN term")
    with pytest.raises(ValueError, match="ParameterMissing"):
        graph.query("RETURN $absent AS value")


def test_nodes_links_paths_and_dates_come_back_as_json(graph: ConceptGraph) -> None:
    """Whatever a query returns can go to an agent as JSON."""
    (row,) = graph.query(
        "MATCH path = (term:Term {id: 'curvature'})-[link:REMEDIED_BY]->() RETURN term, link, path, date() AS today"
    )
    assert json.loads(json.dumps(row)) == row
    assert row["term"]["properties"]["id"] == "curvature"
    assert "Term" in row["term"]["labels"]
    assert row["link"]["type"] == "REMEDIED_BY"
    assert len(row["path"]["nodes"]) == 2
    assert isinstance(row["today"], str)


def test_each_thread_opens_a_graph_of_its_own() -> None:
    """A connection is tied to its thread, so the tool opens a graph per call."""
    pytest.importorskip("relate")
    counts: list[int] = []

    def count_terms() -> None:
        with ConceptGraph() as opened:
            counts.append(opened.query("MATCH (term:Term) RETURN count(term) AS n")[0]["n"])

    threads = [threading.Thread(target=count_terms) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert counts == [len(DATA["nodes"])] * 3


def test_a_forked_child_leaves_the_file_of_its_parent(tmp_path: Path) -> None:
    """The file of terms is removed when the process that built it exits, and by no other."""
    module._remove_store(str(tmp_path), os.getpid() + 1)
    assert tmp_path.exists()
    module._remove_store(str(tmp_path), os.getpid())
    assert not tmp_path.exists()


# --- a study's steps, in a file ---------------------------------------------------------------


def test_a_step_is_linked_to_what_it_found_and_what_it_recommends(campaign: ConceptGraph) -> None:
    """Curvature found, axial runs recommended: each with the book's definition, line and links."""
    assert campaign.record_step("Reactor yield", CURVED) == 1
    explained = {(row["role"], row["term"]): row for row in campaign.explain("Reactor yield")}
    assert set(explained) == {
        ("found", "Curvature"),
        ("recommends", "Axial point"),
        ("recommends", "Face-centred central composite design"),
    }
    curvature = explained["found", "Curvature"]
    assert (curvature["step"], curvature["action"]) == (1, "augment_to_rsm")
    assert curvature["book_line"] == next(t for t in DATA["nodes"] if t["id"] == "curvature")["evidence"]
    assert {"link": "REMEDIED_BY", "term": "Quadratic model"}.items() <= curvature["links"][0].items()
    assert explained["recommends", "Axial point"]["links"] == []  # the book links to it, not from it


def test_every_finding_of_the_evidence_is_linked(campaign: ConceptGraph) -> None:
    """Lack of fit and an effect confounded with an interaction are found too; an action with no term links none."""
    step: dict[str, Any] = {
        "action": "investigate",
        "evidence": {"lack_of_fit": {"significant": True}, "aliasing": [{"term": "A", "aliased_with": "B:C"}]},
    }
    campaign.record_step("Reactor yield", step)
    found = {(row["role"], row["term"]) for row in campaign.explain("Reactor yield")}
    assert found == {("found", "Confounding"), ("found", "Lack-of-fit test")}
    campaign.record_step("Reactor yield", {"action": "stop"})
    assert campaign.explain("Reactor yield") == []  # the latest step: nothing found, and stopping has no term


def test_the_history_is_per_study_and_survives_closing_the_file(tmp_path: Path) -> None:
    """A later session reads the actions back, in order, as ``recommend_next_step`` takes them."""
    pytest.importorskip("relate")
    file = tmp_path / "campaign.rl"
    with ConceptGraph(file) as opened:
        assert opened.history("Reactor yield") == []
        assert opened.record_step("Reactor yield", CURVED) == 1
        assert opened.record_step("Reactor yield", {"action": "confirm"}) == 2
        assert opened.record_step("Coating", {"action": "replicate"}) == 1
    with ConceptGraph(file) as opened:
        assert opened.history("Reactor yield") == ["augment_to_rsm", "confirm"]
        assert opened.history("Coating") == ["replicate"]
        assert opened.schema()["labels"]["Term"] == len(DATA["nodes"])  # the terms were not loaded twice


def test_the_terms_of_a_campaign_file_follow_the_package(tmp_path: Path) -> None:
    """A file written with an earlier graph gets the package's terms on opening, and its steps keep their links."""
    pytest.importorskip("relate")
    file = tmp_path / "campaign.rl"
    with ConceptGraph(file) as opened:
        opened.record_step("Reactor yield", CURVED)
        # As a file from an earlier commit of the graph would be: a term since reworded, one since dropped.
        opened.query("MATCH (term:Term {id: 'curvature'}) SET term.definition = 'An earlier wording.'")
        opened.query("CREATE (:Term:Concept {id: 'since_dropped', name: 'Since dropped'})")
        opened.query("MATCH (source:ConceptSource) SET source.commit = 'an-earlier-commit'")
    with ConceptGraph(file) as opened:
        (curvature,) = [row for row in opened.explain("Reactor yield") if row["term"] == "Curvature"]
        assert curvature["definition"] == next(t for t in DATA["nodes"] if t["id"] == "curvature")["definition"]
        assert opened.schema()["labels"]["Term"] == len(DATA["nodes"])
        assert opened.query("MATCH (source:ConceptSource) RETURN source.commit AS commit") == [
            {"commit": DATA["source"]["commit"]}
        ]
        assert opened.history("Reactor yield") == ["augment_to_rsm"]


def test_a_step_that_cannot_be_recorded_says_why(graph: ConceptGraph, campaign: ConceptGraph) -> None:
    """No file, no study name, or an action that is not one of ``recommend_next_step``'s."""
    with pytest.raises(ValueError, match="needs a campaign file"):
        graph.record_step("Reactor yield", CURVED)
    with pytest.raises(ValueError, match="study must be a name"):
        campaign.record_step("", CURVED)
    with pytest.raises(ValueError, match="got 'guess'"):
        campaign.record_step("Reactor yield", {"action": "guess"})
    assert campaign.history("Reactor yield") == []


@pytest.mark.integration
def test_a_real_recommendation_is_recorded_and_explained(campaign: ConceptGraph) -> None:
    """Centre runs above the plane of the corners: the engine decides, and the graph gives the book's reason."""
    factors = [Factor(name="T", low=300, high=340), Factor(name="P", low=1, high=3)]
    corners = [(300, 1, 50.1), (340, 1, 60.2), (300, 3, 55.0), (340, 3, 65.3)]
    again = [(300, 1, 49.8), (340, 1, 59.9), (300, 3, 55.4), (340, 3, 64.8)]
    centre = [(320, 2, 66.0), (320, 2, 66.4), (320, 2, 65.7)]
    runs = pd.DataFrame(corners + again + centre, columns=["T", "P", "y"])

    step = recommend_next_step(factors, runs, "y", history=campaign.history("Reactor yield"))
    campaign.record_step("Reactor yield", step)

    assert campaign.history("Reactor yield") == [step["action"]] == ["augment_to_rsm"]
    remedies = campaign.query(
        "MATCH (:Study {name: $study})-[:HAS_STEP]->(:Step)-[:FOUND]->(found:Term)-[link:REMEDIED_BY]->(fix:Term) "
        "RETURN found.id AS found, fix.id AS remedy, link.evidence AS book_line",
        {"study": "Reactor yield"},
    )
    assert [(row["found"], row["remedy"]) for row in remedies] == [("curvature", "quadratic_model")]
    assert remedies[0]["book_line"].endswith(".rst:272")


# --- the tool ---------------------------------------------------------------------------------


def test_the_tool_is_registered_with_a_closed_input() -> None:
    """The spec an agent reads exists, and refuses a key it does not name."""
    (spec,) = get_tool_specs(["query_concept_graph"])
    assert spec["input_schema"]["additionalProperties"] is False
    assert set(spec["input_schema"]["properties"]) == {"cypher", "parameters", "max_rows"}


def test_the_tool_with_no_statement_returns_the_schema() -> None:
    """The first call tells an agent what it can ask."""
    pytest.importorskip("relate")
    result = execute_tool_call("query_concept_graph", {"cypher": "  "})
    assert set(result) == {"schema", "source"}
    assert result["schema"]["labels"]["Term"] == len(DATA["nodes"])
    assert result["source"]["commit"] == DATA["source"]["commit"]


def test_the_tool_returns_rows_up_to_its_limit() -> None:
    """A long answer is cut to ``max_rows``, and says so."""
    pytest.importorskip("relate")
    asked = {"cypher": "MATCH (term:Term) RETURN term.id AS id ORDER BY term.order", "max_rows": 3}
    result = execute_tool_call("query_concept_graph", asked)
    assert [len(result["rows"]), result["n_rows"], result["truncated"]] == [3, len(DATA["nodes"]), True]
    one = execute_tool_call(
        "query_concept_graph",
        {"cypher": "MATCH (term:Term {id: $id}) RETURN term.name AS name", "parameters": {"id": "curvature"}},
    )
    assert (one["rows"], one["truncated"]) == ([{"name": "Curvature"}], False)


def test_the_tool_reports_a_refused_statement_as_an_error() -> None:
    """A write is an error envelope, not an exception, and the terms are as they were."""
    pytest.importorskip("relate")
    result = execute_tool_call("query_concept_graph", {"cypher": "MATCH (term:Term) DETACH DELETE term"})
    assert "could not answer the query" in result["error"]
    count = execute_tool_call("query_concept_graph", {"cypher": "MATCH (term:Term) RETURN count(term) AS n"})
    assert count["rows"] == [{"n": len(DATA["nodes"])}]
