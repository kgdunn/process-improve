# (c) Kevin Dunn, 2010-2026. MIT License.
"""The book's graph of experimental-design terms, queried in Cypher.

*Process Improvement using Data* names 237 terms of the design and analysis of experiments
and states 490 links between them (``curvature`` is ``REMEDIED_BY`` a ``quadratic_model``).
Every term and every link cites the line of the book it comes from. The graph is built and
checked against the book in the book's repository, and is the one the concept map at
https://factori.al/map/ draws.

:class:`ConceptGraph` loads it into `relate <https://relatedb.ai/>`_, an embedded graph
database, so that a question about how the terms connect is one Cypher query, and its
answer carries the book's own lines::

    from process_improve.experiments.knowledge import ConceptGraph

    with ConceptGraph() as graph:
        graph.query(
            "MATCH (:Term {id: $id})-[link:REMEDIED_BY]->(fix:Term) "
            "RETURN fix.name AS remedy, link.evidence AS book_line",
            {"id": "curvature"},
        )
    # [{'remedy': 'Quadratic model', 'book_line': 'design-analysis-experiments/response-surface-methods.rst:272'}]

Given a file, the graph also keeps a study's steps beside the terms: each reply of
:func:`~process_improve.experiments.recommend_next_step` is recorded with what it found and
what it recommends, linked to the book's terms, so a later session reads the study's history
from the file rather than from the caller::

    with ConceptGraph("reactor.rl") as graph:
        step = recommend_next_step(factors, runs, "yield", history=graph.history("Reactor yield"))
        graph.record_step("Reactor yield", step)
        graph.explain("Reactor yield")  # the step's terms: definition, book line, and their links

relate is not installed with process-improve: ``pip install relate-db``.
"""

from __future__ import annotations

import atexit
import functools
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Self

_DATA_FILE = Path(__file__).parent / "data" / "concept_graph.json"

#: Why relate is not a declared extra: ``[all]`` holds every extra, and relate has no wheel for
#: free-threaded Python, so ``pip install 'process-improve[all]'`` would fail there.
_INSTALL_HINT = (
    "The concept graph needs the relate-db package, which process-improve does not install. "
    "Install it with:\n    pip install relate-db"
)

#: The properties of a term in the graph, as an agent reads them in ``schema()``.
TERM_PROPERTIES = (
    "id",
    "name",
    "type",
    "definition",
    "aliases",
    "found_as",
    "other_names",
    "level",
    "order",
    "chapter",
    "book_file",
    "book_label",
    "evidence",
    "quote",
    "also",
)

#: The book's terms that each action of ``recommend_next_step`` recommends. An action the book
#: has no term for recommends none.
ACTION_TERMS: dict[str, tuple[str, ...]] = {
    "stop": (),
    "confirm": ("confirm_the_prediction",),
    "add_center_points": ("center_point",),
    "replicate": ("replicate",),
    "widen_ranges": ("operating_range",),
    "de_alias": ("foldover",),
    "augment_to_rsm": ("axial_point", "face_centred_central_composite_design"),
    "add_runs_for_model": ("augmentation",),
    "move_region": ("steepest_ascent",),
    "investigate": (),
}

#: Queries that show the graph's shape to an agent. Each returns rows (a test runs them).
EXAMPLE_QUERIES: dict[str, str] = {
    "Find a term by any of its names": (
        "CALL db.index.fulltext.queryNodes('term_text', 'confounded effects') YIELD node, score "
        "RETURN node.id AS id, node.name AS name, node.definition AS definition LIMIT 5"
    ),
    "What a term links to, with the book line of each link": (
        "MATCH (:Term {id: 'curvature'})-[link]->(other:Term) "
        "RETURN type(link) AS link, other.name AS term, link.evidence AS book_line"
    ),
    "How two terms are connected": (
        "MATCH path = shortestPath((:Term {id: 'alias_structure'})-[*..8]-(:Term {id: 'steepest_ascent'})) "
        "RETURN [term IN nodes(path) | term.name] AS terms, [link IN relationships(path) | type(link)] AS links"
    ),
    "The designs that have a property": (
        "MATCH (design:DesignType)-[:HAS_PROPERTY]->(property:Term) "
        "RETURN property.name AS property, collect(design.name) AS designs"
    ),
}

_LOAD_TERMS = "UNWIND $terms AS row CREATE (term:Term) SET term = row, term:$(row.type)"
_LOAD_LINKS = """
UNWIND $links AS row
MATCH (source:Term {id: row.source}), (target:Term {id: row.target})
CREATE (source)-[:$(row.type) {evidence: row.evidence, also: row.also}]->(target)
"""
_INDEX_TERMS = (
    "CREATE FULLTEXT INDEX term_text IF NOT EXISTS "
    "FOR (term:Term) ON EACH [term.name, term.other_names, term.definition]"
)
# A step keeps the ids of its terms, and its links to them are made from those ids, so the
# links survive a reload of the terms from a later commit of the graph.
_LINK_STEPS = (
    "MATCH (step:Step) UNWIND step.found AS id MATCH (term:Term {id: id}) MERGE (step)-[:FOUND]->(term)",
    "MATCH (step:Step) UNWIND step.recommends AS id MATCH (term:Term {id: id}) MERGE (step)-[:RECOMMENDS]->(term)",
)
_EXPLAIN = """
MATCH (:Study {name: $study})-[:HAS_STEP]->(step:Step)
WITH step ORDER BY step.number DESC LIMIT 1
MATCH (step)-[about:FOUND|RECOMMENDS]->(term:Term)
OPTIONAL MATCH (term)-[link]->(other:Term)
WITH step, about, term, collect({link: type(link), term: other.name, book_line: link.evidence}) AS links
RETURN step.number AS step, step.action AS action, toLower(type(about)) AS role, term.name AS term,
       term.definition AS definition, term.evidence AS book_line,
       [entry IN links WHERE entry.term IS NOT NULL] AS links
ORDER BY role, term
"""


def _relate() -> ModuleType:
    """Import relate, or say how to install it."""
    try:
        import relate  # noqa: PLC0415 - optional, and not installed with the package
    except ImportError as error:
        raise ImportError(_INSTALL_HINT) from error
    return relate


@functools.cache
def _concept_data() -> dict[str, Any]:
    """Read the graph the package ships: its terms, its links and the book commit it was built from."""
    return json.loads(_DATA_FILE.read_text(encoding="utf-8"))


def _load_terms(database: Any) -> None:  # noqa: ANN401 - a relate connection; relate is optional
    """Write the book's terms and links into ``database``, unless it already holds this commit's."""
    data = _concept_data()
    commit = data["source"]["commit"]
    if database.execute("MATCH (source:ConceptSource {commit: $commit}) RETURN 1", {"commit": commit}).data():
        return
    terms = [
        {name: term.get(name) for name in TERM_PROPERTIES}
        | {"other_names": ", ".join(term["aliases"] + term["found_as"])}
        for term in data["nodes"]
    ]
    database.execute("MATCH (term:Term) DETACH DELETE term")  # those of an earlier commit of the graph
    database.execute(_LOAD_TERMS, {"terms": terms})
    database.execute(_LOAD_LINKS, {"links": data["edges"]})
    database.execute(_INDEX_TERMS)
    for statement in _LINK_STEPS:
        database.execute(statement)
    database.execute("MERGE (source:ConceptSource) SET source.commit = $commit", {"commit": commit})


def _remove_store(folder: str, owner: int) -> None:
    """Remove the terms' file when its process exits; a forked child leaves its parent's alone."""
    if os.getpid() == owner:
        shutil.rmtree(folder, ignore_errors=True)


@functools.cache
def _concept_store(process_id: int) -> Path:
    """Build, once in this process, the file that ``ConceptGraph()`` opens read-only.

    relate cannot close a database in memory to writes once it is loaded, and a database opened
    read-only is what guarantees that no query changes the terms. So the terms go to a file in a
    temporary folder, which each graph then opens read-only. ``process_id`` keys the cache, so a
    forked worker builds a file of its own.
    """
    folder = tempfile.mkdtemp(prefix="process-improve-concepts-")
    atexit.register(_remove_store, folder, process_id)
    store = Path(folder) / "concepts.rl"
    database = _relate().connect(str(store))
    try:
        _load_terms(database)
    finally:
        database.close()
    return store


def _plain(value: Any) -> Any:  # noqa: ANN401 - any value a query returns
    """Turn a value relate returns into one JSON can hold: a node, a link and a path become dicts."""
    if isinstance(value, str | int | float | bool | None):
        return value
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if hasattr(value, "to_dict"):  # a node, a link or a path
        return _plain(value.to_dict())
    return str(value)  # a date, a time, a duration or a point


class ConceptGraph:
    """The book's terms of experimental design and their links, and optionally one campaign's steps.

    Parameters
    ----------
    campaign_file : str or os.PathLike or None
        ``None`` opens the book's terms read-only: no query can change them. A path opens, or
        creates, a relate file that holds the terms and the steps recorded with
        :meth:`record_step`; the terms in it follow the graph the installed package ships.

    Notes
    -----
    A graph belongs to the thread that opened it: relate refuses a connection used from another
    thread. Open one per thread, and close it (or use it in a ``with`` block).

    Examples
    --------
    >>> from process_improve.experiments.knowledge import ConceptGraph
    >>> with ConceptGraph() as graph:  # doctest: +SKIP
    ...     graph.query("MATCH (term:DesignType) RETURN count(term) AS designs")
    [{'designs': 18}]
    """

    def __init__(self, campaign_file: str | os.PathLike[str] | None = None) -> None:
        self._relate = _relate()
        self._writable = campaign_file is not None
        if campaign_file is None:
            self._database = self._relate.connect(str(_concept_store(os.getpid())), read_only=True)
        else:
            self._database = self._relate.connect(os.fspath(campaign_file))
            _load_terms(self._database)

    def __enter__(self) -> Self:
        """Return the graph, for a ``with`` block that closes it."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the graph."""
        self.close()

    def close(self) -> None:
        """Close the graph; a campaign file is written to disk."""
        self._database.close()

    @property
    def source(self) -> dict[str, str]:
        """Where the terms come from: the book's repository, the file there and its commit."""
        return dict(_concept_data()["source"])

    def query(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run one Cypher statement and return its rows.

        Parameters
        ----------
        cypher : str
            The statement. Terms are ``(:Term)`` nodes that also carry their type as a label
            (``:DesignType``, ``:Method``, ...); :meth:`schema` lists the labels, the links and
            the properties.
        parameters : dict[str, Any] or None
            Values for the statement's ``$name`` parameters.

        Returns
        -------
        list[dict[str, Any]]
            One dict per row, keyed by the statement's column names. A node, a link or a path
            is returned as a dict; a date, a duration or a point as text.

        Raises
        ------
        ValueError
            When relate refuses the statement: a syntax error, a missing parameter, or a write
            to the read-only graph of terms.
        """
        try:
            rows = self._database.execute(cypher, parameters or {}).data()
        except self._relate.Error as error:
            raise ValueError(f"The concept graph could not answer the query: {error}") from error
        return [{column: _plain(value) for column, value in row.items()} for row in rows]

    def schema(self) -> dict[str, Any]:
        """Describe the graph for whoever writes the queries.

        Returns
        -------
        dict[str, Any]
            ``labels`` and ``links`` (each name with its count), ``term_properties``,
            ``full_text_index`` (the index over a term's names and definition) and ``examples``
            (queries that run as they are).
        """
        labels = self.query(
            "MATCH (node) UNWIND labels(node) AS label RETURN label, count(*) AS n ORDER BY n DESC, label"
        )
        links = self.query("MATCH ()-[link]->() RETURN type(link) AS link, count(*) AS n ORDER BY n DESC, link")
        return {
            "labels": {row["label"]: row["n"] for row in labels},
            "links": {row["link"]: row["n"] for row in links},
            "term_properties": list(TERM_PROPERTIES),
            "full_text_index": "term_text",
            "examples": dict(EXAMPLE_QUERIES),
        }

    def record_step(self, study: str, step: dict[str, Any]) -> int:
        """Add to ``study`` the step that :func:`~process_improve.experiments.recommend_next_step` replied.

        The step keeps its action, headline, reasons and number of runs. It is linked to the
        book's term for each thing the evidence found (curvature, lack of fit, an effect
        confounded with an interaction) and for what the action recommends (:data:`ACTION_TERMS`).

        Parameters
        ----------
        study : str
            The study's name. Its first step creates it.
        step : dict[str, Any]
            The reply of ``recommend_next_step``.

        Returns
        -------
        int
            The step's number in the study, from 1.

        Raises
        ------
        ValueError
            For a graph opened without a campaign file, an empty study name, or an action that
            is not one of ``recommend_next_step``'s.
        """
        if not self._writable:
            raise ValueError("record_step needs a campaign file: open the graph as ConceptGraph('<name>.rl').")
        if not study:
            raise ValueError("study must be a name, got an empty one.")
        action = step.get("action")
        if action not in ACTION_TERMS:
            raise ValueError(f"step['action'] must be one of {sorted(ACTION_TERMS)}, got {action!r}.")
        evidence = step.get("evidence", {})
        found = [
            term
            for term, seen in (
                ("curvature", evidence.get("curvature", {}).get("significant")),
                ("lack_of_fit_test", evidence.get("lack_of_fit", {}).get("significant")),
                ("confounding", evidence.get("aliasing")),
            )
            if seen
        ]
        rows = self.query(
            """
            MERGE (study:Study {name: $study})
            WITH study
            OPTIONAL MATCH (study)-[:HAS_STEP]->(earlier:Step)
            WITH study, count(earlier) + 1 AS number
            CREATE (study)-[:HAS_STEP]->(:Step {number: number, action: $action, headline: $headline,
                    reasons: $reasons, n_runs: $n_runs, found: $found, recommends: $recommends})
            RETURN number
            """,
            {
                "study": study,
                "action": action,
                "headline": step.get("headline", ""),
                "reasons": list(step.get("reasons", [])),
                "n_runs": int(step.get("n_runs", 0)),
                "found": found,
                "recommends": list(ACTION_TERMS[action]),
            },
        )
        for statement in _LINK_STEPS:
            self.query(statement)
        return rows[0]["number"]

    def history(self, study: str) -> list[str]:
        """Return the actions of ``study``'s steps, oldest first: the ``history`` of ``recommend_next_step``."""
        rows = self.query(
            "MATCH (:Study {name: $study})-[:HAS_STEP]->(step:Step) RETURN step.action AS action ORDER BY step.number",
            {"study": study},
        )
        return [row["action"] for row in rows]

    def explain(self, study: str) -> list[dict[str, Any]]:
        """Return the book's account of ``study``'s latest step.

        Returns
        -------
        list[dict[str, Any]]
            One dict per term the step found or recommends: ``step`` (its number), ``action``,
            ``role`` (``"found"`` or ``"recommends"``), ``term``, ``definition``, ``book_line``
            (where the book defines the term) and ``links`` (what the book links the term to,
            each with the line that states it). Empty for a study with no step, or a step whose
            action and evidence the book has no term for.
        """
        return self.query(_EXPLAIN, {"study": study})
