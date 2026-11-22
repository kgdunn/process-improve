- **The book's graph of design terms, queried in Cypher (a prototype).** `ConceptGraph`, in
  `process_improve.experiments.knowledge`, loads the 237 terms and 490 links of *Process Improvement using Data*
  (the graph the concept map at factori.al/map draws) into [relate](https://relatedb.ai/), an embedded graph
  database. Every term and link carries the line of the book it comes from, so an answer can cite its source:
  `graph.query("MATCH (:Term {id: 'curvature'})-[link:REMEDIED_BY]->(fix) RETURN fix.name, link.evidence")` returns
  the quadratic model and `response-surface-methods.rst:272`. `ConceptGraph()` is read-only. The new tool
  `query_concept_graph` gives an agent the same graph: called with no statement it returns the labels, the link
  types, a term's properties and example queries.
- **A study's steps kept beside the terms.** `ConceptGraph("campaign.rl")` keeps a file: `record_step(study, step)`
  stores each reply of `recommend_next_step` and links it to the book's term for what it found (curvature, lack of
  fit, an effect confounded with an interaction) and for what it recommends; `history(study)` returns the actions
  in order, as `recommend_next_step(history=...)` takes them; `explain(study)` returns the latest step's terms with
  their definitions, book lines and links.
- relate is not installed with the package, and is not part of `[all]`, because it has no wheel for free-threaded
  Python: `pip install relate-db`. Without it, `ConceptGraph` and the tool say so.
