# (c) Kevin Dunn, 2010-2026. MIT License.
"""Per-analysis implementations behind :func:`analyze_experiment` (ENG-02).

Each module holds one (or a small cohesive group) of the ``_run_*`` analysis
routines that ``process_improve.experiments.analysis.analyze_experiment``
dispatches to. They operate on an already-fitted statsmodels OLS result (and,
where relevant, the design DataFrame), so they have no dependency on the
``analysis`` dispatcher and can be tested in isolation.
"""
