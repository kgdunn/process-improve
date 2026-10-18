Evaluating an agent that designs experiments
============================================

The study benchmark (``process_improve.experiments.benchmark``) follows
``recommend_next_step`` mechanically, round by round. In use, an agent reads the
tools' answers, makes its own choices and can overrule them. This page records how
to test that agent: an LLM agent runs whole studies, with the process-improve
tools and without them, on processes whose truth is known.

The laboratory
--------------

``scripts/doe_lab.py serve`` holds one benchmark case in memory and answers over
HTTP on localhost. The agent learns the process only through the runs it pays for,
as at a real bench:

- ``GET /problem``: the factors, their ranges and descriptions, the response, the
  goal and the run budget. Nothing of the true model.
- ``POST /runs``: a round of runs in the factors' own units. Each run must lie
  inside its range and the budget. Refused runs are not charged.
- ``GET /history``: every run so far.
- ``POST /answer``: the final settings, the predicted response and a summary.

Every request goes to a log. ``scripts/doe_lab.py score`` reads the logs afterwards
and reports the runs used, the true response at the answer, and the shortfall from
the true best in noise standard deviations.

The protocol
------------

- **Cases**: the adhesion problem of the "AI agents in DoE" study (8 factors, noise
  3), and the processes hypothesised for its Problems 02 (12 factors, noise 5) and
  03 (10 factors, noise 8). The goal is to maximize the response. Every factor
  range is a hard limit, and the budget is 150 runs.
- **Conditions**: T, with the process-improve tools through the ``doe-designer``
  skill's command line; B, with general Python only (numpy, pandas, scipy,
  statsmodels, scikit-learn, pyDOE3, openpyxl).
- **Repeats**: two noise seeds per case and condition, 12 sessions in all. Both
  conditions use the same seeds.
- **Blindness**: each session has its own lab, port and work folder. The tools run
  from an environment that does not hold the benchmark's truth models. The brief
  forbids reading files outside the work folder (and, for T, the toolkit), and each
  agent lists the files it read.

The brief, shared by both conditions (port, folder and budget filled in)::

    You are an experimenter. A laboratory at http://127.0.0.1:{PORT} makes
    experimental runs for you. Your goal: find factor settings that MAXIMIZE the
    response, using as few runs as you sensibly can. Every run costs time and
    material; the budget is 150 runs, but a good study uses far fewer. The
    response is measured with run-to-run noise.
    [the four endpoints, as above]
    Rules: treat the laboratory as a black box; read no file outside your work
    folder; plan in rounds, as a careful experimenter would: design, run,
    analyse, decide what next.
    When you finish, reply with: the runs used; the designs used in each round;
    the final settings and predicted response; the factors and terms that
    matter; why you stopped where you did; and every file you read outside your
    work folder.

Condition T adds::

    You have the process-improve design-of-experiments toolkit. Read
    skills/doe-designer/SKILL.md first, then use its tools through the command
    line (doe_tool.py list / spec / call). Useful ones: recommend_strategy,
    generate_design, simulate_design_performance, analyze_omars,
    analyze_experiment, recommend_next_step (after every round: it says what
    the results call for next), write_run_sheet / read_run_sheet.

Condition B adds::

    Use Python with numpy, pandas, scipy, statsmodels, scikit-learn, pyDOE3 and
    openpyxl. Do not install or import any other package.

Results
-------

Twelve sessions, 2026-10-18, with ``recommend_next_step`` as of commit ``6ff8a31``: before a
model that does not fit the repeated runs was judged against them. Shortfall is the true best
minus the true response at the agent's answer, in noise standard deviations:

.. list-table::
   :header-rows: 1

   * - Case
     - T: runs
     - T: shortfall [sd]
     - B: runs
     - B: shortfall [sd]
   * - Adhesion, 8 factors
     - 24, 24
     - 0.000, 0.000
     - 67, 84
     - 0.000, 0.000
   * - Hardness 02, 12 factors
     - 51, 32
     - 0.090, 0.010
     - 81, 63
     - 0.003, 0.307
   * - Impact 03, 10 factors
     - 28, 57
     - 1.012, 0.030
     - 77, 89
     - 0.253, 0.012
   * - Mean
     - 36
     - 0.190
     - 77
     - 0.096

With the tools, the agents used half the runs, 36 against 77 on average. Every
T session followed ``recommend_next_step`` round by round and stopped when it
said stop. The B sessions built their own local designs around the best corner,
spent more runs, and got slightly closer on average.

What the sessions showed about the tools:

- **One study stopped too early.** T on impact 03, seed 11, confirmed after its
  first 25-run definitive screening design and fell 1 sd short. The design found
  four of the seven active main effects at noise 8. The factors left out of the
  model went to the end their small effect favours, and one of them (X09) has a
  strong square in the truth. The deterministic benchmark shows the same weak spot
  on this case (mean shortfall 0.8 sd).
- **Blocks changed the advice.** T on hardness 02, seed 11, got "confirm" without
  blocks and "add 8 runs" with blocks, because the blocked judging found squares
  of X04 and X05 active (p = 0.042 and 0.010). The agent's own fit gave them
  p = 0.65 and 0.22, and the truth has neither. Squares that the stepwise search
  proposes are then tested in the model that proposed them, which flatters them.
  This is an open issue.
- **No agent read anything it was told not to.** Each listed only the toolkit's
  skill file, and the instruction files the session loads by itself.

Running it again
----------------

#. Start one lab per session, each with its own port and log::

       uv run python scripts/doe_lab.py serve --case "impact (03), hypothesised" \
           --port 8815 --seed 11 --log runs/T-impact-1/lab.jsonl

#. Give each agent the brief, with its port and work folder.
#. Score every log at the end::

       uv run python scripts/doe_lab.py score runs/*/lab.jsonl
