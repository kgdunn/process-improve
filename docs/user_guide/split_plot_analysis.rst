.. _split-plot-analysis:

Split-plot experiments
======================

Some factors are slow or expensive to change: a furnace temperature, a reactor
configuration, a batch of raw material. A split-plot experiment sets such a
*hard-to-change* factor once and then runs several settings of the easy factors
before changing it again. Each group of runs that shares one setting of the hard
factors is a *whole plot*; each run within it is a *subplot*.

That restriction gives the experiment two sources of error. Runs in the same whole
plot share whatever happened when it was set up (the furnace was a little hotter
than its dial said that day), and each run also has its own run-to-run error. The
hard-to-change factors change only from one whole plot to the next, so their effects
must be judged against the larger, whole-plot error, with as many degrees of freedom
as there are whole plots to spare. The easy factors are compared within whole plots,
where the whole-plot error cancels, so they are judged against the smaller run-to-run
error.

Ordinary least squares, which every other ``analysis_type`` uses, treats all the runs
as independent. It judges every effect against one pooled error, so the hard-to-change
factors look more significant than they are, and the easy factors less.
``analysis_type="split_plot"`` fits the model with a random whole-plot effect instead.

The corrosion experiment
------------------------

Box, Hunter and Hunter (2005, chapter 9) describe steel bars treated with four
coatings and baked at three furnace temperatures. Resetting the furnace is slow, so
six heats were run, each temperature twice, with one bar of each coating placed at
random in the furnace in every heat. The heats are the whole plots.

.. code-block:: python

   from process_improve.experiments import analyze_experiment, datasets

   df = datasets.corrosion().drop(columns="Position")
   df["Temperature"] = df["Temperature"].astype(str)  # three settings, analysed as categories, as in the book

   result = analyze_experiment(
       df, response_column="Resistance", whole_plot="Heat", analysis_type=["anova", "split_plot"]
   )
   ols_p = {row["source"]: row["p_value"] for row in result["anova_table"]}
   print(f"{'term':<22}{'stratum':<12}{'F':>7}{'df':>9}{'p':>8}{'OLS p':>8}")
   for row in result["split_plot"]["tests"]:
       df_text = f"{row['df']}, {row['df_denominator']:.0f}"
       print(
           f"{row['source']:<22}{row['stratum']:<12}{row['F']:>7.2f}{df_text:>9}"
           f"{row['p_value']:>8.3f}{ols_p[row['source']]:>8.3f}"
       )

::

   term                  stratum           F       df       p   OLS p
   Temperature           whole_plot     2.75     2, 3   0.209   0.003
   Coating               subplot       11.48     3, 9   0.002   0.386
   Temperature:Coating   subplot        4.38     6, 9   0.024   0.852

The two analyses reach opposite conclusions. Least squares finds temperature highly
significant and coating not significant at all. The split-plot analysis tests
temperature against the variation between heats, on 3 degrees of freedom (6 heats,
less 3 for the temperature means), and finds it not significant. It tests coating and
the interaction against the variation between bars within a heat, which is much
smaller, and finds both significant. These are the F ratios of the book's split-plot
ANOVA (Table 9.2).

The variance components say why:

.. code-block:: python

   components = result["split_plot"]["variance_components"]
   print(f"whole plot {components['whole_plot']:.1f}, residual {components['residual']:.1f}, "
         f"eta {components['eta']:.1f}")
   print(result["split_plot"]["error_df"])

::

   whole plot 1172.2, residual 124.5, eta 9.4
   {'whole_plot': 3, 'subplot': 9}

Heat-to-heat variation is more than nine times the bar-to-bar variation. Least squares
spreads that large heat-to-heat variation across all 24 bars, which drowns the coating
effects. It also compares temperature with a pooled error far smaller than the
heat-to-heat error temperature actually has to beat.

From a split-plot design to its analysis
----------------------------------------

``generate_design`` builds a split-plot design when ``hard_to_change`` names the slow
factors (with the optional ``pyoptex`` package installed). The design records each
run's whole plot in a ``WholePlot`` column, numbered from 1, so the analysis finds the
whole plots without being told:

.. code-block:: python

   import numpy as np

   from process_improve.experiments import Factor, generate_design

   factors = [
       Factor(name="Temperature", low=360, high=380, units="degC"),
       Factor(name="Pressure", low=1, high=3, units="bar"),
       Factor(name="Ratio", low=0.5, high=1.5),
   ]
   design = generate_design(
       factors, design_type="d_optimal", budget=20, hard_to_change=["Temperature"], model_type="interactions"
   )
   runs = design.design
   print(runs.head(4))

::

      RunOrder  Temperature  Pressure  Ratio  WholePlot
   1         1          1.0      -1.0    1.0          1
   2         2          1.0       1.0   -1.0          1
   3         3          1.0      -1.0   -1.0          1
   4         4         -1.0      -1.0    1.0          2

Run the experiment, join the responses on, and analyse. Here the responses are
simulated, with a whole-plot error whose standard deviation is four times the run-to-run error's:

.. code-block:: python

   rng = np.random.default_rng(1)
   whole_plot_error = rng.normal(0, 2.0, runs["WholePlot"].max() + 1)[runs["WholePlot"]]
   y = (
       50 + 3 * runs["Temperature"] + 2 * runs["Pressure"] + runs["Pressure"] * runs["Ratio"]
       + whole_plot_error + rng.normal(0, 0.5, len(runs))
   )
   analysis = analyze_experiment(runs.assign(y=y), response_column="y", analysis_type="split_plot")
   for row in analysis["split_plot"]["tests"]:
       print(f"{row['source']:<22}{row['stratum']:<12}{row['df_denominator']:>6.1f}{row['p_value']:>9.4f}")

::

   Temperature           whole_plot     4.0   0.0220
   Pressure              subplot        9.1   0.0000
   Ratio                 subplot        9.1   0.9094
   Temperature:Pressure  subplot        9.0   0.4915
   Temperature:Ratio     subplot        9.1   0.3582
   Pressure:Ratio        subplot        9.1   0.0001

The 20 runs sit in 6 whole plots. Temperature, the only whole-plot term, is judged on
about 4 degrees of freedom; the others on about 9. Any other ``analysis_type`` on a
frame with a ``WholePlot`` column warns that least squares ignores the whole plots.
For data from elsewhere, name the column that labels the whole plots with
``whole_plot="Heat"``; it is never treated as a factor.

How the analysis works
----------------------

The model is :math:`y = X\beta + Zu + e`, with :math:`Z` assigning each run to its whole
plot, independent whole-plot effects :math:`u` of variance :math:`\sigma^2_{wp}`, and
run errors :math:`e` of variance :math:`\sigma^2`.

* **Restricted maximum likelihood (REML).** The variance components are estimated by
  REML, which allows for the degrees of freedom the fixed effects use up, as the ANOVA
  mean squares do. With a single random effect, the likelihood depends only on the
  ratio :math:`\eta = \sigma^2_{wp} / \sigma^2`, and the fit solves for it to machine
  precision. In a balanced design such as the corrosion experiment, REML gives exactly
  the classical split-plot ANOVA. A whole-plot variance can be estimated as zero; the
  estimates are then those of least squares, and ``split_plot_note`` says so.
* **Satterthwaite's degrees of freedom.** Each test's denominator degrees of freedom
  come from Satterthwaite's approximation, which in a balanced design gives the
  whole-plot and subplot error degrees of freedom exactly. A comparison that spans both
  strata gets a value in between. Kenward and Roger's adjustment is not applied; in a
  balanced design it changes nothing.
* **The strata.** A term is in the ``whole_plot`` stratum when its model columns are
  constant within every whole plot, and in the ``subplot`` stratum otherwise. An
  interaction of a hard-to-change factor with an easy one is a subplot term.
* **Marginal (Type III) tests.** Each term is tested adjusted for all the others. For
  this, the named models code a categorical factor with sum-to-zero contrasts, so that
  a main effect is averaged over the factors it interacts with. Write ``C(name, Sum)``
  in an explicit formula for the same.

The analysis refuses a model it cannot separate: one with aliased terms, one whose
whole-plot terms use up every whole plot (no whole-plot error is left to judge them
by), or one with no run-to-run error left.

The result, under ``result["split_plot"]``, holds:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Contents
   * - ``variance_components``
     - ``whole_plot`` and ``residual`` variances, and their ratio ``eta``, which a
       split-plot design is optimised for (``generate_design`` assumes 0.5)
   * - ``error_df``
     - Degrees of freedom of the ``whole_plot`` and ``subplot`` error strata
   * - ``tests``
     - One F-test per model term: ``df``, ``df_denominator``, ``F``, ``p_value`` and ``stratum``
   * - ``coefficients``
     - One t-test per model column, on its own degrees of freedom, with a confidence interval
   * - ``significant_terms``, ``not_significant_terms``
     - The terms on either side of ``significance_level``
   * - ``method``, ``df_method``, ``test_type``
     - ``"REML"``, ``"Satterthwaite"`` and ``"III"``

References
----------

* Box, G. E. P., Hunter, J. S. and Hunter, W. G. (2005). *Statistics for Experimenters*,
  2nd edition, chapter 9. Wiley.
* Goos, P. and Jones, B. (2011). *Optimal Design of Experiments: A Case Study
  Approach*, chapters 10 and 11. Wiley.
* Satterthwaite, F. E. (1946). An approximate distribution of estimates of variance
  components. *Biometrics Bulletin* 2(6), 110-114.
* Fai, A. H.-T. and Cornelius, P. L. (1996). Approximate F-tests of multiple degree of
  freedom hypotheses in generalized least squares analyses of unbalanced split-plot
  experiments. *Journal of Statistical Computation and Simulation* 54(4), 363-378.
