Panel diagnostics: can this attribute be modelled?
==================================================

:doc:`sensory_panel` scores each *panelist*. This page is about the prior
question, asked of each *attribute*: does it behave like an intensity that
assessors read off a linear scale at all? Two things break that assumption, and
both break it quietly, so :mod:`process_improve.sensory.diagnostics` tests them
before a model is fitted rather than after.

.. code-block:: python

   from process_improve.sensory import (
       assessor_variance_equality,
       boundary_occupancy,
       detection_rate,
   )

   occupancy = boundary_occupancy(validated.normalized_df)
   equality = assessor_variance_equality(validated.normalized_df)

An attribute pinned against a bound
-----------------------------------

The Mixed Assessor Model's premise is that assessors compress or expand a linear
scale. In a region where everyone records the same value, no scaling difference
is expressible: there is nothing for the model to estimate, and what is left
over is reported as disagreement.

:func:`~process_improve.sensory.boundary_occupancy` measures how much of each
attribute lives against the floor or the ceiling:

.. code-block:: python

   >>> boundary_occupancy(panel).query("frac_floor > 0.5")[["attribute", "frac_floor", "frac_exact_zero"]]
        attribute  frac_floor  frac_exact_zero
   2       burnt        0.78             0.71
   5   medicinal        0.64             0.00

Floor, ceiling and exact-zero occupancy are reported separately because they are
separate questions. Look at the second row above: ``medicinal`` is floor-pinned
but has no exact zeros at all. That is the signature of a panel whose convention
is to record "not perceived" as a small positive number rather than a zero. It
looks pinned and is not, and only the ``exact_zero`` column tells the two apart.

The response that suits such an attribute
-----------------------------------------

For an attribute most assessors do not perceive, "how intense is it" has no
answer, but "how often is it perceived at all" does.
:func:`~process_improve.sensory.detection_rate` gives a product-by-attribute
table of detection probabilities:

.. code-block:: python

   >>> detection_rate(panel)["burnt"].sort_values(ascending=False).head(3)
   product
   P07    0.83
   P02    0.42
   P11    0.08

.. warning::

   A detection rate is **not comparable with an intensity score**. It is a
   probability on ``[0, 1]``, it does not carry the attribute's units, and it
   must not be dropped into the same table, correlation matrix or PLS block as
   intensity-scored attributes without saying what it is. Two attributes with
   the same mean intensity can have very different detection rates, and the
   reverse.

A product-attribute pair nobody assessed comes back ``NaN`` rather than 0:
never detected and never asked are different answers.

Unequal assessor variance, mistaken for scale use
-------------------------------------------------

This is the highest-value check on the page. Grossmann et al. (2023) show that
the Mixed Assessor Model reads unequal assessor variance as a *scaling* effect:
an assessor who is simply noisier than the rest loads onto the same term that a
scale-compressor does, which shifts the MAM F-test so that real disagreement is
understated.

:func:`~process_improve.sensory.assessor_variance_equality` tests that
precondition directly. Residuals are taken against the product mean first, which
removes the genuine product effects that would otherwise dominate the spread;
Levene's test (median-centred, the Brown-Forsythe variant) then compares
assessors:

.. code-block:: python

   >>> assessor_variance_equality(panel).query("p_equal_variance < 0.05")
       attribute  levene_stat  p_equal_variance  spread_ratio_max_min  n_assessors
   1     bitter          6.42            0.0004                  5.31           11

A small ``p_equal_variance`` means the assessors genuinely differ in spread, and
the MAM's scaling coefficients for that attribute are measuring partly that.
``spread_ratio_max_min`` is the effect size to read alongside it: a p-value
below 0.05 with a ratio of 1.4 across eleven assessors is a different situation
from the same p-value with a ratio of 5.

An empty panel is an error
--------------------------

:func:`~process_improve.sensory.mixed_assessor_model` used to return frames with
no columns when handed a panel with no rows, so an over-filtered panel surfaced
as ``KeyError: 'f_product_mam'`` somewhere downstream. It now raises a
``ValueError`` naming the condition and pointing at the filter that is the usual
cause. The three functions on this page do the same.

References
----------

Grossmann, Ellis, Hopfer and others, "The effect of unequal assessor variance on
the Mixed Assessor Model", *Food Quality and Preference*, 105, 104792, 2023,
`doi:10.1016/j.foodqual.2022.104792
<https://doi.org/10.1016/j.foodqual.2022.104792>`_.

API
---

.. automodule:: process_improve.sensory.diagnostics
   :members:
   :undoc-members:
   :show-inheritance:
