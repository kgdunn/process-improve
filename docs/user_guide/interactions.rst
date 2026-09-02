Interaction terms (provisional)
===============================

.. warning::

   **The three functions in** :mod:`process_improve.interactions` **are
   unvalidated on real data.** They exist to test conditions that a small
   observational study rarely meets, so this code may go unreached for a long
   time; its coverage is unit tests only, and nothing here has been run against
   a real product-by-compound block with a real sensory response. Treat the API
   as subject to change, and read any result from it as a hypothesis rather
   than a finding.

An interaction between two chemical predictors is a real thing to look for: a
compound whose perceptual effect depends on the level of another is exactly what
a linear additive model misses. Whether the data can support the claim is a
separate matter, and usually the binding one.

Does the pair support a term at all?
------------------------------------

An interaction term says the effect of A depends on the level of B. Estimating
that needs products where A is high and B is low, and products where the reverse
holds, as well as the two agreeing corners. With one corner empty the term is
fitted from three points of support and will report whatever noise lives there.

.. code-block:: python

   >>> from process_improve.interactions import pair_coverage
   >>> covered, detail = pair_coverage(x["linalool"].to_numpy(), x["geraniol"].to_numpy())
   >>> covered, detail["low_high"], detail["high_low"], detail["correlation"]
   (False, 1, 0, 0.91)

Two variables that co-vary occupy only the agreeing corners and fail this check.
**That is the correct answer, not a defect to work around.** Their interaction
is not identifiable from these observations, and no amount of regularisation
makes it so. The returned ``correlation`` is nearly always the explanation.

Each variable is split at its own median, so the marginal split is balanced by
construction and only the *joint* distribution can fail.

Building the terms
------------------

The order is not negotiable: transform, centre and scale, multiply, then
**re-centre and re-scale the products**.

.. code-block:: python

   from process_improve.interactions import interaction_terms

   terms, constants = interaction_terms(x_scaled, [("linalool", "geraniol")])

The second pass is not tidiness. The product of two standardised columns is not
itself centred: for approximately bivariate normal parents its mean is the
parents' correlation :math:`r` and its variance is :math:`1 + r^2`. Skipping the
re-centring therefore leaks correlation into the intercept, and skipping the
re-scaling hands the model a column with more variance than a genuine predictor
would have, inflating exactly the pairs whose interactions deserve the least
trust.

``constants`` records ``center``, ``divisor`` and ``parent_correlation`` per
term, so a held-out block can be built with training constants: multiply the
same parents, subtract ``center``, divide by ``divisor``. Parents that do not
look centred and unit-variance draw a
:class:`~process_improve.multivariate.SpecificationWarning`, since both the
reasoning above and the reported correlation depend on them being so.

How stable is a selection?
--------------------------

A selection made once on all the data is a selection made once.
:func:`~process_improve.interactions.stability_selection` repeats it on
complementary half-samples and reports how often each name comes back:

.. code-block:: python

   >>> stability_selection(select, x_scaled, sensory_means, n_iter=50).head(3)
       name  n_selected  n_subsamples  selection_frequency
   0  ethyl_hexanoate         98           100                 0.98
   1        linalool          61           100                 0.61
   2         nonanal          14           100                 0.14

The halves are complementary: each split is used in both directions, so every
product appears in exactly half the subsamples and the two runs of a split share
no rows.

References
----------

Meinshausen and Buhlmann, "Stability selection", *Journal of the Royal
Statistical Society: Series B*, 72(4), 417-473, 2010,
`doi:10.1111/j.1467-9868.2010.00740.x
<https://doi.org/10.1111/j.1467-9868.2010.00740.x>`_.

Shah and Samworth, "Variable selection with error control: another look at
stability selection", *Journal of the Royal Statistical Society: Series B*,
75(1), 55-80, 2013, `doi:10.1111/j.1467-9868.2011.01034.x
<https://doi.org/10.1111/j.1467-9868.2011.01034.x>`_.

API
---

.. automodule:: process_improve.interactions
   :members:
   :undoc-members:
   :show-inheritance:
