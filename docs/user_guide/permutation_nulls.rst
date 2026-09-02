Permutation nulls: is there anything here at all?
=================================================

At the sample sizes a sensory-to-chemistry study runs at, twenty or thirty
products against fifty compounds, "is there anything here at all" is the hard
question, and the obvious ways of answering it do not work.

Two ways that do not work
-------------------------

**A high in-sample R².** With thirty products and three components, a good R² is
close to guaranteed. It measures capacity, not evidence.

**A count of variables exceeding VIP 1.** This one is worth spelling out,
because it looks like a statistic and is not. VIP is normalised so that

.. math::

   \sum_{j=1}^{K} \text{VIP}_j^2 = K

*exactly*, for any model, on any data:

.. code-block:: python

   >>> from process_improve.multivariate import PLS, vip
   >>> model = PLS(n_components=2, scale=False).fit(x_scaled, y_scaled)
   >>> float((vip(model) ** 2).sum()), x_scaled.shape[1]
   (14.000000000000002, 14)

That identity is exactly what makes the familiar "VIP > 1" rule a sensible
*relative* cut-off: the mean square is 1 by construction, so a score above 1
means above-average within this model. It also means the exceedance count
describes the *shape* of the VIP distribution rather than the presence of a
relationship. Permute the response and the count barely moves, so a null built
on it has essentially no power and will report a false-discovery rate near 100%
on data that does contain signal.

What does work
--------------

Ask a permutation what it can achieve, and compare it with *out-of-sample*
performance.

.. code-block:: python

   from process_improve.multivariate import check_predictive_signal

   check_predictive_signal(chem, sensory_means, n_perm=999)

.. code-block:: text

     attribute  q2_observed  q2_null_mean  q2_null_p95  p_value  n_permutations
   0    fruity        0.612        -0.174        0.201    0.001             999
   1     green        0.088        -0.166        0.213    0.284             999

``fruity`` predicts held-out products far better than any reshuffling manages;
``green`` sits inside its own null. That is a distinction the VIP count above
cannot draw.

Note the blocks go in **unscaled**. The default cross-validates PLS for you and
re-derives the centring and scaling constants inside every fold, which is the
second of the two traps below; scaling first would hand each fold constants
computed from the row it is meant to be predicting.

This is expensive, and irreducibly so: every permutation refits once per fold,
so leave-one-out on twenty products with ``n_perm=999`` is twenty thousand fits.
Develop at ``n_perm=50`` and raise it for the number you intend to report.

Pass your own ``fit_predict`` to use a different model or a cheaper fold scheme:

.. code-block:: python

   def fit_predict(x, y):
       """Return out-of-sample predictions, one row per product."""
       ...  # your own CV loop; see the two responsibilities below

   check_predictive_signal(chem, sensory_means, fit_predict, n_perm=999)

Two things :func:`~process_improve.multivariate.check_predictive_signal` then
leaves to you, inside ``fit_predict``:

* **The cross-validation scheme is yours to choose.** Leave-one-out is right
  when every product is precious, and needlessly expensive at hundreds of rows,
  where a permutation null multiplies the cost by ``n_perm``.
* **Re-derive the response constants inside each fold too.** It is easy to nest
  the predictor preprocessing and forget the response. Centre and scale the
  response on the training rows, predict, then back-transform with those same
  training constants before returning; otherwise the score is computed on a
  scale the fold never saw.

Whole rows are permuted, not columns. Permuting column by column would destroy
the correlation structure among the attributes and inflate the null, making the
test look more impressive than it is.

The p-value has a floor
-----------------------

The p-value uses the ``(1 + count) / (n_perm + 1)`` form, which counts the
observed statistic as one of its own null draws so it can never be exactly zero.
The consequence is worth knowing before you choose ``n_perm``: the smallest
attainable p-value is ``1 / (n_perm + 1)``. The default 500 permutations cannot
report anything below 0.002, and a multiplicity correction over twenty
attributes needs a floor well below the corrected threshold.

Testing the whole procedure, not one model
------------------------------------------

:func:`~process_improve.multivariate.count_discoveries_under_null` takes a
callable that runs filtering, transformation, scaling and selection end to end,
and counts discoveries under a permuted response:

.. code-block:: python

   >>> result = count_discoveries_under_null(select, chem, sensory_means, n_perm=200)
   >>> result["observed"], result["null_mean"], result["null_to_observed_ratio"]
   (9, 1.8, 0.2)

``null_to_observed_ratio`` is not clipped. A value above 1 says shuffling found
*more* than the real response did, which is the strongest evidence the procedure
has nothing, and it is precisely the case worth seeing rather than rounding to a
tidy-looking rate.

That ratio is a property of the procedure **as run**, which is the only version
worth quoting: a procedure whose selection step is honest but whose filtering
step peeked at the response has an FDR no formula recovers.

The response-independent steps are deliberately not hoisted out of the loop.
Re-running them per permutation is a no-op when they really are
response-independent, and hoisting them would be an assumption about your code.
What *is* checked is determinism: a selector that disagrees with itself on two
identical calls makes the FDR meaningless, and draws a
:class:`~process_improve.multivariate.SpecificationWarning` rather than a
silently contaminated number.

Recovering the expected class
-----------------------------

:func:`~process_improve.multivariate.class_enrichment` asks a different kind of
question, hypergeometrically: is a chemically expected class of compounds
over-represented at the top of a ranking?

.. code-block:: python

   >>> class_enrichment(ranking_for_fruity, all_compounds, r"acetate|butanoate")
   {'in_top': 5, 'class_size': 9, 'n_compounds': 61, 'n_drawn': 12,
    'p_value': 0.0004, 'matched': [...]}

At small sample sizes this is frequently the stronger evidence. Recovering the
esters at the top of *fruity* is structure that noise does not produce, whereas
a high R² on few products with several components very nearly is.

.. note::

   Check where the ranking came from first. A one-component PLS has coefficient
   matrix ``outer(x_weights, y_loadings)``, so the absolute coefficients order
   identically for **every** attribute, and the attributes differ only in sign
   and magnitude. One-component solutions are common at small sample sizes, so
   treat this as the normal case rather than an edge case: an enrichment that
   looks attribute-specific may be one ranking reported many times.

API
---

.. automodule:: process_improve.multivariate._null
   :members:
   :undoc-members:
   :show-inheritance:
