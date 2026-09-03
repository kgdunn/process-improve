Chemistry: preparing a product-by-compound block
================================================

A sensory-to-chemistry study has two tables with one row per product: a block
of sensory attribute means, and a block of concentrations or integrated peak
areas. Relating the second to the first with PLS is the easy part. Getting the
second one ready is where the answers are quietly won or lost, and
:mod:`process_improve.chemistry` is the four decisions that make up that work.

The order is fixed
------------------

.. code-block:: text

    trim  ->  transform  ->  centre  ->  scale

Trim first, because a compound detected in two of forty products has no
concentration worth transforming. Transform before centring, because the range
ratio that decides between a log and a linear scale is a property of the raw
values. Centre before scaling, because a scaling constant estimated around the
wrong centre is the wrong constant.

.. code-block:: python

   from process_improve.chemistry import (
       apply_transform,
       center_and_scale,
       classify_zero_states,
       normalisation_check,
       trim_by_prevalence,
   )

   states = classify_zero_states(chem, lod={"linalool": 0.02})
   totals, outside = normalisation_check(chem)

   kept, dropped, presence = trim_by_prevalence(chem, min_nonzero=3)
   transformed, applied = apply_transform(kept, lod={"linalool": 0.02})
   scaled, constants = center_and_scale(transformed, presence[kept.columns])

``scaled`` is now ready for :class:`~process_improve.multivariate.PLS` with
``scale=False``, since it is already centred and scaled.

A zero is not self-describing
-----------------------------

A zero in a concentration table is one of two completely different things:

* a **rounded** zero, also called left-censored: the compound *is* present, at
  a concentration the instrument could not resolve;
* an **essential** zero: the compound is genuinely absent.

They want opposite handling. A rounded zero should be substituted with
something below the detection limit and then modelled as a small quantity; an
essential zero is a categorical fact and modelling it as a small quantity
invents chemistry that is not there. The distinction cannot be recovered from
an exported table, so :func:`~process_improve.chemistry.classify_zero_states`
does not guess:

.. code-block:: python

   >>> classify_zero_states(chem).set_index("compound")["zero_state"].unique()
   array(['unknown'], dtype=object)

``unknown`` is the default, and it is not a failure state: it records that
nobody has yet said which of the other two applies. **Never default to
censored.** Declaring a detection limit *is* the claim that non-detects lie
below it, so passing one in ``lod`` classifies the compound as ``rounded``;
nothing else does that on your behalf.

A trimmed compound is not a discarded compound
----------------------------------------------

:func:`~process_improve.chemistry.trim_by_prevalence` returns three frames, and
the third is the interesting one:

.. code-block:: python

   kept, dropped, presence = trim_by_prevalence(chem, min_nonzero=3)

``presence`` covers **every** compound, kept and dropped alike. For a rare
compound the binary fingerprint often carries more than the concentration ever
could: "this compound appears in exactly the three products the panel called
*green*" is a finding, and a column of thirty-seven zeros and three numbers
states it badly. A missing measurement stays ``NaN`` in that layer rather than
becoming a zero, because never measured and measured-as-absent are different.

Choosing a transform
--------------------

:func:`~process_improve.chemistry.choose_transform` reads the range ratio of the
*detected* values: largest over smallest, ignoring zeros and missing cells. A
compound spanning orders of magnitude is multiplicative and belongs on a log
scale; one varying by a factor of two or three is additive and does not. In
between, the honest answer is ``"ambiguous"``, and
:func:`~process_improve.chemistry.apply_transform` resolves it with a
caller-chosen default rather than a coin toss:

.. code-block:: python

   >>> choose_transform(chem["limonene"])
   'log'
   >>> choose_transform(chem["ethanol"])
   'linear'

A log-transformed compound has its non-detects substituted, by half the
declared detection limit or half the smallest value seen, before the log is
taken. Detected values survive exactly. A linear compound passes through
untouched: a zero on a linear scale is a usable number and needs no
substitution.

``detected_only`` is off for a reason
-------------------------------------

The rule that centring and scaling constants must not be estimated from imputed
values is sound, and easy to over-apply. Where no imputation has happened, the
zeros are real observations of "not detected", and excluding them puts every
one of those zeros many standard deviations below a centre computed from a
handful of detected values. The column becomes effectively binary with a very
large magnitude, and since PLS follows variance, the components then track *how
sparse a variable is* rather than how it relates to the response. Every
attribute comes back with the same handful of rare compounds at the top of its
list, which reads as a finding and is an artefact.

So :func:`~process_improve.chemistry.center_and_scale` defaults
``detected_only=False``. Switch it on **after**
:func:`~process_improve.chemistry.apply_transform` has substituted non-detects
for a log-scaled compound, and not before. The ``detected`` layer is a required
argument either way, so the flag cannot be reached without the mask in hand.

Preprocessing a held-out fold honestly
--------------------------------------

Both fitting steps have a replay partner:

.. code-block:: python

   train, test = chem.iloc[train_rows], chem.iloc[test_rows]

   kept, _dropped, presence = trim_by_prevalence(train, min_nonzero=3)
   train_t, applied = apply_transform(kept)
   train_s, constants = center_and_scale(train_t, presence[kept.columns])

   test_t = apply_fitted_transform(test[kept.columns], applied)
   test_s = apply_fitted_center_scale(test_t, constants)

Without that pair, honest nested cross-validation is not possible: re-deriving
the transform offsets and the scaling constants from the test rows would let
those rows influence their own preprocessing, and the cross-validated score
would then be measuring something other than out-of-sample performance.

The ``constants`` table names its column ``divisor`` and is **divided** by, not
multiplied by. That is deliberate: the standalone
:func:`~process_improve.multivariate.scale` returns a *multiplier* while
:func:`~process_improve.multivariate.center` returns a *subtrahend*, and the two
have been confused often enough that the new table leaves no room for it.

.. seealso::

   :doc:`sensory_diagnostics` for the matching question on the sensory side:
   whether an attribute can be modelled as an intensity at all.

API
---

.. automodule:: process_improve.chemistry.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:
