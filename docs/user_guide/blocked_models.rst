Models fitted across blocks
===========================

An experiment run in blocks (one day at a time, one batch of raw material at a
time, one sensory panel session at a time) sees every response shifted a little
in each block. A run tested in two blocks gives two results, and both are data:
the unit is one run's result in one block. Runs tested again in a later block,
the *anchors*, tie the blocks together.

:func:`~process_improve.experiments.across_blocks.fit_across_blocks` fits such
data and answers three questions after each block:

- **Which model form?** Each candidate form, a model formula's right-hand side,
  is fitted to every response by least squares with a block effect coded to
  sum to zero, so the intercept is the average block. The form chosen is the
  one with the lowest leave-one-run-out error, averaged over the responses:
  each run is left out of every block it was tested in and predicted there
  from the others. Fix the candidates before the data arrive, so the rule, not
  the data, makes the choice.
- **How large is each block's shift?** ``block_shift`` gives each block's
  offset from the average block, with its standard error.
- **What will a new run give?** ``predict`` gives the prediction with a
  prediction interval in a block of the fit, in the average block, or in a
  coming block whose shift is read from the anchors tested in it.

.. code-block:: python

    import numpy as np
    import pandas as pd

    from process_improve.experiments import fit_across_blocks

    rng = np.random.default_rng(0)
    grid = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
    rows = []
    for day, shift in {"day 1": 0.5, "day 2": -0.2, "day 3": -0.3}.items():
        for k, (a, b) in enumerate(grid):
            if day == "day 1" or k in (0, 4, 8, 2 + len(rows) % 5):  # anchors, and a few others
                y = 5 + 2 * a - b + 0.8 * a**2 + shift + rng.normal(0, 0.1)
                rows.append({"run": f"r{k}", "A": a, "B": b, "day": day, "y": y})
    data = pd.DataFrame(rows).set_index("run")  # the index names the run

    fit = fit_across_blocks(
        data, ["y"],
        {"linear": "A + B", "quadratic": "A + B + I(A**2)", "full": "A + B + A:B + I(A**2) + I(B**2)"},
        block="day",
    )
    fit.chosen_form  # "quadratic"
    fit.selection  # every candidate's leave-one-run-out RMSE, best first
    fit.block_shift  # each day's shift from the average day

    new = pd.DataFrame({"A": [0.5], "B": [0.5]})
    anchors = data[data["day"] == "day 3"].loc[["r0", "r4"]]  # as if day 3 had just tested its anchors
    fit.predict(new, anchors=anchors)  # the prediction and its 95% interval for that day

With ``anchors``, the coming block's shift is the anchors' mean difference from
the model, so the prediction for a new run :math:`x_0` is
:math:`(x_0 - \bar{x}_a)\hat\beta + \bar{y}_a`. Its error variance is
:math:`\sigma^2 \left(1 + 1/n_a + (x_0 - \bar{x}_a)^T (X^TX)^{-1} (x_0 - \bar{x}_a)\right)`,
exact under the model. With neither a block nor anchors, the interval adds the
spread of the fitted block shifts, which overstates the block-to-block variance
a little, as it includes the shifts' own estimation error.

A run that alone carries a term (the only run at a factor's new level) cannot
be left out: ``selection["runs_alone"]`` names it, and a form whose terms the
data cannot separate is marked not ``estimable`` and never chosen. Code the
factors before fitting when a form's meaning depends on their centre and scale;
a categorical factor can be written ``C(F, Sum)`` for effects coding.
