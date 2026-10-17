.. _glm-analysis:

Counts, proportions and skewed responses: generalized linear models
===================================================================

Least squares assumes that the response varies by the same amount about every fitted value.
Several common responses do not:

- a count of defects varies more where the count is larger;
- a number of successes out of a number of trials is bounded by 0 and the trials;
- a time to failure spreads in proportion to its size.

A generalized linear model relates the mean of such a response to the factors through a *link*
(the log of a count's mean, the log-odds of a proportion), and lets the variance follow the mean as
the response's distribution (its *family*) says it should. ``analysis_type="glm"`` fits one, by
maximum likelihood.

The coupon experiment
---------------------

Montgomery (2013, section 15.1) describes a test of three features of a coupon mailing: A, B and C,
each at two levels. Each of the eight combinations was mailed to 1000 customers, and the response is
the number who redeemed the coupon: a number of successes out of 1000 trials, so the family is
binomial.

.. code-block:: python

   import pandas as pd
   from process_improve.experiments import analyze_experiment

   runs = pd.DataFrame({
       "A": [-1, 1, -1, 1, -1, 1, -1, 1],
       "B": [-1, -1, 1, 1, -1, -1, 1, 1],
       "C": [-1, -1, -1, -1, 1, 1, 1, 1],
       "redeemed": [200, 250, 265, 347, 210, 286, 271, 326],
   })
   out = analyze_experiment(
       runs, response_column="redeemed", model="redeemed ~ A + B + C + B:C",
       analysis_type="glm", family="binomial", trials=1000,
   )["glm"]
   a = out["coefficients"][1]
   print(round(a["coefficient"], 4), round(a["exp_coefficient"], 2))   # 0.1687 1.18

Under the logit link, a coefficient is a change in the log-odds of redemption, and
``exp_coefficient`` is the *odds ratio*: moving A by one coded unit multiplies the odds of
redemption by 1.18. The result also holds:

- each coefficient's Wald ``z_value``, ``p_value`` and interval;
- the ``log_likelihood``, ``deviance`` and ``pearson_chi2``;
- the ``goodness_of_fit`` tests, with Hosmer and Lemeshow's for a binomial response;
- the ``fitted`` mean of each run with its interval.

The interval for a mean is worked on the scale of the link and carried back, so a proportion's
interval stays between 0 and 1, and is not symmetric about the fitted value.

Families, links and conventions
-------------------------------

``family`` is ``"binomial"``, ``"poisson"``, ``"gamma"``, ``"gaussian"`` or
``"inverse_gaussian"``, and ``link`` defaults to the family's canonical link (logit, log, inverse,
identity and inverse squared, in that order). A gamma response is often fitted with
``link="log"`` instead, so the factors act on the response multiplicatively.

Packages differ in two choices, and both are offered:

- ``dispersion``: the binomial and Poisson families fix it at 1. For the others it is
  estimated by Pearson's statistic over the residual degrees of freedom by default, as R's
  ``glm`` does, or by maximum likelihood with ``dispersion="ml"``, as SAS's GENMOD does.
- ``information``: the covariance of the coefficients inverts the expected information by
  default, as R does, or the observed information with ``information="observed"``, as GENMOD
  does. The two agree under the canonical link, and differ otherwise.

Montgomery's gamma model of the worsted yarn (Table 15.6) is reproduced, interval for interval,
with ``family="gamma", link="log", dispersion="ml", information="observed"``.
