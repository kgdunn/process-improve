.. _region-limits:

Limits on the Settings
======================

Most processes cannot run every combination of their factors' ranges. A reactor can run
hot, or at high pressure, but not both at once; a pump cannot deliver its highest flow at
the highest temperature. Limits like these cut the box of factor ranges down to the
settings that can be run, the *region*. :ref:`constrained-designs` shows how
``generate_design`` places every run inside it. This page covers the rest of the work
with limits: writing them, checking what they leave, proposing one from the settings
known to fail, drawing the region, and comparing designs and plotting a response over it.

The region's diagnostics and the two functions that propose a limit take and return the
factors' actual units. The settings they return (ranges, corners, a central setting, the
outline of a picture) and the numbers of a limit's form are rounded to 12 significant
figures, so ``250.00000000000009`` prints as ``250.0``; a factor whose range is narrow next
to its values, such as a year or an absolute pressure, gets the further figures that keep
its corners inside the region. ``slice_slack`` takes actual units too, and returns for
each point a straight limit's distance from its boundary in coded units, or a curved
limit's own value. The comparison of designs and the plots work in coded units, as noted
where they appear.

Every function is deterministic, so it gives the same answer on every run. The region's
diagnostics and the proposals compute with Python, numpy and scipy alone, and give the
same answer on every platform, a browser running Python included, to within the last of
those 12 figures: a value whose 13th figure sits next to a 5 can round either way when two
platforms' arithmetic differs in its last bit. Averages over a thin region agree less
closely. When so small a share of the box can be run that its points cannot be drawn by
rejection (under 1% for the 20000 points of the comparison of designs), they come from
chains of random steps, which magnify those last-bit differences: in 6 or more factors,
such averages agree across platforms to about 7 to 11 figures.

.. _region-limits-hard:

When a factor's own range is a limit
------------------------------------

The simplest limit is a factor's own range: a vessel rated to 3 bar, a dose a protocol caps, an
instrument that reads nothing below 1 bar. ``Factor(..., hard_limits=True)`` says that ``low``
and ``high`` are such limits, not only the range studied, and every tool then keeps runs inside
them:

.. code-block:: python

   from process_improve.experiments import Factor, generate_design

   factors = [
       Factor(name="T", low=150, high=200, units="degC"),
       Factor(name="P", low=1, high=3, units="bar", hard_limits=True),
   ]
   ccd = generate_design(factors, design_type="ccd")
   print(ccd.alpha, ccd.design_actual["P"].min(), ccd.design_actual["P"].max())
   # 1.0 1.0 3.0

* ``generate_design`` makes a CCD face-centred unless an ``alpha`` is given, and refuses a
  design that would set the factor past its limits. A rotatable CCD here would ask for
  0.59 bar to 3.41 bar.
* ``augment_design`` takes the names in ``hard_limits=[...]``, since its design is in coded
  units. Its axial runs then sit on the faces of the cube, and an ``alpha`` above 1 is
  refused.
* ``recommend_strategy`` names the limits in its assumptions, and plans no stage past them.
* ``recommend_next_step`` never widens or moves such a range. The path of steepest ascent
  moves only the other factors, and the reasons say which factors would improve past their
  limits (see :ref:`next-step-hard-limits`).

A range that is not marked keeps the old behaviour: a design may step past it, with a warning
that names the settings it asks for.

Writing limits as text
----------------------

A :class:`~process_improve.experiments.region.DesignRegion` joins the factors and their
limits. A limit can be a bare inequality, the dict form a saved region holds, or a
:class:`~process_improve.experiments.factor.Constraint`, mixed freely, and it can name
any number of the numeric factors:

.. code-block:: python

   from process_improve.experiments import Constraint, DesignRegion, Factor

   factors = [
       Factor(name="T", low=150, high=200, units="degC"),
       Factor(name="P", low=1, high=3, units="bar"),
       Factor(name="F", low=10, high=20, units="L/min"),
   ]
   limits = ["T + 25*P <= 250", {"expression": "T*F <= 3600", "type": "nonlinear"}]
   region = DesignRegion(factors, limits)
   print(region)
   # DesignRegion(kind='box', factors=['T', 'P', 'F'], constraints=['T + 25*P <= 250', 'T*F <= 3600'])

The first limit is straight: the pressure the vessel is rated for falls by 1 bar for every
25 degC. The second is curved: the heat load, which grows with temperature times flow,
has a ceiling. A limit is written in the factors' names and actual units, as described in
:ref:`constrained-writing-constraints`; it is parsed, never passed to ``eval``.

What the limits leave
---------------------

Before a design is built, it is worth knowing what the limits leave of the factor ranges.
:meth:`~process_improve.experiments.region.DesignRegion.volume_fraction` gives the share
of the box that can be run, and a region built from one limit at a time gives the share
each one removes:

.. code-block:: python

   print(f"{region.volume_fraction():.3f}")  # 0.849
   for constraint in region.constraints:
       alone = DesignRegion(factors, [constraint])
       print(constraint.expression, f"{1 - alone.volume_fraction():.3f}")
   # T + 25*P <= 250 0.125
   # T*F <= 3600 0.041

   print(region.is_empty())  # False
   middle = region.centroid()
   print({name: round(value, 1) for name, value in middle.items()})  # {'T': 172.0, 'P': 1.9, 'F': 14.9}

The straight limit removes the corner where temperature and pressure are both high, a
prism whose share is 0.125, and the curved one a sliver at high temperature and high flow.
With straight limits only, the share is exact: in closed form for factors one limit ties
together, and from the volume of the polytope the limits leave for up to five factors tied
together by several. A curved limit's share is integrated when one factor runs through
every limit that ties the others to it, as the temperature does here. At each temperature,
the pressure and the flow each have an allowed range, and the product of the two is
integrated over the temperature, to about 1e-10: the heat load removes 0.041404. A curved
limit on three factors is counted over a fixed Sobol sample of 16384 points instead, the
same on every run and within about 0.005 of the exact share. Beside straight limits, the
sample counts only what the curved limit takes from the exact share the straight ones
leave, and when the straight limits leave too thin a region for any of the 16384 points,
points spread uniformly through that region take their place, so its share does not read
as 0.
:meth:`~process_improve.experiments.region.DesignRegion.is_empty` says whether any
setting is left at all, and
:meth:`~process_improve.experiments.region.DesignRegion.centroid` gives a central setting
that can be run: where to hold the factors a plot does not show.

For a mixture, the share is of the blends the components' ranges allow (the whole simplex
when every range is 0 to 1). A mixture's limits are straight, so the share is exact:

.. code-block:: python

   blends = [Factor(name=name, type="mixture") for name in ("A", "B", "C")]
   print(DesignRegion(blends, ["A <= 0.5"]).volume_fraction())  # 0.75, all but the corner where A > 0.5

A limit can also narrow a factor's range everywhere, which the ranges then overstate:

.. code-block:: python

   tighter = DesignRegion(factors, ["T + 25*P <= 215"])
   print(tighter.implied_bounds())  # {'T': (150.0, 190.0), 'P': (1.0, 2.6), 'F': (10.0, 20.0)}
   print(region.implied_bounds())  # {'T': (150.0, 200.0), 'P': (1.0, 3.0), 'F': (10.0, 20.0)}

With a rating of 215, the pressure is at least 1 bar, so the temperature can never pass
190 degC; and the temperature is at least 150 degC, so the pressure can never pass 2.6 bar.
:meth:`~process_improve.experiments.region.DesignRegion.implied_bounds` reads these
extremes off the straight limits. Curved limits are left out, so with any of them the
bounds are outer bounds: no setting lies outside them, but an end may not be reached. The
two limits of the running example only cut corners off, and every range is whole.

Two more checks catch limits typed by mistake. A limit that removes nothing the others do
not already remove is redundant, and a set of limits can leave nothing at all:

.. code-block:: python

   typed = DesignRegion(factors, [*limits, "T + 25*P <= 260", "P <= 3.5"])
   print(typed.redundant_constraints())  # [2, 3]

   conflicting = DesignRegion(factors, ["T + 25*P <= 250", "T >= 190", "P >= 2.5"])
   print(conflicting.is_empty())  # True

The third limit is implied by the first, and the fourth by the pressure's own range. Two
copies of the same limit are both reported, since each is redundant given the other, so
remove one at a time.

Finally, each straight limit can be read back as a sum of terms with a lower and an upper
bound (``None`` for a curved limit), and a region with straight limits only has corners
that can be listed:

.. code-block:: python

   print(region.linear_forms())  # [({'T': 1.0, 'P': 25.0}, None, 250.0), None]

   straight = DesignRegion(factors, ["T + 25*P <= 250"])
   corners = straight.vertices()
   print(len(corners), corners[corners[:, 2] == 10].tolist())
   # 10 [[150.0, 1.0, 10.0], [150.0, 3.0, 10.0], [175.0, 3.0, 10.0], [200.0, 1.0, 10.0], [200.0, 2.0, 10.0]]

The vertices are the extreme settings: here the box corners the limit keeps and the two
ends of the cut, (175, 3) and (200, 2), each at both flows. A D-optimal design for a
main-effects model puts its runs on them; one for a quadratic model also puts runs between
them, on edges and faces, and with enough runs inside the region.

Renaming a factor inside a limit
--------------------------------

Limits name the factors, so renaming a factor must rename it inside every limit too.
:func:`~process_improve.experiments.designs_constrained.rename_factors` does so for one
expression:

.. code-block:: python

   from process_improve.experiments import rename_factors

   print(rename_factors("T + 25*P <= 250", {"T": "Temp", "P": "Pres"}))  # Temp + 25 * Pres <= 250
   print(rename_factors("T*F <= 3600", {"T": "F", "F": "T"}))  # F * T <= 3600
   print(rename_factors("sqrt(T) * P <= 40", {"T": "Temp"}))  # sqrt(Temp) * P <= 40

Every name is renamed at once, so a swap works, and function names such as ``sqrt`` are
left alone. The expression is checked before and after, and the result is written in one
canonical form (a space on each side of an operator), so renaming a factor and renaming it
back gives the canonical text. A new name that is not a valid identifier, or a renaming
that would merge two factors, raises ``ValueError``.

Proposing a limit from the settings you cannot run
--------------------------------------------------

Someone who knows the process often knows which settings fail before they can write the
inequality that excludes them.
:func:`~process_improve.experiments.separating.separating_constraint` takes the settings
that cannot be run and those that can, and proposes the straight limit that rules out the
first and keeps the second:

.. code-block:: python

   import itertools

   from process_improve.experiments import NotSeparableError, constraint_through_cuts, separating_constraint

   box_corners = [list(c) for c in itertools.product((150, 200), (1, 3), (10, 20))]
   cannot = [c for c in box_corners if c[0] == 200 and c[1] == 3]  # hot and at high pressure, at any flow
   can = [c for c in box_corners if c not in cannot] + [[175, 2, 15]]  # the other corners and the centre
   print(separating_constraint(factors, cannot, can).expression)  # T + 25 * P <= 250

   one_corner = [[200, 3, 20]]
   others = [c for c in box_corners if c != [200, 3, 20]] + [[175, 2, 15]]
   print(separating_constraint(factors, one_corner, others).expression)  # T + 25 * P + 5 * F <= 350

The proposal is the cut that lies as far from both sets as it can (the maximum-margin
plane, worked out with each factor's range scaled to the same length). It is unique, so a
symmetric pattern gets a symmetric cut, and a factor the pattern does not depend on gets a
coefficient of exactly zero and drops out, as the flow does in the first proposal. The
cut is then written for a reader: its coefficients are rounded to the fewest digits that
move it by at most 1% and keep every setting on its side, and the right-hand side is the
roundest number in the middle of the gap between the two sets. Every setting is checked,
in exact arithmetic, to lie strictly on its side.

When no straight cut separates the two sets,
:class:`~process_improve.experiments.separating.NotSeparableError` (a ``ValueError``)
says so, and names the settings to blame:

.. code-block:: python

   square = [[150, 1], [150, 3], [200, 1], [200, 3]]
   try:
       separating_constraint(factors[:2], [[175, 2]], square)
   except NotSeparableError as err:
       print(err.inside_can)  # [0]

The centre of the square lies between four corners that can be run, so a straight cut that
rules it out rules one of them out too; ``inside_can`` holds its index in ``cannot``. Such a
setting needs a curved limit, or more settings marked. ``inside_cannot`` holds, the other
way round, the settings that can be run but lie between settings that cannot.

A proposal can then be adjusted one factor at a time.
:func:`~process_improve.experiments.separating.constraint_through_cuts` writes the limit
that cuts a corner off the box at given values: from the corner, how far each factor can
go with the others held there, or ``None`` when that factor does not limit it:

.. code-block:: python

   corner = {"T": 200, "P": 3, "F": 20}
   print(constraint_through_cuts(factors, corner, {"T": 190, "P": 2.5, "F": None}).expression)  # T + 20 * P <= 250

With the pressure at 3 bar the temperature can go up to 190 degC, and with the temperature
at 200 degC the pressure can go up to 2.5 bar; the limit passes through those two
settings, whatever the flow. The arithmetic is exact, so round values give whole-number
coefficients.

Pictures of the region
----------------------

A region in more than two factors cannot be drawn whole, but it can be drawn one pair of
factors at a time, in two ways:

* its **shadow** on the pair,
  :meth:`~process_improve.experiments.region.DesignRegion.shadow`: the settings of the two
  factors that some setting of the others can run with, the region seen along the other
  factors' axes;
* a **slice** through it,
  :meth:`~process_improve.experiments.region.DesignRegion.slice_polygon`: the settings of
  the two that can be run with every other factor held at given values.

For straight limits both are exact polygons, corners listed counter-clockwise; with a
curved limit they are masks on a grid. Each straight limit's boundary in the picture,
clipped to the pair's ranges, comes from
:meth:`~process_improve.experiments.region.DesignRegion.slice_lines`:

.. code-block:: python

   print(straight.shadow("T", "P"))
   # {'kind': 'polygon', 'points': [[150.0, 1.0], [200.0, 1.0], [200.0, 2.0], [175.0, 3.0], [150.0, 3.0]]}
   print(straight.slice_lines("T", "P"))  # [{'limit': 0, 'points': [[175.0, 3.0], [200.0, 2.0]]}]
   print(region.slice_polygon("T", "F", at={"P": 1.5})["kind"])  # mask

These are plain lists, ready for any plotting library. With matplotlib, the shadow of every
pair and the limits that cross it:

.. code-block:: python

   import matplotlib.pyplot as plt
   from matplotlib.patches import Polygon

   pairs = list(itertools.combinations(["T", "P", "F"], 2))
   fig, axes = plt.subplots(1, len(pairs), figsize=(10, 3))
   for ax, (x, y) in zip(axes, pairs):
       ax.add_patch(Polygon(straight.shadow(x, y)["points"], alpha=0.3))
       for line in straight.slice_lines(x, y):
           ax.plot(*zip(*line["points"]), color="black")
       ax.autoscale()
       ax.set(xlabel=x, ylabel=y)

:meth:`~process_improve.experiments.region.DesignRegion.slice_slack` gives, on a grid of
two factors in actual units, with the other factors held at given values, how far each
point is from meeting every limit: at most zero exactly where it can be run. For a straight
limit, and for a factor's range, the slack is a distance in coded units, so it changes
linearly across the grid; for a curved limit it is the limit's own value, in its units.
The plots below shade the region with the limits' slack, taken on their own grid in coded
units.

Comparing designs over the region
---------------------------------

:func:`~process_improve.experiments.compare.compare_designs` scores designs on the
measures of design quality, and two of them depend on where predictions are made: I, the
prediction variance averaged over the factor space, and G, its largest value. Over the
whole cube both count settings that cannot be run. Given the region, they are taken over
the settings that can be run instead:

.. code-block:: python

   from process_improve.experiments import compare_designs, generate_design

   options = {"budget": 14, "constraints": limits, "model_type": "quadratic", "random_state": 1}
   designs = {
       "D-optimal": generate_design(factors, "d_optimal", **options),
       "I-optimal": generate_design(factors, "i_optimal", **options),
       "Box-Behnken": generate_design(factors, "box_behnken", n_center_points=2),
   }
   comparison = compare_designs(designs, "quadratic", region=region)
   print(comparison.table.loc[["I, average prediction variance", "G, largest prediction variance"]].to_string())
   print(comparison.notes["Box-Behnken [n=14]"])

.. code-block:: text

                                  Better D-optimal [n=14] I-optimal [n=14] Box-Behnken [n=14]
   I, average prediction variance      ↓           0.4867           0.3992             0.4290
   G, largest prediction variance      ↓           0.9199            1.597              1.438
   ('2 of its 14 runs lie outside the region, while I and G are computed over the region.',)

Over the region, the I-optimal design has the lowest average prediction variance and the
D-optimal design the lowest worst case. Over the whole cube (the same call without
``region``) the Box-Behnken design has the lowest I: the cube includes the part the limits
remove, where the two optimal designs have no runs and must extrapolate, while the
Box-Behnken design has two runs there, which cannot be run.

The designs are in coded units, and the region's factor ranges define the coding. I is
averaged over 20000 points drawn uniformly from the region with a fixed seed, and G is
searched over the region's corners, points on its boundary (on a curved limit too), a
grid, and the same sample (``comparison.g_points`` describes them, and says when a region
has too many corners to list, more than 65536, so that G can come out low). With a curved
limit, local searches then climb each design's prediction variance from the points where
it is largest: the largest value usually lies where the curved boundary meets faces of the
box, which a fixed set of points does not reach. The other measures do not depend on where
predictions are made and are the same with or without a region. A mixture region, or one
that names a factor the designs do not have, is refused.

Plotting a response over the region
-----------------------------------

The contour, 3D surface and prediction-variance plots take the region too. The contour
and prediction-variance plots shade the settings that cannot be run, with the factors not
plotted held at ``hold_values``; the 3D surface leaves those settings out:

.. code-block:: python

   from process_improve.experiments.visualization import visualize_doe

   coefficients = [
       {"term": "Intercept", "coefficient": 72.0},
       {"term": "T", "coefficient": 6.0},
       {"term": "P", "coefficient": 4.0},
       {"term": "F", "coefficient": -2.0},
       {"term": "T:P", "coefficient": 1.5},
       {"term": "I(T ** 2)", "coefficient": -3.0},
   ]
   figure = visualize_doe(
       plot_type="contour",
       analysis_results={"coefficients": coefficients},
       factors_to_plot=["T", "P"],
       hold_values={"F": 0.0},
       region=region,
   )
   print([trace["name"] for trace in figure["plotly"]["data"]])  # ['Response', 'Cannot be run']

The plots are drawn in coded units, as the model's coefficients are, and ``hold_values``
is coded too: ``"F": 0.0`` holds the flow at 15 L/min, the middle of its range. The
response stays the first trace. The shading is a second, translucent trace, a constraint
contour of the limits' slack at zero, which Plotly fills on the side that cannot be run;
for a straight limit the slack is linear, so its edge is drawn straight. The 3D surface is
``NaN`` where a setting cannot be run, and the ECharts option draws the shading as a
masked heatmap over the response's grid. Without a region, or with a region without
limits, every figure is as it was.
