.. _run-sheets:

Run Sheets
==========

A design leaves the analysis as a run sheet, and its results come back as one: a spreadsheet
for the people who make the runs, the file an online study asks you to upload, a CSV from a
laboratory system. ``run_sheet`` writes one and ``read_run_sheet`` reads one back, so the
units and the column names are set once, by the factors, and not by hand.

Writing a sheet
---------------

.. code-block:: python

   from process_improve.experiments import Factor, generate_design, read_run_sheet, run_sheet

   factors = [
       Factor(name="NCO_OH", low=0.9, high=1.3),
       Factor(name="Catalyst", low=0, high=400, units="ppm"),
       Factor(name="Cure_temperature", low=60, high=100, units="degC"),
   ]
   design = generate_design(factors, "full_factorial", n_center_points=2, random_state=7)
   sheet = run_sheet(design, factors, labels="X", responses=["Hardness"])
   sheet.to_csv("runs.csv", index=False)
   print(sheet.head(4).to_string(index=False))

.. code-block:: text

    Run  X01   X02   X03  Hardness
      1  1.1 200.0  80.0       NaN
      2  0.9   0.0  60.0       NaN
      3  1.3 400.0 100.0       NaN
      4  0.9   0.0 100.0       NaN

The rows are the runs in the design's randomised order, numbered in ``Run``. The choices:

* ``units``: ``"actual"`` (the default) writes the settings in the factors' own units;
  ``"coded"`` writes them from -1 at a factor's low value to +1 at its high value. A
  categorical factor keeps its levels either way.
* ``labels``: the factors' own names (the default), ``"X"`` for ``X01``, ``X02``, ... in the
  order of ``factors``, or a mapping from a factor's name to its column.
* ``responses``: the columns left empty for the results.

The runs can also be a table in actual units, such as the ``runs`` that
:func:`~process_improve.experiments.recommend_next_step` returns for the next round.

Reading the results back
------------------------

.. code-block:: python

   runs = read_run_sheet("runs.csv", factors, labels="X")
   print(runs.attrs["units"])

``read_run_sheet`` takes a path to a ``.csv``, ``.xlsx`` or ``.xls`` file, the bytes of a
workbook, a file object holding CSV, or a DataFrame. It returns a column per factor under its
own name and in its actual units, then the sheet's other columns (the responses, the run
numbers) as they were:

* A factor's column is found under its label or its own name, ignoring case, spaces and the
  leading zeros of an ``X`` label, so ``x1`` reads as ``X01``.
* With ``units="auto"`` (the default), the settings are read as coded when every one of them
  lies between -1 and +1 and at least one cannot be an actual value, because it lies outside
  its factor's range; otherwise as actual. ``runs.attrs["units"]`` says which. Pass
  ``units="coded"`` or ``"actual"`` when a sheet could be either.
* A setting outside its factor's range is read, with a warning that names the factor and the
  extremes the sheet holds.
* Rows with no setting at all, such as blank rows at the end of a workbook, are left out.

Both are MCP tools too, ``write_run_sheet`` and ``read_run_sheet``. They exchange the sheet as
CSV text or a base64 ``.xlsx`` workbook, so no tool reads or writes a file.
