# Multivariate browser app: building blocks

The building blocks of a multivariate analysis app (PCA, PLS) that runs in the browser
on Pyodide, next to the designed-experiments app in `web/`. It replaces the server
version of ScorePilot: the same ideas, no server, the data never leaves the machine.
Plain JavaScript modules, Plotly for the charts, and the computation in
`process_improve` itself.

Deployment, and moving the pieces both apps share into one place, come later. Nothing
here is published by `scripts/build_web_app.py` yet.

## The layers

| Layer | Where | Runs in | Tested by |
|---|---|---|---|
| Dataset recipe, model variant, group explanation, project file | `src/process_improve/workbench/` | Python, natively or in Pyodide | `tests/workbench/` |
| Glue: JSON in, JSON out, one `api_*` function per action | `web/mva/bootstrap.py` | the Pyodide worker | `tests/web/mva/` |
| Selection store | `web/mva/selection.js` | the page | Node (`tests/web/mva/*.test.mjs`) |
| Linked Plotly views | `web/mva/views.js` (figures), `web/mva/link.js` (events) | the page | Node for the figures, Chromium for the events |
| Data table | `web/mva/table.js` | the page | Node for the layout, Chromium for the drawing |

The rule that keeps this testable: everything that decides *what* to draw is a pure
function of plain data, and runs under Node. Only `link.js` and the drawing half of
`table.js` touch the document.

## Identity: rows and columns

Every observation has a `rowId` (its index label, as text) and every variable a
`colId` (its column name). Plots, the table and the Python glue speak only in these
ids, never in positions, so a row keeps its identity through a sort, a filter, a
dropped neighbour or a refit.

## 1. The dataset recipe (`process_improve.workbench.Recipe`)

The recipe is the source of truth. It is a list of typed steps over a source table:

| Step | What it does | Python it writes |
|---|---|---|
| `DropColumns` | removes columns | `df = df.drop(columns=[...])` |
| `KeepColumns` | keeps only these columns, in this order | `df = df[[...]]` |
| `RenameColumns` | renames | `df = df.rename(columns={...})` |
| `DropRows` | removes rows by id, with the reason | `df = df.drop(index=[...])  # reason` |
| `FilterRows` | keeps the rows where an expression is true | `df = df[(df["T"] > 200)]` |
| `AddColumn` | a new column from an expression | `df["dT"] = df["Tout"] - df["Tin"]` |
| `Transform` | log, log10, sqrt, logit, power, ... of a column, in place or as a new one | `df["x"] = np.log(df["x"])` |
| `SetIndex` | makes a column the row ids | `df = df.set_index("Batch")` |

- `recipe.apply(df)` runs the steps.
- `recipe.to_python()` writes a script that does the same with pandas and numpy only.
- `recipe.to_dict()` / `Recipe.from_dict()` round-trip through JSON. Undo and redo are
  a list operation.
- A test runs the generated script and checks that it rebuilds the same table **bit for
  bit**, for every step type and for random sequences of steps.

Expressions (`AddColumn`, `FilterRows`) are a small, safe language: numbers, column
names, `+ - * / **`, comparisons, `&`, `|`, `~`, and `log`, `log10`, `exp`, `sqrt`,
`abs`. A name with spaces is written in back-ticks, as pandas does: `` `Real coffee` * 2 ``.
An expression is parsed, never evaluated as Python, so a recipe from a file is data.

**Drop is not exclude.** Dropping a row is a recipe step: the row is gone. Excluding a
row is a choice of the model variant (below): the row stays, the model is fitted
without it, and it is still projected and drawn.

## 2. The model variant (`process_improve.workbench.Variant`)

A variant is a model over a recipe's table: kind (PCA or PLS), number of components,
X and Y columns, scaling, and the excluded rows (ids). `variant.fit(table)` returns the
fitted model, plus every row's scores, Hotelling's T² and SPE, **the excluded rows
included**, projected through the model they were not used to fit, each flagged.
`variant.to_python()` writes the model code that follows the recipe's code.

## 3. Why is this group different? (`Fit.explain_group`)

Lasso a group of points on a score plot. `explain_group(group, reference=None)` returns,
for each component of the plot:

- the per-variable contributions to the difference between the group's average score
  and the rest's (`group_contributions` with `reference`), ranked by size;
- the raw data of the top variables, in run order, with the group marked, so each bar
  has its evidence beside it.

## 4. The project file (`.mvproj`, `save_project` / `load_project`)

One zip file:

| Member | Holds |
|---|---|
| `manifest.json` | format version, library version, the recipe, the variants, the saved selections |
| `source/<file name>` | the source file's own bytes, exactly as read; the recipe holds its SHA-256 |
| `recipe.py` | the recipe as a script: unzip, and it runs next to its source file |
| `fits/<k>/<array>.npy` | variant `k`'s fitted arrays: scores, loadings, T², SPE |

- Nothing executes on load: JSON checked field by field, and `numpy.load(allow_pickle=False)`.
  `recipe.py` is written for the reader and never read back.
- The data is the source file itself, so nothing is re-encoded and no dependency is
  added (Parquet would need pyarrow, a large download in the browser).
- Opening a project refits each variant from the source and the recipe, and compares
  the result with the stored arrays. A difference is reported, with what changed (the
  library version or the data), instead of silently trusted.

## 5. The selection store (`web/mva/selection.js`)

One store per page. It holds the selected rows and columns, the hovered row and column,
the excluded rows, and **named sets** (a saved lasso) with set algebra: union,
intersection, difference. Every view subscribes. Changes are batched, so a lasso of
5,000 points causes one redraw. The store saves to and loads from the project file.

## 6. Linked views (`web/mva/views.js`, `web/mva/link.js`)

`views.js` builds Plotly figures from points and the store's state, and never touches
the document:

- **Encodings:** colour by a number (a colour scale) or a category (a colour-blind
  safe palette), marker size by a number (area-proportional), symbol by a category.
- **State:** selected points full strength with an outline; the rest faded while a
  selection exists; **excluded points hollow**, still projected and drawn; the hovered
  point enlarged in every view.
- **Labels** on the selection and the extremes only, each placed on the side away from
  its neighbours, so a label never sits on another point.
- **Annotations:** the T² ellipse, origin lines, control limits, and arrows the text
  can point at.
- Score plots carry the percent of variance per component on their axes.

`link.js` binds a figure to a `<div>`: Plotly's lasso, box, click, double-click and
hover events write to the store, and a store change redraws with `Plotly.react` on the
next animation frame.

## 7. The data table (`web/mva/table.js`)

A canvas-drawn grid that renders only the visible cells, so 100,000 rows or 1,000
columns scroll smoothly. Each column header carries a histogram, a run-order sparkline,
a strip of its missing values and its quality flags. Clicking or dragging over rows
selects them in every plot; clicking a header selects the variable, which the loading
plot shows. Excluded rows show struck through, dropped ones do not show.

## Where it stands

| Building block | Status |
|---|---|
| Recipe engine, code generation, bit-for-bit test | done: `process_improve.workbench.Recipe` |
| Variant fit with projected excluded rows; group explanation | done: `Variant`, `Fit.explain_group` |
| Project file with refit-and-verify | done: `Project` |
| Selection store, figure builders, event binder | done: `selection.js`, `views.js`, `link.js` |
| Canvas table with header sparklines and an ARIA grid | done: `table.js` |
| Glue and a demo page wiring both leaps | done: `bootstrap.py`, `worker.js`, `index.html`, `app.js` |

Measured under Pyodide on the LDPE data: open 36 ms, fit 40 ms, explain a group 19 ms,
save 30 ms, open a project (refit and verify) 27 ms.

Later: the latent-space what-if with `PLS.invert`, a refit that animates from the old
model to the new, table cells coloured by their SPE contribution, branching variants
and their comparison, a Python console on the same session.

## Running the demo page

The page needs a wheel of this checkout (the release on PyPI has no workbench yet):

```bash
uv build --wheel -o web/mva/wheels
python -c "import json, pathlib; w = next(pathlib.Path('web/mva/wheels').glob('*.whl')).name; pathlib.Path('web/mva/wheels/manifest.json').write_text(json.dumps({'wheel': w}))"
python -m http.server -d web 8000   # then open http://localhost:8000/mva/
```

`web/mva/wheels/` is git-ignored. The tests:

```bash
uv run pytest tests/workbench tests/web/mva tests/web/js --no-cov
PI_CHROMIUM=/path/to/chrome uv run --with playwright==1.64.1 pytest tests/web/e2e/test_mva_page.py --e2e --no-cov
```
