# Multivariate app: feature catalogue

The features of a PCA and PLS application, grouped by kind: the reference list for
planning the multivariate browser app in this folder. This is a collection only: there
is no comparison with what `process_improve` or the page already does. The choices made
from it, and what the page is built to, are in [`URS.md`](URS.md).

**Source tags.** Each item says which sources describe it:
- no tag: source A only;
- `[B]`: source B only;
- `[A B]`: both sources, so validated by a second source.

A tag on a sub-heading covers every item under it, and a sub-item of a `[B]` item is
also `[B]`.

A later source adds its letter, so the number of letters on an item is its weight.

1. [Validated by two sources](#validated-by-two-sources)
2. [Data organization](#1-data-organization)
3. [Data handling: import, edit, merge, export](#2-data-handling-import-edit-merge-export)
4. [Preprocessing and transformation](#3-preprocessing-and-transformation)
5. [Algorithms](#4-algorithms)
6. [Batch processes](#5-batch-processes)
7. [Visualization](#6-visualization)
8. [UI expectations](#7-ui-expectations)
9. [Settings and defaults](#8-settings-and-defaults)
10. [Reporting, audit and sharing](#9-reporting-audit-and-sharing)
11. [Results catalogue](#10-results-catalogue-what-can-be-plotted-or-listed)
12. [Glossary and notation](#11-glossary-and-notation)
13. [Designed experiments](#12-designed-experiments)
14. [Model exchange and deployment](#13-model-exchange-and-deployment)
15. [Extensibility](#14-extensibility)

---

## Validated by two sources

The items both sources describe, by section: the first candidates for priority.

- **Data**:
  - tables kept apart from the models fitted to them;
  - a row's role in a model (fitted, test, removed, cross-validation segment);
  - a model's own selection of rows and columns that never changes the stored table;
  - descriptive statistics (mean, SD, skewness, kurtosis);
  - missing-value codes;
  - a log of every change.
- **Import and export**: plain text, spreadsheets, Matlab, JCAMP-DX, SPC and vendor NIR
  formats; export of data and results as text.
- **Preprocessing**:
  - multiplicative scatter correction;
  - Savitzky-Golay derivatives;
  - interaction and square terms built from centred and scaled parents;
  - per-variable centring and weights;
  - a record of every transform applied;
  - plug-in user transforms.
- **Algorithms**:
  - PCA and PLS, both by NIPALS;
  - the PLS regression vector B = W(P'W)⁻¹Q';
  - class models (one PCA model per class);
  - cross-validation;
  - the suggested number of components;
  - explained variance per component, calibration and validation, adjusted for degrees
    of freedom;
  - residual variance per row and per variable;
  - distance to the model;
  - modelling power;
  - leverage;
  - Hotelling's T² with its F limit and ellipse, identical in both;
  - jackknife uncertainty from the cross-validation models;
  - RMSE on calibration and on prediction;
  - a standard error for each predicted value;
  - standardized residuals;
  - outlier detection against limits;
  - correlation loadings.
- **Plots**: histograms; a regression line on a scatter; the T² ellipse; response
  surfaces.
- **Settings**: model centring on or off; the numbers of components used for results and
  for predictions.
- **Sharing**: a model exported for use elsewhere; a named catalogue of results.

---

## 1. Data organization

**Project and workspace**
- A project holds `[A B]`:
  - one primary table `[A B]`;
  - any number of secondary tables, for prediction and validation, with the same
    variable names and new row ids `[A B]`;
  - any number of models `[A B]`.

  B keeps each table and each model's results as separate files, numbered by default
  (data table n, result file n) `[B]`.
- Several projects can be open at once, with one tab each. Project names must be unique.
- Each row's role in a model is recorded and shown `[A B]`:
  - fitted ("workset", WS) or test set (TS), in every prediction list `[A B]`;
  - B also records "removed", "cross-validation segment n", and "frozen", a row kept in
    the calibration set of every segment `[B]`.

**Working copy ("workset")**
- Each model is built on a working copy. It records:
  - which rows and columns take part, and each column's role (X, Y or excluded) `[A B]`;
  - classes;
  - transforms, lags and expanded terms;
  - scaling and trimming.

  Changing it never changes the stored table `[A B]`. B builds interaction and square
  terms afresh from the selection each time a model is made, never storing them with the
  raw data `[B]`.
- Start a working copy:
  - new: the whole primary table, X and Y as marked at import, unit variance;
  - "new as model": a copy of another model's working copy;
  - edit: if the model is fitted, it is replaced by an unfitted one.

  Deleting a working copy deletes its model.
- Only one unfitted model at a time. Leaving the working copy creates it, active, as PCA
  of X when there is no Y and PLS when there is.
- The column roles can be saved as the default for every new working copy.
- Statistics of the working copy `[A B]`:
  - N, % missing, min, max, min/max, mean, median, SD, SD/mean, skewness, a skewness
    test and kurtosis `[A B]`;
  - percentiles: minimum, lower quartile, median, upper quartile, maximum `[B]`;
  - the precision of a group of rows (replicates), as their SD `[B]`;
  - computed on the included rows, raw or transformed and trimmed;
  - for a class model, on that class only.
- A summary line: number of rows, columns, lagged columns and expanded terms.

**Models**
- Numbered, with a title (editable in the model window) and a date. One is active; all
  results and predictions refer to it. The model records its creator and date `[A B]`.
- Model types by name: PCA of X, PCA of Y, PCA of all, PLS, OPLS, PLS-DA, class PCA,
  class PLS. B adds MLR, PCR, PLS1, PLS2, three-way PLS, analysis of effects and response
  surface `[B]`.
- Changing the model type of a model creates a new model on the same working copy.
- Component counts `[A B]`:
  - one for results and one for predictions (shown on the status line) `[A B]`;
  - B stores a suggested (optimal) count with the model, and lets residuals,
    coefficients and predictions use none, the optimal, or all components `[B]`.
- Models can be base or top models of a hierarchy, and children of a parent (per phase,
  per class).

**Identifiers, roles and classes**
- Rows and columns each have a primary id (unique, case-sensitive, long) and any number
  of secondary ids. A secondary id can be added later and filled by pasting.
- An id can be read from part of another: a start position and a length, where -1 means
  to the end, with a live sample of the result.
- Column types: X, Y, qualitative X (dummy columns in the working copy), generated
  (recorded with its formula). B also marks the design and category columns of a
  designed experiment ("levelled variables") `[B]`.
- Classes of rows, set:
  - from marked rows, the class number incrementing each time;
  - from part of an id;
  - from a range of a column or a score;
  - automatically into n classes of equal range or equal size;
  - one class per level of a qualitative column.

  Only included rows can belong to a class. Columns belong to all classes, or only to
  chosen ones.
- Kinds of process variable (a concept for the user): controlled settings, results,
  raw-material properties, intermediate results, uncontrolled conditions.

**Three-way data** `[B]`
- A three-way table: I rows × K variables in a first mode × L in a second (for example
  excitation and emission wavelengths). It is stored unfolded as I × KL, with primary
  and secondary variable modes.
- Three-way PLS fits it directly (§4.1), and user-defined analyses can act on it (§14).

**Missing values**
- A blank is missing; so are common tokens (".", "*", "$", "--", "N/A", "NaN") and a
  settable numeric code (-99 by default) `[A B]`. B writes missing as "m" in its text
  formats, or as a reserved number `[B]`.
- Each working copy has a missing-value tolerance per row and per column (50% by
  default). A check lists the rows and columns over it, and asks to exclude each one
  (yes, yes to all, no, no to all).
- Rows or columns 100% missing are excluded automatically. A zero-variance column
  prompts; if kept, its weight is 1. A last check runs just before fitting.
- Guidance: each row and column should have more than 5 values per component.
- A column with too few distinct values is excluded (the minimum is settable).
- Some operations refuse missing values: differentiation needs a complete curve `[B]`.
- A missing-value map: the whole table in pixels, missing cells in red, with the count and
  the percentage.

---

## 2. Data handling: import, edit, merge, export

**Import**
- Text files `[A B]`:
  - tab, blank, semicolon or comma separated `[A B]`;
  - with comment lines;
  - a plain matrix format with a header (rows, columns, an optional name and text), then
    row and column names `[B]`.
- Spreadsheets, with dates, one or more worksheets `[A B]`; B also reads Lotus files
  `[B]`.
- Matlab files `[A B]`.
- Design-of-experiments files.
- Spectroscopy files `[A B]`:
  - JCAMP-DX with several blocks, in its XYDATA and XYPOINTS forms `[A B]`;
  - SPC spectra `[A B]`, with their constituents read from a paired CFL file `[B]`;
  - vendor NIR formats `[A B]`;
  - 3-D fluorescence files `[B]`.
- JCAMP-DX details `[B]`:
  - X values (wavelengths) become the variable names, the Y values become the data;
  - concentrations become Y columns;
  - the sample description becomes the row id;
  - several files can be merged in one import;
  - dialects differ, so some fields may be lost.
- Instrument parameters (gain, averages, baseline correction, the instrument's own math
  pretreatments) are kept with the data, and written back on export `[B]`.
- Chromatography files (NetCDF, ANDI).
- Other chemometrics packages' files.
- Databases, through a query wizard: tables, columns, filter, sort, a saved query.
- Images, compressed to one row each (§4.8).
- An import wizard over a colour-coded preview, where the user marks:
  - the id rows and columns;
  - the X, Y and qualitative columns;
  - the excluded rows and columns;
  - the missing-value code.

  Ids are generated when the file has none. A summary page comes before finishing.

**The table as a spreadsheet**
- A context menu:
  - undo, cut, copy, paste, delete;
  - create a list, a plot, or a plot of X or Y rows;
  - add to favourites or report;
  - add rows, columns or secondary ids;
  - quick info;
  - zoom out;
  - the missing-value map;
  - the trimming overview;
  - save as text, print, properties.
- Insert, append, delete and replace rows, columns and values `[A B]`.
- Sort the table by a marked column (fitted models are kept) `[A B]`.
- Move (shift) columns within the table `[B]`.
- Lock the first n columns.
- Zoom out to see the whole table in pixels.
- Transpose (fitted models are deleted) `[A B]`.
- Index columns.
- Wildcard search and exclude.
- Keep every k-th row; exclude empty rows. B can also reduce a table by averaging groups
  of rows or columns `[B]`.

**Merge**
- Put a second file or table above-below (more rows) or side by side (more columns):
  - match by id or by position;
  - an alias file maps names that differ between files;
  - an optional column records each row's source file;
  - the source table can be deleted afterwards.
- Merge several id columns into one, with a separator.

**Export and housekeeping**
- Export a table `[A B]`: DIF, text or Matlab 4 in A; any data or result as text, and
  data or the regression vector as JCAMP-DX, in B.
- Save any list as text, any plot as an image of a chosen size; print.
- Export chosen columns or model results to an external visual-analytics tool.
- Table properties `[A B]`:
  - counts by type;
  - the import log, and the preprocessing done `[A B]`;
  - each column's type and comment;
  - B also keeps per file a notes file, a warnings file, and the display preferences in
    effect when it was closed `[B]`.
- Delete secondary tables or chosen models. Encrypt a project. Compact a project for
  deployment (§13).

---

## 3. Preprocessing and transformation

**3.1 Looking at one column or one row ("quick info")**
- A live panel for the column or row under the cursor:
  - statistics to choose from: N, % missing, min, max, min/max, mean, median, SD,
    SD/mean, skewness, kurtosis, and a generated column's formula `[A B]`;
  - B defines kurtosis as the excess (0 for a normal distribution, above 0 for a
    peaked one, below 0 for a flat or bimodal one), and skewness as 0 for a symmetric
    one `[B]`;
  - up to four plots: histogram `[A B]`, values in order (a time series, or a spectrum
    for a row), autocorrelation, power spectrum.
- Autocorrelation: detrend by mean (default), line or none; lags up to min(30, N/4).
- Power spectrum, by Welch's method, of a series of at least 16 points. Options:
  - detrending: mean, line or none;
  - scaling: unbiased or by peak height;
  - window: Hanning, Welch or Bartlett;
  - window length: the smaller of 256 and N/2, padded to 2ⁿ;
  - overlapping segments;
  - a sampling frequency, which labels the axis up to Nyquist;
  - decibels or linear.
- A scroll bar to step through the table; a "use only" subset, with find; dockable.
- Click a point to remove that row or column, or to replace that cell.
- For batch data: one phase, all batches or chosen ones.

**3.2 Trimming and winsorizing (changes the stored table)**
- Trimming sets values beyond a limit to missing; winsorizing replaces them.
- One column, by value:
  - limits as a value, or as median ± k robust SDs (interquartile range over 1.075;
    1.349 for a normal);
  - with the count and the percentage beyond each limit;
  - replace by an SD multiple, a limit, a percentile, or the last value inside (for
    time series), or exclude the rows;
  - apply and reset.
- Graphically: drag over the histogram or time series to cover what to keep (cut values
  turn yellow), or drag a band of rows to set them missing.
- The whole table or a subset, by SD multiples, values or percentiles.
- Limits read from rows of the table.
- A trimming overview: a pixel map of changed cells (red trimmed, grey winsorized).

**3.3 Generated columns (changes the stored table)**
- A wizard over the table: type an expression, preview it with statistics and quick info,
  rename it, finish. The formula is kept with the column `[A B]` (B computes a new
  matrix or vector and records it in the log).
- The language, case-insensitive:
  - vN and sets v[1,5,7:10];
  - ^ * / + - with precedence;
  - Log10, Ln, Exp;
  - set arithmetic, pairwise or against one column or a constant (set with set refused
    for * / ^).
- Functions:
  - Ones() and Index();
  - Lag(v, k) and lags of sets;
  - Diff1, Diff2 and the seasonal Delta(v, k);
  - Bins(v, k), k dummy columns;
  - row-wise Avg, Stdev, Sum and Abs;
  - Cavg.
- Plug-in user functions (prefixed, nestable) `[A B]`; B's plug-ins are whole
  transforms (§14).
- Columns from another table (D2:v[...]).
- Columns from a model's results, by model, component and column, for the fitting rows
  or the prediction set: scores, T², distances to the model, probability of belonging,
  leverages, predictions and their standard errors, residuals, Y. For example
  M2:C3:VarPred[2:4].

**3.4 Spectral filters (row-wise)**
- Each filter opens a wizard:
  - X columns and rows, or "as model";
  - optional transforms;
  - a preview on any row.
- A writes a new project: the filtered X plus untouched columns, optionally the original
  X; left-out rows as a secondary table; later secondary tables filtered automatically.
- Orthogonal signal correction (OSC): removes from X only what is orthogonal to Y.
  1. Start from a loading vector (a row of ones, the first PC of uncentred X, the
     average spectrum).
  2. t = Xp/p'p.
  3. Orthogonalize t to Y.
  4. Find w by PLS so that Xw ≈ t.
  5. Find the new p.
  6. Subtract tp' from X.

  Repeat, usually for two components. Each component shows the angle between t and Y,
  the % of X's sum of squares left, and the eigenvalue. Serious outliers should be
  removed first. The OSC model has its own score and loading plots.
- Multiplicative scatter correction `[A B]`:
  - regress each spectrum on the average, x = a + b·m `[A B]`;
  - full correction (x − a)/b `[A B]`;
  - common offset only, x − a `[B]`;
  - common amplification only, x/b `[B]`;
  - the mean spectrum is stored with the model, so new spectra are corrected the same
    way `[B]`.
- Standard normal variate: (x − x̄ᵢ)/√(Σ(x − x̄ᵢ)²/(K − 1)), per spectrum.
- Savitzky-Golay first and second derivatives `[A B]`:
  - quadratic or cubic `[A B]`, or any polynomial order `[B]`;
  - an odd window of at least 5 points `[A B]`, or the points to the left and the right
    of each point given separately `[B]`;
  - a point spacing;
  - a preview.
- Norris (gap-segment) derivative `[B]`.
- Smoothing `[B]`: a moving average over a segment around each point; Savitzky-Golay
  smoothing, a polynomial fitted to each segment (order 1 is a moving average).
- Normalization of each row `[B]`: by its mean, x/|x̄ᵢ|; by its maximum, x/max|xᵢ|;
  by its range, x/(max − min).
- Spectroscopic conversions `[B]`:
  - reflectance or transmittance to absorbance, log(1/R);
  - absorbance to reflectance, 10⁻ᴬ;
  - reflectance to Kubelka-Munk units, (1 − R)²/2R.
- Baseline correction `[B]`.
- Added noise, for testing a model's robustness `[B]`: proportional noise (a % of each
  value) and additive noise (a level, or a % of the value), x·(1 + PN/100·N(0,1)) +
  N(0, AN).
- Wavelet denoising and compression of spectra:
  - the family and order, a detrend mode;
  - DWT (multiresolution, for smooth signals) or best basis (for high-frequency
    signals);
  - the target is the average spectrum;
  - keep coefficients by variance (99.5%, from a curve) or by detail level (a table of
    levels with their count and % of energy);
  - pad to 2ⁿ;
  - denoising returns to the original domain; compression keeps the coefficients as
    columns, saving their positions;
  - loadings, coefficients and VIP are reconstructed by zero-filling and the inverse
    transform.
- OSC then wavelet, or wavelet then OSC (keep 99.9% before OSC).
- Chained filters: derivative, MSC or SNV (not both), wavelet and OSC, in sequence.
- A record of every transform applied, with its settings `[A B]`: A shows a filter
  summary; B writes each step to the table's log and into the model file, so a
  deployed model repeats it (§13).

**3.5 Time-series filters (column-wise)**
- Wavelet compression for PLS, to reduce the number of rows:
  - one target Y (several only if positively correlated: check by PCA first);
  - X and Y padded and centred;
  - DWT or best basis on the target;
  - keep by variance of Y (85%) or by detail level;
  - the same positions kept for every column;
  - no reconstruction needed, since the X-Y relationship is unchanged;
  - selecting detail levels gives a multiscale analysis.
- Wavelet denoising and decimation, column by column: remove D1 to halve the rows, D1
  and D2 to quarter them. Append to the table or start a new project.

**3.6 Working-copy level (one model only)**
- Transforms per column:
  - none;
  - linear;
  - log₁₀(C1·X + C2), and its negative;
  - exp;
  - logit, log₁₀((X − C1)/(C2 − X));
  - power, (C1·X + C2)^C3.

  For batch data, set per phase.
- Skewness, min, max and min/max per column; red marks those needing a transform;
  histograms before and after.
- Auto-transform: log the selected columns that need it, or all of them.
- Lags: any column, lags as "1-5" or "1 2 4". A lagged column is X and takes its parent's
  transform and weight.
- Expanded terms (interactions and squares) `[A B]`:
  - cross terms and squares `[A B]`; cubes and three-factor terms built by hand;
  - parents centred and scaled before multiplying `[A B]`: B centres each main variable
    on the calibration rows and divides it by its SD;
  - the model's own centring and weighting are applied after the terms are made `[B]`;
  - mixed types take the higher type;
  - can be block scaled or given a modifier;
  - changing a lag deletes the terms built on it;
  - predictions from PCA, PCR and PLS rebuild the terms automatically; an MLR or
    response-surface equation needs the stored centres and weights `[B]`.
- Scaling: weight = base × block × modifier `[A B]` (B: one centre and one weight per
  X and per Y column).
  - Base types:
    - UV;
    - UVN (uncentred);
    - Pareto, ParN;
    - centre only;
    - none;
    - freeze.
  - Centring on the mean, or none (the origin) `[A B]`.
  - Block: 1/√K or 1/K^¼ (X and Y cannot share a block).
  - Modifier: 1 by default.
  - Custom centre and weight (Frozen, FrozenC, FrozenN, UVF, NoneF); the weight given as
    the weight, the SD, or a % of the centre.
  - From two table rows or a file (names, mean, weights; per phase for batch data).
  - Weights recomputed whenever the included rows change, except frozen ones. A class
    model's weights come from its class.
  - Each column's average and SD shown beside its scaling.
- Trim or winsorize the working copy only, graphically too, with a trim summary.
- The working copy shown raw or in scaled units `[A B]` (B: the raw and the weighted
  blocks are both results).

---

## 4. Algorithms

**4.1 Model types**
- PCA of X, of Y, or of all `[A B]`; PLS `[A B]`; OPLS; one PCA, PLS or OPLS model per
  class `[A B]` (B: one PCA model per class); PLS-DA.
- B also offers `[B]`:
  - MLR, solved by singular value decomposition;
  - PCR: PCA of X, then y regressed on the scores, b = P·diag(1/τ)·P'X'y. It equals MLR
    when A = K, and is stabler when X is collinear;
  - PLS1 (one Y) and PLS2 (several Y) as separate algorithms;
  - three-way PLS (tri-PLS): X = T(W⁽²⁾ ⊙ W⁽¹⁾)' + E with the Khatri-Rao product, a
    weight vector per variable mode, inner-relation coefficients over all earlier
    scores, and a Tucker core for the X residuals.
- Hierarchical models: a model's scores and residuals become variables of a top model.
  Adding a component to a base model reverts it.
- PCA: X = 1·x̄' + TP' + E `[A B]`. PLS: X = 1·x̄' + TP' + E, Y = 1·ȳ' + UC' + F, with
  the inner relation U = T + H `[A B]`.
- Both computed by NIPALS, one component at a time `[A B]`. B spells it out `[B]`:
  - PCA starts from the column with the largest remaining sum of squares; p = X't/t't,
    normalized; t = Xp; τ = t't as the eigenvalue;
  - PLS1: w = cX'y normalized, t = Xw, p, q, then deflate X and y; no inner loop;
  - PLS2: u starts from the Y column with the largest sum of squares, and iterates;
  - convergence ‖t_old − t‖ < 10⁻¹², with at most 100 iterations;
  - T and W are normalized and orthogonal; P and Q are not.
- NIPALS with missing values, using a correction factor capped at 3.0.

**4.2 Validation and the number of components**
- Commands: autofit; the first two components; one more; zero components; remove the
  last; autofit all class models.
- Significance:
  - R1: Q² > limit. For PLS, 0 above 100 rows and 0.05 at 100 or fewer; for PCA it rises
    with the component number.
  - R2: Q²V over the limit for at least √K columns (PCA), or for one Y (PLS).
  - R3 and R5: undecided PCA components, judged by the next one's eigenvalue.
  - NS, N3 (too few degrees of freedom; PCA stops past N/2 or K/2) and N4 (under 1% of X
    or Y explained, no column over 2% or 3%).
  - For large tables, an eigenvalue limit: a component counts if its normalized
    eigenvalue is above 2.
- A suggested number of components, stored with the model `[A B]`.
- Q² = 1 − PRESS/SS per component. Q²(cum) = 1 − Π(PRESS/SS)ₐ, with PRESS/SS truncated
  for an insignificant component; A's guide states the truncation as "to −0.1", which no
  ratio of sums of squares can be (URS MOD-8). Q²V per variable, the same way. B reports the same
  quantity as the explained validation variance `[A B]`.
- PCA cross-validation predicts element (i, k) as a score times a loading. The score comes
  from a round with column k held out, the loading from a round with row i held out
  (Eastment-Krzanowski), with a special estimation that keeps a component from rotating
  into later ones.
- PLS cross-validation holds out rows: PRESS = ΣΣ(Y − Ŷ)².
- Validation methods `[A B]`:
  - cross-validation `[A B]`: 7 rounds by default in A; groups by every n-th row, by a
    score, a column or an id; B adds random or systematic segments and frozen rows
    `[B]`;
  - test-set validation on rows set aside `[A B]`, chosen at random or by hand `[B]`;
  - leverage correction `[B]`: a fast estimate without refitting, each Y residual
    divided by (1 − hᵢ), so extreme rows weigh more;
  - an iteration cap; no validation as an option.
- Whether each cross-validation segment re-centres and re-scales its data is a stated
  choice `[B]`: B re-centres but no longer re-scales, which it reports as slightly
  less conservative.

**4.3 Statistics**
- Model summary per component, from component 0 `[A B]`:
  - R²X and R²X(cum), with the eigenvalue `[A B]`;
  - R²Y and R²Y(cum) `[A B]`;
  - Q², limit and Q²(cum) `[A B]`;
  - significance;
  - iterations.

  B gives explained variance per component and cumulative, for calibration and
  validation, in %, clamped at 0 when a component adds nothing `[B]`.
- The OPLS summary: P, a rotation line, each O component, and a total.
- R² as explained sum of squares, or adjusted for degrees of freedom (variance
  explained); per variable as R²V and R²V(adj) `[A B]`.
- Degrees of freedom by context `[B]`, for example:
  - I_c − C − a for an MLR residual;
  - K − a per row;
  - (K·I_c − K·C − a·max(I_c − C, K))/K per variable.

  B notes that packages differ in this correction, so variances need not match another
  package's.
- Residual variance per column and per row, for calibration and for validation
  `[A B]`:
  - per column: Q²(cum), SD and residual SD (original and scaled), degrees of freedom,
    N `[A B]`;
  - total residual variance, also per cross-validation segment `[B]`;
  - explained variance per row `[B]`.
- Modelling power, Mpowₖ = 1 − SVₖ/SV₀ₖ, the same as B's 1 − √(ResVar(a, k)/ResVar(0, k))
  `[A B]`. Fully relevant at 1; around A/K means little relevance.
- Leverages `[A B]`:
  - in X, H₀ = T(T'T)⁻¹T', and in Y, H_Y = U(U'U)⁻¹U', proportional to T² `[A B]`;
  - B adds 1/I for the centring, hᵢ = 1/I + Σₐ tᵢₐ²/tₐ'tₐ `[B]`;
  - variable leverages, Hₖ = Σₐ wₖₐ²/wₐ'wₐ `[B]`;
  - for MLR, hᵢ = 1/I + xᵢ'(X'X)⁻¹xᵢ `[B]`.
- Hotelling's T², identical in both `[A B]`:
  - Tᵢ² = Σₐ tᵢₐ²/sₐ² `[A B]`;
  - T²·N(N − A)/(A(N² − 1)) ~ F(A, N − A), which gives the limit `[A B]`; measured, A's own
    exports use the fitted-row form, T²(N − A)/(A(N − 1)) ~ F(A, N − A), and one published
    figure of A's draws its ellipse at A·F(A, N − A) (URS STA-4);
  - the ellipse axes are (sₐ²·F(2, N−2, α)·2(N² − 1)/(N(N − 2)))^½ `[A B]`;
  - any range of components;
  - 0.05 and 0.01 limits `[A B]`;
  - a zero significance hides them.
- Distance to the model in X (DModX) `[A B]`:
  - sᵢ = √(Σeᵢₖ²/(K − A))·v, where v is a correction factor slightly above 1 for fitted
    rows `[A B]`; measured on A's screenshots, v = √(N/(N − A − A₀)) (URS STA-5);
  - normalized by the pooled RSD s₀ = √(ΣΣe²/((N − A − A₀)(K − A))) `[A B]`;
  - (sᵢ/s₀)² ~ F, which gives the critical distance and the probability of belonging
    `[A B]`; A's guide gives F(K − A, (N − A − A₀)(K − A)), but a limit A printed (1.389)
    follows no stated degrees of freedom (URS STA-5);
  - optionally weighted by modelling power.
- Distance to the model in Y (DModY): sᵢ = √(Σfᵢₘ²/M), normalized by its s₀ `[A B]` (B:
  residual Y variance per row).
- For predictions, "distance +": DModXPS+ = √(DModXPS² + d²), where d is how far the
  projection falls outside the model's score range, in residual units.
- Coefficients `[A B]`:
  - scaled and centred (CoeffCS), the same as B's weighted coefficients `[A B]`;
  - raw (y = b₀ + Σbₖxₖ + Σbₖₖxₖ² + ..., for computation only) `[A B]` (B: B and B₀ on
    the original scale);
  - Y in original units, X centred (CoeffC);
  - multiple-regression form (CoeffMR);
  - all with the centring of squared and cross terms handled;
  - B = W(P'W)⁻¹C' = W*C', and b₀ = ȳ − x̄'b `[A B]`;
  - rotated toward pure profiles, K = B(BᵀB)⁻¹.
- VIP: VIPₖ = √(Σₐ wₐₖ²·(SSYₐ₋₁ − SSYₐ)·K/(SSY₀ − SSY_A)). The sum of squares equals
  K.
- Jackknife over the cross-validation models `[A B]`:
  - the uncertainty of scores, loadings, coefficients, predictions and VIP, with
    intervals from a t value `[A B]`;
  - B's uncertainty test (after Martens) `[B]`:
    - each segment's coefficients, loadings, loading weights and scores, rotated to the
      full model;
    - their variances;
    - a p-value for every coefficient, loading and weight, to select variables.
- Error measures `[A B]`:
  - RMSE on calibration (RMSEE in A, RMSEC in B) and on prediction (RMSEP) `[A B]`;
  - B's cross-validated RMSEP pools every segment's residuals, each segment with its
    own Y weights `[B]`;
  - bias, the mean of ŷ − y, and SEP, the SD of the residuals about the bias `[B]`;
  - multiple correlation, for calibration and validation `[B]`.
- Standard error of a prediction `[A B]`:
  - A, after Höskuldsson: V(ŷ) = (1/N + t₀(T'T)⁻¹t₀')·σ², with σ² from the Y residuals
    over N − A − A₀;
  - B's prediction deviation, after De Vries and Ter Braak `[B]`: √(ResYValVar·(ResXValSampᵢ/ResXValTot
    + hᵢ + 1/I)·(1 − (a + 1)/I)).
- Residuals raw, standardized or studentized `[A B]`: B's studentized residual is
  fᵢⱼ/(σⱼ√(1 − hᵢ)), so that rows can be compared on one scale `[B]`.
- Correlation loadings: the correlation of each raw X and Y column with the scores
  `[A B]`.
- Statistical condition number of X, for MLR `[B]`.

**4.4 Diagnostics**
- Contributions (difference in scaled units × a normalized weight):
  - for scores and T² range;
  - for the distance in X (scaled residuals) and in Y;
  - for predicted Y;
  - row against average, row against row, group against average, group against group
    (not on T²).
- Weights:
  - scores, PLS: none, p, pp, w*, w*w*, pRange, w*Range, RX, coefficients, VIP;
  - scores, PCA: none, p, pp, RX;
  - distance in X: none, RX, coefficients, VIP;
  - distance in Y: none, RY;
  - predicted Y: the normalized coefficient vector.
- "Gap contribution": the difference between two rows, optionally weighted.
- Observation risk: the change in the Y residuals with the row in or out, per Y and
  pooled.
- Prediction error per training row, from the model fitted without it.
- Response permutation:
  - permute Y 20 times and refit;
  - plot R² and Q² against the correlation with the original Y;
  - the intercepts measure over-fit.
- Normal probability plot of residuals: normality, and outliers beyond about ±4 SD.

**4.5 Prediction and classification**
- The prediction set:
  - a table;
  - the rows not fitted ("complement");
  - the working copy;
  - a class;
  - a mix of rows from several of these;
  - typed or pasted in.

  It stays current. Centres can be imported.
- Predicted scores with missing values follow Nelson, Taylor and MacGregor.
- Prediction projects each new row through the model, giving its scores and residuals
  as well as ŷ `[A B]`:
  - B's "full" prediction steps through each component, giving outlier statistics `[B]`;
  - B's "short" prediction is ŷ = b₀ + x'b alone `[B]`.
- Results:
  - scores;
  - distance in X (plain or +) and its probability;
  - T²;
  - predicted Y, back-transformed by default, with standard errors and jackknife
    intervals `[A B]`;
  - residuals;
  - distance in Y and its probability;
  - hierarchical Y.
- Prediction with reference values `[A B]`: when Y is known for the prediction set, its
  RMSEP `[A B]`, and in B also SEP, bias, slope, offset, correlation, corrected SEP, and
  the slope and offset labelled ICM `[B]`.
- Cross-validated predictions for test rows, from each round's submodel, with their
  jackknife SE.
- Adaptive predictions: an EWMA of the centring of chosen columns, λ = 0.1. λ can be
  found from training rows, or updated continuously from three branches; a project of
  the centres can be saved.
- A lagged model needs the same sampling interval; the first rows (up to the maximum lag)
  are incomplete.
- Save the prediction set as a secondary table.
- Classification with one model per class `[A B]`:
  - Coomans' plot (distance to one model against distance to another);
  - a classification list or table of members `[A B]`, with probability per class model
    (green above 10%, red 5-10%, below 5% not a member);
  - B decides membership by two limits `[B]`:
    - a leverage limit, hᵢ ≤ 3(a + 1)/I;
    - a distance limit, S_max = S₀·√F_crit(1, df), with the degrees of freedom set by the
      validation method;
  - B's distance of a row to a class model is its residual SD after projection onto
    that model, for new rows and for each calibration row projected onto other models
    `[B]`;
  - model distance between two class models `[B]`;
  - discrimination power of each variable between two classes `[B]`;
  - modelling power of each variable within a class `[A B]`;
  - a PLS-DA misclassification table.
- What-if:
  - sliders per column, as a % of the value, a % of the range, or a value (±25% of the
    range by default);
  - Predict duplicates the set, the copies starred and drawn red;
  - restore, reset, save as a secondary table, or send to the batch level.

**4.6 Monitoring (control charts)**
- Multivariate charts, of t or u scores, and univariate charts of any column (on a table,
  or on a model's fitting rows only). Items include scores, X and Y columns and their
  residuals, each for the fitting rows or the prediction set.
- Shewhart:
  - individual, with 2σ and 3σ limits;
  - mean/range, subgroups of 5 by default, with UCL = D₄R̄, LCL = D₃R̄, x̄ ± A₂R̄,
    σ = R̄/d₂;
  - mean/SD, with B₃, B₄, A₃ and c₄.
- CUSUM:
  - σ by MSSD for subgroups, ordinary SD otherwise;
  - dead band K = σ/2, action limit H = 4.5σ;
  - high and low sums (SH, SL), with the deviation from target.
- EWMA, ŷₜ₊₁ = ŷₜ + λ(Yₜ − ŷₜ):
  - limits target ± 3σ√(λ/(2 − λ));
  - λ estimated to minimize the error sum of squares, or given.
- EWMA and Shewhart on one chart.
- On an individual or EWMA chart of scores, the distance to the model is drawn below,
  with its critical line.
- Target and σ estimated from the training set, or entered.
- Notation: s, target, SAvg, RAvg; the source (table, model, user); the within-group SD
  (average SD) and the between-group SD (MSSD).

**4.7 Time series and signals**
- PLS time series (ARX-style): Yₜ = Σ bᵢXₜ₋ᵢ + Σ cᵢYₜ₋ᵢ + e, as PLS on X expanded with
  lags of X and Y. Weights and coefficients plotted against lag show the important lags.
- Finite impulse response:
  - a wizard picks one Y, the X columns and rows, transforms and scaling, a maximum lag
    and an increment, then fits PLS on the lagged X;
  - the coefficients against lag form the impulse response of each X;
  - drag over it to choose the integration interval (it starts at 0), and the step
    response, its running sum, is previewed and plotted;
  - saved FIR models can be reused.
- Auto- and cross-correlation (lags 0 to 30 and −30 to +20 by default), computed by FFT,
  with optional centring or detrending.
- Wavelets:
  - DWT with fast multiresolution (Mallat) or the best basis (lowest entropy,
    Coifman-Wickerhauser);
  - families: Beylkin, Coiflet, Daubechies, Symmlet (orthogonal); biorthogonal "bior
    Nr.Nd", symmetric and linear phase;
  - choose by support (time versus frequency localization), symmetry, vanishing moments
    (compression) and regularity (smoothness);
  - Daubechies-4 is recommended for smooth spectra, a short support for best basis, and
    biorthogonal when dephasing matters.

**4.8 Images**
- Two aims:
  - each image as one row, for classification or calibration;
  - each pixel as one row, to find features within an image.
- For the first, each image is compressed to one row:
  - preprocessing: none, detrend, detrend and normalize, normalize, remove a scalar;
  - wavelets: Symmlet 4 and 8, Daubechies 4, 10 and 20, Haar;
  - optional padding to 2ⁿ;
  - a target image, compressed by variance (a curve of coefficients kept) or by scale;
  - a preview of the reconstructed image, with its compression ratio and R²;
  - a saved parameter file.
- The rows open as a project and are modelled as usual (for example PLS-DA with classes
  from the first letter of the file name).
- The model's loadings, weights and coefficients are reconstructed and shown as images.
  New images are compressed the same way for prediction.

**4.9 Automatic warnings** `[B]`

Outlier detection against limits is in both `[A B]`: A draws critical limits on its
distance and T² plots. B adds a set of automatic checks, each with a default and an
allowed range, and collects every warning in an outlier list `[B]`:

| Check | Default | Range |
|---|---|---|
| Leverage over average leverage | 3.0 | 2 to 10 |
| Increase in residual variance | 6% | 0% to 15% |
| Calibrated over validated multiple correlation | 2.0 | 1.5 to 5.0 |
| Statistical condition number (MLR) | 50 | 10 to 1000 |
| Calibrated over validated residual variance | 0.5 | 0.2 to 0.7 |
| Total explained variance, below | 20% | 5% to 90% |
| Validated over calibrated multiple correlation | 1.3 | 1.0 to 3.0 |
| Row outlier, calibration (residual over its SD) | 3.0 | 2 to 10 |
| Row outlier, validation | 2.6 | 2 to 10 |
| Variable outlier, calibration | 3.0 | 2 to 10 |
| Variable outlier, validation | 3.0 | 2 to 10 |
| Validated over calibrated residual variance | 0.75 | 0.5 to 1.0 |
| Single-value outlier, calibration | 3.0 | 2 to 10 |
| Single-value outlier, validation | 2.6 | 2 to 10 |

- Each check applies to the analyses where it makes sense (descriptive statistics, PCA,
  PCR, PLS, MLR, analysis of effects, response surface, prediction, classification).
- The limits are set in a dialog available from every model dialog. Outlier detection and
  leverage detection can each be switched off.

---

## 5. Batch processes

**5.1 The data**
- N batches × J time points × K variables, unfolded to one row per time point per batch.
  The batch and phase ids come from columns, or from part of one id.
- An optional time or maturity column per phase; otherwise a local batch time.
- Whole-batch columns (initial conditions, final quality) are set aside.
- Import checks:
  - phases out of order;
  - a phase page: rename, delete, merge, choose the maturity column, assign variables;
  - a batch page: rows per batch and phase, delete, conditional delete.
- Local centring per batch, phase and variable, from a file.
- Generated columns per phase. Transforms and scaling per phase.

**5.2 Modelling the evolution (observation level)**
- One PLS model per phase, with time or maturity as Y, under a parent model.
- The Y can be:
  - smoothed to increase strictly;
  - shifted to start at 0;
  - normalized (linear warping to the median length);
  - or treated as an ordinary response.
- Fitting rules:
  - at least 3 components;
  - continue to 85% of X, each new component above 7%, then 5%, stopping below 3%;
  - at most half the batches, half the variables, or 15.
- Reading the components: t1 is monotonic, t2 rises then falls, t3 is cubic.
- Predicted maturity as an index: ahead of or behind the normal rate.
- The batch working copy: include or exclude batches and phases; crop and downsize
  phases; assign variables; edit all phases or one; scaling from a file. The average
  batch as a table. Trimming across batches.

**5.3 Control charts and monitoring**
- Limits from the training batches: the average and SD at each time point, ±2 and ±3 SD
  or a custom list.
- Charts for:
  - scores;
  - distance to the model;
  - T² range;
  - observed and predicted maturity;
  - each variable.
- Phase, component and batches chosen; aligned or unaligned; optionally normalized.
- Out-of-control summary per batch: the area outside the limits as a % of the area
  inside, always on aligned vectors, for every chart type and for training and
  prediction batches.
- Contributions:
  - double-click a batch;
  - mark points for a comparison against the average batch at the same times, or
    against another group.
- New batches projected and drawn on the same charts.
- Observed against smoothed maturity; unaligned against aligned.
- Aligned vectors: time or maturity, distance, scores, T², X, Y and predicted Y, each
  with its average and SD.
- Mark a whole batch and exclude it everywhere.

**5.4 Modelling whole batches (batch level)**
- One row per batch. The columns can be:
  - initial conditions;
  - unfolded scores or raw variables;
  - summaries per phase (min, max, mean, median, SD, robust SD, interquartile range,
    slope, duration, end point);
  - final quality as Y.

  The choice can differ by phase.
- PCA, PLS or PLS-DA at this level shows groups and outliers, critical periods, important
  initial conditions, and the ideal evolution.
- Classify new batches; predict quality.
- Partial models: phase by phase, or at 25%, 50%, 75% and 100% of completion.
- Hierarchical batch models.
- Batch VIP per phase and per Y. A contribution as a line over a phase.
- Several batch-level models per evolution model; rebuild both levels without an outlier;
  OSC at this level.

**5.5 What-if for batches, and export**
- Sliders per variable per quarter of the batch, predicted phase by phase. Restore,
  reset, save, or send to the batch level, where initial conditions change too.
- Export a batch project for on-line monitoring.

---

## 6. Visualization

**6.1 Plot types**
- Standard: scatter (2D and 3D), line, column, histogram `[A B]`, time series, normal
  probability.
- Special:
  - contour;
  - 3D response surface `[A B]`;
  - control charts;
  - wavelet structure;
  - wavelet power spectrum;
  - step response;
  - a linked visualization dashboard;
  - lists;
  - export to an external visual-analytics tool.
- From the table: marked rows as lines; with nothing marked, every X row (spectra).
- Any line plot shown as a histogram. Any plot made from any list, and any list from any
  plot.

**6.2 The plot builder (every regular plot and list)**
- Data series:
  - the data source (primary table, a secondary table, the prediction set, a fitted
    model, the OSC model);
  - the data type (§10);
  - the item, its variable, component and cross-validation round;
  - an x-axis series plus any number of series;
  - each series scaled and offset, to separate lines;
  - scaled 0 to 1;
  - each shown in original, transformed, or working-copy units.
- Label types: the id for points and for axis labels, which part of it (start, length),
  axis-label rotation and interval, the ids in titles; saved as the default.
- Item selection: move items between selected and unselected; find text or class, X, Y,
  lags or cross terms; invert; select all.
- Colour (scatter only):
  - continuous, between begin and end colours, with colours for missing and for outside
    below or above;
  - by categories, splitting a range into n editable parts with editable colours;
  - by score categories (a score or distance split the same way);
  - by density, with n levels;
  - by predictions;
  - by models and predictions;
  - by what-if;
  - by identifiers;
  - by patterns;
  - by class, the default when classes exist.
- Size: enlarge the points whose distance is over the critical limit (size settable;
  not in 3D).
- Transformation of a series:
  - auto- or cross-correlation (detrend, maximum lag, the common vector);
  - power spectrum;
  - wavelet coefficients (family, order, detrend);
  - EWMA smoothing (subgroup, λ, target, SD: estimated or given), which tames a dense
    score trajectory.
- Confidence level (column plots): default, none, 99%, 95%.
- Number format and precision (lists).

**6.3 Standard analysis plots**
- Summary:
  - model overview;
  - X/Y overview;
  - X/Y component;
  - component contribution;
  - summary list.
- Scores:
  - t against t and u against u, with the T² ellipse `[A B]`;
  - 3D;
  - line plots with ±2 and ±3 SD;
  - t against u, with a regression line;
  - scaled by R²X.
- Loadings: p, w, w*, c, w*c; p(corr), c(corr) and pc(corr) in the unit circle `[A B]`;
  scaled by R²X; biplot.
- A guide for the reader: what score and loading plots reveal (groups, trends, outliers,
  similarity), and how a direction in one is explained by the same direction in the
  other.
- T² range with its 95% and 99% limits.
- Coefficients: an overview of every Y, or one Y with jackknife bars, by type.
- VIP sorted with bars; unsorted VIP as columns.
- Residuals; their normal probability plot. Distances to the model in X and Y, with the
  limit.
- Observed against predicted, with RMSEE or RMSEP `[A B]`.
- Observation risk.
- The permutation plot.
- Coomans' plot.
- Time series of observed and predicted Y.
- Contributions from any of these.
- Loadings, weights, VIP and coefficients of a lagged variable against lag.

**6.4 Special plots**
- Contour and response surface `[A B]`:
  - model, components, Y, two X on the axes with low and high limits;
  - the other X held at low, centre, high or a custom value, shown beside the plot and
    editable live;
  - resolution, number of levels, colour, level labels;
  - B finds and reports the maximum, minimum or saddle point of a response surface
    `[B]`.
- Wavelet structure:
  - the signal A0 with its approximations A1 to A4 and details D1 to D4;
  - a bar chart of the % of the sum of squares per detail level (without the mean);
  - any levels chosen;
  - scale 1 is the highest frequency.
- Wavelet power spectrum: a 3D plot of normalized squared coefficients over time and
  scale (octave bands up to Nyquist). A signal changing frequency shows as peaks moving
  between scales.
- Visualization dashboard:
  - four linked plots (a score scatter beside line or histogram plots of three
    variables);
  - a property bar to choose the model, components and the colouring variable;
  - marking in one marks in all;
  - right-click to change a plot's type, reusing its window.

**6.5 Encodings, annotations and styling**
- Labels (cap 500), free text, shaded alarm zones, regression lines `[A B]`, ellipses,
  limits and bars at a chosen confidence, a header and a time-stamped footer.
- Statistics of any 2D scatter, printed on it `[B]`:
  - the least-squares line, slope and offset;
  - bias, the mean of y − x;
  - the correlation coefficient;
  - RMSED and SED, the root mean square of y − x and its SD about the bias.
- Cell colours in lists: out-of-limit predictions in red; classification probabilities
  green or red; misclassifications highlighted; significant components in green.
- Axes:
  - range, ticks, decimals;
  - scientific or time labels;
  - log or reversed;
  - a second Y axis.
- Grids, titles, fonts. Line, symbol and column styles; restyle marked points. Legend
  and backgrounds. Contour and 3D settings. Save a style as the default; restore the
  factory styles.

---

## 7. UI expectations

**7.1 Selection ("marking")**
- Marker shapes: lasso, box, x band, y band, none; batch marking.
- Ctrl keeps the marks; a click on empty space clears them.
- Find:
  - wildcards (? and *), case-sensitive;
  - ids or comments;
  - begins with, contains, exact;
  - select all, complement.
- Find by rule; the result replaces, joins or intersects the marks.
- A marking pane: include, exclude, set a class, relabel; the previous and current
  groups.

**7.2 Acting on a selection**
- With the contribution tool:
  - double-click a point for its contributions against the average;
  - click one point, then double-click another, for the difference between them.
- A default-action button that picks the contribution to show from what is marked.
- A flag tool shows the raw value behind a contribution bar.
- Exclude or include marked points from any plot. This creates a new working copy and an
  unfitted model, and keeps the old model.
- Set a class; label or restyle the marks; list them; plot their raw values.
- Plot from the table by marking rows or columns.

**7.3 Plots and lists**
- A right-click menu on every plot and list:
  - copy;
  - create a list or a plot;
  - show as histogram;
  - add to favourites, add to report;
  - quick info;
  - sort;
  - plot settings;
  - maximize the plot area;
  - save as the default style;
  - save, print;
  - properties.
- The properties dialog is the plot builder (§6.2).
- A list warns when it would include every item, and asks for only those needed.
- Sort lists and bar charts by value, or by part of an id then value.
- Zoom (box, x, y, subplot) and zoom out; coordinates; 3D rotation; hover gives the id,
  the series and the value.
- An item-information pane, which draws the chemical structure when an id is a SMILES
  string.
- Lock the first columns of a table.
- An outlier list collects every warning a model raised (§4.9) `[B]`.

**7.4 Windows and navigation**
- Arrow keys change components. Page Up and Page Down change the model (or the phase).
- A property bar (model, components, column) and a status bar (model, type, A, A for
  prediction, R²X, R²Y, Q², adaptive).
- The model window with its summary and buttons to its options and working copy.
- The working copy as tabs: overview, variables, observations, transform, lag, expand,
  scale, spreadsheet. Each has "as model", and selections persist across tabs.
- Toolbars (standard, shortcut, customizable plot toolbar); tooltips and status-bar
  hints.
- Dockable panes:
  - audit trail;
  - advisor;
  - favourites;
  - item information;
  - marked items;
  - observations;
  - variables;
  - what-if;
  - quick info.
- Favourites: commands and plots in folders; run a whole folder.

**7.5 Guidance and help**
- A working-copy wizard (simple mode) that states the model it will build and fits it;
  the advanced mode is the tabbed window.
- An advisor explaining each analysis plot; a tutorial with example data and example
  results `[A B]`.
- A help system `[A B]`:
  - contents, index, full-text search (match similar words, titles only);
  - a Help button in every dialog.
- A glossary of terms, with a reading list of the method papers behind each technique
  (§11) `[A B]`.
- The analysis cycle:
  1. Import.
  2. Preprocess.
  3. Specify the working copy.
  4. Fit and interpret.
  5. Predict.
- Cautions:
  - undesigned data do not show cause and effect;
  - coefficients are only independent when X comes from a design;
  - add cross terms only when X comes from a response-surface design.

---

## 8. Settings and defaults

- Levels `[A B]`:
  - factory;
  - global `[A B]` (B: a defaults file per user);
  - project;
  - model, which overrides the project's.

  A fitted model keeps its options until it is reset.
- Fit:
  - cross-validation rounds;
  - segment selection, random or systematic `[B]`;
  - iteration cap;
  - missing tolerance (50%);
  - minimum distinct values;
  - confidence level (95%);
  - significance level (0.05);
  - wavelet reconstruction on or off;
  - centring on the mean or none `[A B]`;
  - the warning limits, and outlier and leverage detection on or off (§4.9) `[B]`.
- Predictions: units of predicted Y; apply the working copy's trimming; adaptive λ (0.1).
- Distance: normalized or absolute; weighted by modelling power or not.
- Labels: the default id and its part; title ids; a 500-label cap.
- Residuals raw or standardized; R² explained or adjusted; coefficient type;
  cross-validation groups; components for results and predictions `[A B]`.
- General:
  - open the last project on start-up;
  - folders (including the plug-in folder);
  - tab position;
  - simple or advanced mode;
  - print lists in colour;
  - caching.
- Each file remembers its display preferences when it closes `[B]`.
- Restore factory plot styles, favourites, toolbars and "don't show again" messages.

---

## 9. Reporting, audit and sharing

**Report writer**
- Templates: blank, a default per model type, or the user's own. The default sections:
  1. Objective.
  2. Data.
  3. Preprocessing.
  4. Model summary.
  5. Transformation and scaling.
  6. Goodness of fit.
  7. Results.
  8. Predictions.
  9. Conclusion.
- Live placeholders (update, reset, freeze). Add plots and lists from their menus or by
  grabbing them. Insert links, images and files.
- Append, insert or replace another model's report; choose batch phases.
- Plot size and image format. HTML, browser view, e-mail, print.

**Audit and sharing**
- A log of every change `[A B]`:
  - A: an audit trail of every event, with the user, date and time, exported as XML, and
    an administrator lock (21 CFR Part 11);
  - B: a log per table of every change (append, compute, delete, import, insert,
    replace, shift, sort, transform, and each analysis), readable as plain text, with
    short codes for each operation `[B]`.
- Free-text notes per table and per model result `[A B]`.
- Encrypt; compact for deployment (§13); export to an external visual-analytics tool.

---

## 10. Results catalogue (what can be plotted or listed)

Every quantity has a name, in both sources `[A B]`. A's suffixes say where a result
comes from:
- PS: the prediction set;
- DS: a table;
- cv: one cross-validation round;
- cvSE: its jackknife SE over the rounds;
- St: standardized.

B names each result matrix, with calibration ("Cal") and validation ("Val") versions,
and per-segment versions for cross-validation `[B]`.

Data types:

**Functions of rows (one value per row)**
- Scores t, u, tPS, tcv, tcvSE, tPScv, tPScvSE `[A B]`; B also has preliminary scores,
  and scores within each class model `[B]`.
- T², T²PS, T² range, T² range PS `[A B]`.
- DModX, DModXPS and DModXPS+, each absolute, normalized or modelling-power weighted
  `[A B]`.
- DModY, DModYPS. PModX, PModXPS, PModXPS+, PModY, PModYPS.
- Leverages in X and Y `[A B]`, and within each class model `[B]`. Observation risk,
  per Y and pooled.
- Explained and residual variance per row, calibration and validation `[B]`.
- X and Y columns from the working copy, the prediction set or a table; X reconstructed
  from TP'. The raw and the weighted blocks `[A B]`.
- Predicted Y, its standard errors (lower, upper, for fitted and predicted rows), its
  cross-validated values and their SE, and the prediction error with the row left out
  `[A B]`. B adds the prediction deviation, and predicted Y for calibration and for
  validation `[B]`.
- X and Y residuals, raw and standardized `[A B]`, and studentized `[B]`, for fitted and
  predicted rows.
- Classification per row: distance to each class model, its ratio to the class's
  typical distance, and membership `[B]`.
- The cross-validation group of each row `[A B]`; the index number; the row id.

**Functions of columns (one value per variable)**
- Averages of X and Y; scaling weights of X and Y `[A B]`.
- p, w, w*, c, w*c, pc; p(corr), c(corr), pc(corr) `[A B]`; with cv and cvSE versions.
- For three-way PLS: loading weights for the primary and the secondary variable modes
  `[B]`.
- Coefficients: Coeff, CoeffC, CoeffCS, CoeffMR `[A B]`, with cv and cvSE versions; B
  also has each segment's coefficients and their p-values `[B]`.
- p-values for every loading and loading weight `[B]`.
- VIP and its cv and cvSE; batch VIP.
- Modelling power `[A B]`; discrimination power between two classes `[B]`.
- Variable leverage `[B]`.
- Q²V and Q²V(cum); R²V, R²V(cum), R²V(adj) and R²V(adj)(cum), for X and Y `[A B]`.
- Residual variances S2VX and S2VY (at component 0, the working copy's variances)
  `[A B]`.
- RMSEE, RMSEP `[A B]`; bias, SEP, multiple correlation `[B]`; X bias and Y bias `[B]`.
- Statistics per variable: mean, SD, N, percentiles, correlations, precision `[A B]`.
- A row of X or Y (fitted, predicted, or from a table), its reconstruction and its
  residuals.

**Functions of the number of components**
- Eigenvalue and iterations `[A B]`.
- Q² and Q²(cum); Q²V and Q²V(cum) `[A B]`.
- R² and R²(cum), and their adjusted forms, for the whole block and per variable
  `[A B]`.
- S2X, S2Y; SSX, SSY `[A B]`.
- Total residual and explained variance per cross-validation segment `[B]`.
- The SD of t and of u `[A B]`.
- The jackknife SE of the fitted Y.

**Functions of lags**
- p, w, w*, VIP and CoeffCS of a lagged variable, against lag.

**Designed experiments** `[B]`
- Effects and their significance; ANOVA tables; lack of fit; t- and p-values for a
  curvature test; multiple comparisons; the maximum, minimum or saddle point of a
  response surface, with the predicted Y there (§12).

**Batch**
- Aligned vectors (training and prediction): time or maturity, distance, scores, T²
  range, X, Y and predicted Y.
- Batch vectors: the out-of-control sum per batch for each of them.

---

## 11. Glossary and notation

**Glossary**
- Time series and control: AR, ARMA and MA models; ARL; CUSUM; EWMA; SPC; MSPC; VSI.
- Data: batch process; class `[A B]`; data set (N observations × K variables)
  `[A B]`; reference (training, calibration) set `[A B]`; test set `[A B]`; prediction
  set; missing value; ordinal data.
- Variables: dependent, response, explanatory, predictor, independent (and why
  "independent" misleads); factor.
- Methods:
  - PCA (and its other names: characteristic-vector, eigenvector, Karhunen-Loève)
    `[A B]`;
  - PLS (projection to latent structures) `[A B]`;
  - PCR `[B]`;
  - discriminant analysis;
  - cluster analysis;
  - correspondence analysis;
  - factor analysis;
  - multidimensional scaling;
  - principal coordinate analysis;
  - multiple (linear) regression `[A B]`;
  - NIPALS `[A B]`;
  - the power method;
  - SVD `[A B]`;
  - projection and subspace methods.
- Geometry:
  - bilinear model;
  - loading and score vectors `[A B]`;
  - projection point;
  - column, row, variable and observation spaces (K-space, M-space);
  - inner and outer products;
  - the Kronecker and Khatri-Rao products, for three-way data `[B]`;
  - eigenvalue and eigenvector `[A B]`;
  - Euclidean and Mahalanobis distance.
- Diagnostics:
  - cross-validation `[A B]`;
  - jackknife `[A B]`;
  - leverage `[A B]`;
  - residual `[A B]`;
  - gap contribution;
  - blockwise scaling;
  - closed form and non-linear;
  - Pareto's principle.

**Notation** `[B]`

B states its symbols in one table:

| Symbol | Meaning |
|---|---|
| a, A | component number, number of components |
| b₀, **B** | intercept, regression coefficients |
| C | 1 if the model is centred, 0 if not |
| d | degrees of freedom |
| **E**, **F** | X and Y residuals for a model with a components |
| h, **H** | leverage |
| i, I | row number, number of rows |
| j, J | Y column number, number of Y columns |
| k, K | X column number, number of X columns |
| **p**, **P** | X loadings |
| **q**, **Q** | Y loadings |
| **t**, **T** | scores |
| **u**, **U** | preliminary (Y) scores |
| **w**, **W** | X loading weights |
| x̄, ȳ, ŷ | means, and a predicted value |

The two sources use different letters for the same counts (rows: N in A, I in B; Y
columns: M in A, J in B). The app should pick one set and use it everywhere.

---

## 12. Designed experiments

Everything in this section is `[B]`. A imports designs from a separate tool (§2), and its cautions refer to them: undesigned
data do not show cause and effect, and cross terms belong to response-surface designs.
B builds and analyses designs itself.

**Designs**
- Full factorial, fractional factorial, Plackett-Burman, combined designs.
- Central composite and Box-Behnken designs, for response surfaces.
- D-optimal designs.
- Mixture designs: lattice, centroid, axial, and D-optimal mixture designs.

**Analysis of effects**
- MLR on the coded levels (low −1, high +1); an effect is twice its coefficient, the
  average change in the response when the factor goes from low to high.
- Interactions and squares from the coded levels, without standardization.

**Response surface analysis**
- MLR with the main variables centred first; ANOVA table, lack of fit, a curvature test.
- The maximum, minimum or saddle point of the fitted surface.

**Significance testing** (MLR-based methods)
- Standard error of each coefficient: σ√((X'X)⁻¹ₖₖ), with σ the RMSE of calibration;
  for b₀, σ√(1/I + x̄'(X'X)⁻¹x̄).
- t = b/SE(b); F = t²; two-sided p-values from F(1, I − K − C).
- Multiple comparisons (Tukey's test), for a factor with more than two levels.
- When there are no residual degrees of freedom:
  - COSCIND, a scale-independent statistic from the sorted effects, with p-values by
    Cochran's approximation;
  - higher-order interaction effects used as the error term (HOIE), from the cube
    samples.

**Mixtures and D-optimal selection**
- The mixture region: a simplex when, for every component, MixSum − U_q ≤ Σᵢ≠q Lᵢ, and
  there are no further linear constraints.
- Bound consistency: bounds that cannot be reached are tightened until they touch the
  region. An empty region is reported, with which bounds to raise or lower.
- Extreme vertices of a constrained region, and the centroids of its faces, as candidate
  points (Cornell's CONVRT and CONAEV).
- D-optimal selection by point exchange (Fedorov's algorithm, as in Miller and Nguyen's
  DOPT):
  1. Start from a default or random set.
  2. Swap a selected point for the candidate that most increases det(X'X).
  3. Repeat while det(X'X) still improves.

**Predictions**
- From an MLR or response-surface model, by its coefficient equation, with the main
  variables centred and the interaction terms rebuilt from the stored centres and
  weights (§3.6).

---

## 13. Model exchange and deployment

A model can be compacted or exported for use elsewhere `[A B]`: A compacts a project for
deployment, and exports a batch project for on-line monitoring. B writes a model file
built for prediction outside the application `[B]`:

**The model file** `[B]`
- Plain text, everything needed for prediction and classification.
- "Mini": the coefficients and intercepts only, for prediction alone.
- "Full" adds what outlier statistics need:
  - X and Y weights and centres;
  - residual variances;
  - loadings and loading weights;
  - sums of squares;
  - the mean calibration leverage;
  - RMSE of calibration;
  - the SD of the scores;
  - calibration means, SDs and values.
- A header:
  - type, format version;
  - model name and date, creator;
  - method;
  - calibration data name;
  - numbers of rows, X and Y columns;
  - validation method;
  - components present, and the suggested number;
  - centring;
  - calibration and test rows, cross-validation segments;
  - instrument parameters;
  - the list of matrices.
- One line per preprocessing step, with its parameters, so the deployed model repeats the
  preprocessing (§3.4). A step that needs stored data (MSC's mean spectrum) refers to a
  matrix in the file, and is flagged as an error when that matrix is missing.
- Each matrix as a header (name, rows, columns) followed by its values; missing as "m".
- The prediction-with-reference statistics, when Y was known (§4.5).

**Instruments** `[B]`
- Read the data files of common spectrometer software, and keep the instrument's
  parameters with the data (§2).
- Write data, or the regression vector, as JCAMP-DX for an instrument to use.
- Write model files that instrument software reads directly.
- Record the instrument's own pretreatments, so they are not applied twice: smoothing,
  derivatives (first to fourth), Savitzky-Golay, Kubelka-Munk, division by wavelength,
  Fourier transform, reference correction, full MSC.

---

## 14. Extensibility

Both sources let the user add their own functions `[A B]`: A through plug-in
functions in the expression language (§3.3) and a plug-in folder (§8); B through
installable transform and analysis components `[B]`.

**Plug-in components** `[B]`
- A user-defined transform acts on the current selection: the chosen rows and columns as
  one matrix, numbered from 0. Generated interaction terms are not part of it.
- It declares what it may do for a matrix of a given size: insert or delete rows, insert
  or delete columns. Anything not declared is refused.
- It can show its own settings dialog, and gives a one-line description of its settings
  for the table's log.
- It stores and restores its settings, so they are saved with the file and the step can
  be repeated.
- It reads and writes single values, whole rows and whole columns. Rows and columns it
  inserts get a name and start as missing. It knows the value that means "missing".
- Every call returns a status, and anything but success is an error.
- User-defined analyses work the same way on three-way tables.
- Development files and sample components are provided.
