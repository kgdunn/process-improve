# Multivariate app: user requirements

The requirements this app is built to, chosen line by line from a feature catalogue in the review of
24 October 2026. Each line has an id to quote in commits and pull requests, says what it covers, and
says how much of it `process_improve` (the library or this page) already has. A line marked *in* or
*partly* is mostly a matter of wiring the existing code into the page.

## Decisions

- **Where it runs.** In the browser, on Pyodide, as `web/mva/`: no server, and the data never leaves
  the machine.
- **Data.** Tables of named variables first. Wide data (spectra) must stay possible: nothing may assume
  a narrow table, and loadings and coefficients must be able to draw as lines along an axis. The spectral
  filters wait.
- **Models update at once.** A change (rows excluded, scaling, a recipe step) makes a new model, fitted
  immediately and linked to the model it came from. The old model is kept, so any two can be compared.
  This replaces the desktop habit of one unfitted model and a Fit button.
- **Projects.** One project per browser tab; tabs inside the app are for models.
- **Reports** are written three ways: an HTML report, a runnable notebook that rebuilds every step from
  the source file, and a Word document.
- **Exported models** carry their preprocessing steps, so whoever uses them repeats the preprocessing.
- **Maybe** lines are built only when they fall out of a Definitely line at almost no cost.
- **Designed experiments** are out of this tool: the designed-experiments app covers them.
- **Batch data** comes later, together with PLS-DA and the drawing of chemical structures.

In numbers: 91 Definitely lines, of which 13 are in process-improve and 57 partly.

## Requirements

### PRJ: Projects and models

- **PRJ-1** A project holds a primary table, secondary tables for prediction and validation, and many models. *(partly in process-improve)*
  - A project holds one primary table, any number of secondary tables (the same columns, new row ids, for prediction and validation) and any number of models.
  - Already there: Project (.mvproj): one source table, its recipe and its model variants in one file.
- **PRJ-2** Several projects open at once, one per browser tab. *(not in process-improve)*
  - Several projects can be open at once, one tab each. Project names are unique.
- **PRJ-3** Each row's role is shown everywhere: fitted, test, excluded, cross-validation segment, frozen. *(partly in process-improve)*
  - Each row's role in each model is recorded and shown in every list and plot: fitted, test set, excluded, cross-validation segment n, or frozen (kept in the calibration set of every segment).
  - Already there: Variant.fit flags the excluded rows and still projects them; the page draws them hollow and the table strikes them through. No test set, segment or frozen role.
- **PRJ-4** Each model has its own working copy; changing it never changes the stored table. *(partly in process-improve)*
  - Each model has its own working copy that records the rows and columns taking part, each column's role (X, Y or excluded), classes, transforms, lags, expanded terms, scaling and trimming. Changing the working copy never changes the stored table.
  - A summary line: the number of rows, columns, lagged columns and expanded terms.
  - Already there: Variant: model kind, X and Y columns, excluded rows, number of components. Recipe steps change the table for every model. No per-model classes, transforms, lags or scaling. The page shows the table's size and the number of rows and columns selected.
  - **PRJ-4.1** Start a working copy new or as a copy of another model's; edit and delete it. *(partly in process-improve)*
    - Start a working copy as new (the whole primary table, X and Y as marked at import, unit variance) or as a copy of another model's. Editing a model makes a new one, fitted at once; the old one is kept. Deleting a working copy deletes its model.
    - Save the column roles as the default for every new working copy.
    - Already there: Variant is a frozen dataclass, so a copy is one call (dataclasses.replace). The page holds one model.
  - **PRJ-4.2** Each change makes a new model, fitted at once and linked to the model it came from; the old one is kept. *(not in process-improve)*
    - A new model is PCA of X when there is no Y, and PLS when there is; it is fitted at once, and the model it came from is kept.
- **PRJ-5** Descriptive statistics of the working copy, on its included rows. *(partly in process-improve)*
  - Descriptive statistics of the working copy, on its included rows, raw or transformed: N, % missing, min, max, min/max, mean, median, SD, SD/mean, skewness with a test, excess kurtosis, and the percentiles (minimum, quartiles, median, maximum).
  - Already there: univariate.summary_stats: N, mean, SD, RSD, min, max, median, the 5, 25, 75 and 95 percentiles and the IQR. No skewness, kurtosis or % missing.
- **PRJ-6** Numbered, titled models; one active model that every result refers to. *(partly in process-improve)*
  - Models are numbered, with an editable title, a creator and a date. One model is active, and every result and prediction refers to it.
  - Already there: Variant.name; the page holds one model.
  - **PRJ-6.1** Separate component counts for results and for predictions; the suggested count stored. *(partly in process-improve)*
    - Two component counts per model, one for results and one for predictions, both on the status line. The suggested count is stored with the model, and residuals, coefficients and predictions can use none, the suggested or all components.
    - Already there: Variant.n_components; CrossValidation.recommended.
- **PRJ-7** Primary and secondary ids for rows and columns, also read from part of another id. *(partly in process-improve)*
  - Rows and columns each have a primary id (unique, case-sensitive) and any number of secondary ids, added later or pasted in. An id can be read from part of another (a start and a length), with a live preview.
  - Already there: SetIndex makes a column the row ids; every point and cell in the page carries its row id and column id.
- **PRJ-8** Column types: X, Y, qualitative (as dummy columns), generated (with its formula), design. *(partly in process-improve)*
  - Column types: X, Y, qualitative X (expanded to dummy columns in the working copy), generated (kept with its formula), and the design and category columns of a designed experiment.
  - Already there: AddColumn keeps its expression; PLSDA one-hot encodes a class column; FAMD and MCA analyse categorical columns.
- **PRJ-9** Classes of rows, set from marks, ids, value ranges, equal bins or a qualitative column. *(partly in process-improve)*
  - Classes of rows, set from marked rows (the class number increments each time), from part of an id, from a range of a column or a score, into n classes of equal range or size, or one per level of a qualitative column. Only included rows belong to a class; a column belongs to all classes or to chosen ones.
  - Already there: Named sets in the selection store: a saved lasso, combined by union, intersection and difference.
- **PRJ-10** Missing values: settable codes, a tolerance check before fitting, a missing-value map. *(partly in process-improve)*
  - Missing values: a blank, the common tokens (".", "*", "$", "--", "N/A", "NaN") and a settable numeric code (-99 by default) all read as missing.
  - A missing-value check before fitting: a tolerance per row and per column (50% by default), listing those over it and offering to exclude each; fully missing rows and columns excluded automatically; a zero-variance column flagged; a column with too few distinct values excluded (minimum settable); more than 5 values per component advised.
  - A missing-value map: the whole table in pixels, missing cells in red, with the count and the percentage.
  - Already there: SourceFile read options take na_values; blank, N/A and NaN read as missing by default. PCA and PLS fit with missing values (missing_data_settings). No tolerance check. The table shows each column's missing values as a strip in its header.

### DAT: Data: import, edit, export

- **DAT-1** Import delimited text and spreadsheets (several sheets, dates). *(partly in process-improve)*
  - Import delimited text: tab, blank, semicolon or comma separated, with comment lines; also a plain matrix format with a header and row and column names.
  - Import spreadsheets, with dates, from one or more worksheets.
  - Already there: SourceFile reads CSV with a chosen separator, decimal mark, header, skipped rows, missing codes and encoding; the page opens CSV files. SourceFile reads Excel, one chosen sheet.
- **DAT-4** An import wizard on a colour-coded preview: mark ids, X, Y, excluded rows, missing codes. *(partly in process-improve)*
  - An import wizard over a colour-coded preview, marking the id rows and columns, the X, Y and qualitative columns, the excluded rows and columns, and the missing-value code. Ids are generated when the file has none; a summary page comes before finishing.
  - Already there: The page opens a file into the table; Use as Y marks the Y columns; recipe steps set the row ids and drop rows and columns.
- **DAT-5** Edit the table as a spreadsheet: insert, delete, replace, sort, transpose, find, undo. *(partly in process-improve)*
  - The table works as a spreadsheet, with a context menu for its actions (undo, cut, copy, paste, delete; create a list or plot; add rows, columns or secondary ids; quick info; save as text, print, properties).
  - Insert, append, delete and replace rows, columns and values.
  - Sort the table by a column (fitted models are kept), move columns, and lock the first n columns.
  - Transpose the table (fitted models are deleted).
  - Zoom out to see the whole table in pixels.
  - Index columns; wildcard search and exclude; keep every k-th row; exclude empty rows; reduce a table by averaging groups of rows or columns.
  - Already there: The canvas table renders 100,000 rows; recipe steps with undo; clicking rows selects them in every plot. DropRows, DropColumns, FilterRows and AddColumn recipe steps. KeepColumns sets the column order; rows keep their ids through any change. FilterRows keeps the rows where an expression is true.
- **DAT-6** Merge tables by rows or by columns, matched by id or by position. *(not in process-improve)*
  - Merge a second file or table above-below (more rows) or side by side (more columns), matching by id or by position, with an alias file for names that differ and a column for each row's source; merge several id columns into one with a separator.
- **DAT-7** Save any list as text and any plot as an image; print. *(partly in process-improve)*
  - Save any list as text and any plot as an image of a chosen size; print.
  - Already there: Every Plotly chart in the page has a PNG download in its toolbar.
- **DAT-9** Table properties and housekeeping: counts by type, import log, notes; delete, encrypt, compact. *(partly in process-improve)*
  - Table properties: counts by type, the import log and the preprocessing done, each column's type and comment, and per table a notes file, a warnings file and the display preferences in effect when it was closed.
  - Delete secondary tables or chosen models; encrypt a project; compact a project for deployment.
  - Already there: The recipe records every step from the source file on, and writes it as Python.

### PRE: Preprocessing

- **PRE-1** Quick info: live statistics and plots for the column or row under the cursor. *(partly in process-improve)*
  - Quick info: a live panel for the column or row under the cursor, with chosen statistics and up to four plots: histogram, values in order (a time series, or a spectrum for a row), autocorrelation, power spectrum.
  - Autocorrelation, detrended by the mean (default), a line or nothing, up to min(30, N/4) lags.
  - Power spectrum by Welch's method (16 points or more): detrending, scaling, window type and length, overlap, a sampling frequency up to Nyquist, decibels or linear.
  - Step through the table, limit the panel to a subset, dock it; click a point to remove its row or column or to replace the cell; for batch data, one phase and all or chosen batches.
  - Already there: Each table column header carries a histogram, a run-order sparkline and its missing values. univariate.autocorrelation, with its approximate standard error.
- **PRE-2** Trim or winsorize values beyond limits, by value, by percentile, or by dragging on a plot. *(not in process-improve)*
  - Trim (set to missing) or winsorize (replace) values beyond limits, changing the stored table.
  - For one column: limits as values or as the median plus or minus k robust SDs, with the count and % beyond each; replace by an SD multiple, a limit, a percentile or the last value inside, or exclude the rows; apply and reset.
  - Graphically: drag over the histogram or the time series (cut values turn yellow), or drag a band of rows to set them missing.
  - For the whole table or a subset, by SD multiples, values or percentiles, or with limits read from rows of the table.
  - A trimming overview: a pixel map of the changed cells (red trimmed, grey winsorized).
  - Trim or winsorize the working copy only, graphically too, with a trim summary.
- **PRE-3** Generated columns from an expression, kept with their formula. *(in process-improve)*
  - Generated columns from an expression, previewed with statistics before finishing; the formula is kept with the column and recorded in the log.
  - Already there: AddColumn recipe step: a parsed, safe expression, kept in the recipe and written as Python; the page's Add column.
- **PRE-7** Expanded terms: cross and square terms from centred, scaled parents. *(partly in process-improve)*
  - Expanded terms: cross terms and squares (cubes and three-factor terms by hand), built from parents centred and scaled on the calibration rows, the model's own centring and weighting applied afterwards; predictions rebuild the terms.
  - Already there: interactions.interaction_terms builds two-factor terms from centred, scaled parents and re-centres them; pair_coverage says whether the data can support each pair.
- **PRE-8** Scaling per column: unit variance, Pareto, centre only, none, frozen, block weights. *(partly in process-improve)*
  - Scaling per column, weight = base × block × modifier, with one centre and one weight per X and Y column.
  - Base types: unit variance, unit variance uncentred, Pareto (centred or not), centre only, none, frozen.
  - Centring on the mean, or none (through the origin).
  - Block weights (1/√K or 1/K^¼) and a modifier (1 by default).
  - Custom centres and weights (frozen), given as the weight, the SD or a % of the centre, from two table rows or a file (per phase for batch data).
  - Weights recomputed whenever the included rows change, except frozen ones; a class model's weights come from its class; each column's average and SD shown beside its scaling.
  - Show the working copy raw or in scaled units.
  - Already there: MCUVScaler (unit variance); chemistry.center_and_scale (autoscale or Pareto); MBPCA and MBPLS weight the blocks. The page always uses unit variance. Unit variance and Pareto in the library. PLS(scale=False) and PCA on uncentred data; MCUVScaler centres. MBPCA and MBPLS weight each block. Variant.fit scales on the included rows only.

### MOD: Models and validation

- **MOD-1** Model types: PCA, PLS, OPLS, and one model per class. *(partly in process-improve)*
  - Model types: PCA of X, of Y, or of all; PLS; OPLS; one PCA, PLS or OPLS model per class. Changing the model type creates a new model on the same working copy.
  - Already there: PCA, PLS, OPLS and PLSDA in the library; the page offers PCA and PLS.
- **MOD-2** Also MLR, PCR, and PLS1 and PLS2 as separate algorithms. *(partly in process-improve)*
  - Further model types: MLR (solved by SVD), PCR, and PLS1 and PLS2 as separate algorithms.
  - The statistical condition number of X, for MLR.
  - Already there: regression.OLS and multiple_linear_regression; PLS takes one or several Y. No PCR.
- **MOD-4** Hierarchical, multi-block models. *(partly in process-improve)*
  - Hierarchical models: a model's scores and residuals become variables of a top model; base and top models, and children of a parent model (per phase, per class).
  - Already there: MBPCA and MBPLS: multi-block (hierarchical, consensus) PCA and PLS.
- **MOD-5** NIPALS fitting, with missing values. *(in process-improve)*
  - PCA and PLS by NIPALS, one component at a time, with the model forms stated (PCA: X = 1·x̄' + TP' + E; PLS adds Y = 1·ȳ' + UC' + F and U = T + H), a stated convergence rule and iteration cap, T and W orthonormal.
  - Fit with missing values.
  - Already there: PCA and PLS fit by NIPALS (or SVD for complete PCA data), with tol and max_iter. missing_data_settings on PCA, PLS, MBPCA and MBPLS; project() handles missing cells in new rows.
- **MOD-6** Fit commands: autofit, one more component, remove the last, autofit all class models. *(partly in process-improve)*
  - Fit commands: autofit, the first two components, one more, none, remove the last, and autofit every class model.
  - Already there: PCA.select_n_components and PLS.select_n_components choose the count; the page's Cross-validate chart sets the count by a click on its bar.
- **MOD-7** Significance rules decide each component and show why; the suggested number is stored. *(partly in process-improve)*
  - Significance rules decide each component, and each decision shows its reason: Q² over a limit (for PLS, 0 above 100 rows and 0.05 at 100 or fewer; rising with the component number for PCA); Q²V over the limit for √K columns or one Y; undecided components judged by the next eigenvalue; stops for too few degrees of freedom or too little explained; an eigenvalue limit of 2 for large tables.
  - A suggested number of components, stored with the model.
  - Already there: select_n_components applies its own rules (a minimum Q² increase, the one-standard-error rule). CrossValidation.recommended; the chart marks it.
- **MOD-8** Q² and Q²(cum) per component, and Q²V per variable. *(partly in process-improve)*
  - Q² = 1 − PRESS/SS per component, Q²(cum), and Q²V per variable.
  - Already there: CrossValidation gives Q² per component with its standard error; PLS.cross_validate gives RMSEP per Y. No Q²V per variable.
- **MOD-9** Cross-validation by element for PCA and by rows for PLS. *(in process-improve)*
  - PCA cross-validation element by element (Eastment-Krzanowski), so a component does not rotate into later ones; PLS cross-validation by rows.
  - Already there: PCA.select_n_components (cv_scheme ekf, ckf, ek, sacv, gcv); PLS.select_n_components, row-wise k-fold, repeated.
- **MOD-10** Validation choices: cross-validation groups, a test set, leverage correction, re-scaling per segment. *(partly in process-improve)*
  - Validation methods: cross-validation (7 rounds by default; groups by every n-th row, by a score, a column or an id; random or systematic; frozen rows), a test set chosen at random or by hand, leverage correction (fast, no refit), or no validation.
  - Whether each segment re-centres and re-scales its data is a stated choice.
  - Already there: select_n_components takes any scikit-learn splitter and repeats it; diagnose and predict score a test set. No leverage correction. PCA.select_n_components takes scale_inside_folds; compare_cv_criteria scales inside the folds and warns about scaling traps.

### STA: Statistics and diagnostics

- **STA-1** A model summary per component: R²X, R²Y, Q² and eigenvalues, with their cumulative values. *(partly in process-improve)*
  - A model summary per component, from component 0: R²X and R²X(cum) with the eigenvalue, R²Y and R²Y(cum), Q² with its limit and Q²(cum), significance and iterations; calibration and validation in %, clamped at 0.
  - The OPLS summary: the predictive component, a rotation line, each orthogonal component and a total.
  - Already there: eigenvalue_summary; R2 per component on every model; the page's model summary and its R² beside Q² chart. OPLS fits the predictive and orthogonal components.
- **STA-2** R² explained or adjusted, per variable too, and residual variance per row and per column. *(partly in process-improve)*
  - R² as an explained sum of squares or adjusted for degrees of freedom, also per variable (R²V, R²V(adj)), with the degrees of freedom stated for each use.
  - Residual variance per column and per row, for calibration and validation (SD and residual SD, original and scaled, degrees of freedom, N); total residual variance per cross-validation segment; explained variance per row.
  - Already there: r2_per_variable_ on PCA and PLS. No adjusted R². SPE per row (spe_) and R² per column (r2_per_variable_).
- **STA-3** Modelling power of each variable, and leverages of rows and variables. *(not in process-improve)*
  - Modelling power of each variable, 1 − SVₖ/SV₀ₖ (1 is fully relevant; near A/K is little relevance).
  - Leverages of rows in X and in Y (with 1/I for the centring), of variables, and for MLR.
- **STA-4** Hotelling's T², with its limits and ellipse. *(in process-improve)*
  - Hotelling's T² over any range of components, with its F-based limits (95% and 99%) and the T² ellipse; a zero significance hides the limits.
  - Already there: hotellings_t2_limit, ellipse_coordinates and t2_plot; the page draws the ellipse and the limit.
- **STA-5** Distance to the model in X and in Y, with its critical limit and probability. *(partly in process-improve)*
  - Distance to the model in X (DModX), absolute or normalized by the pooled residual SD, with its F-based critical distance and probability of belonging, optionally weighted by modelling power; the distance in Y (DModY); for predictions, the distance plus how far a row falls outside the model's score range.
  - Already there: SPE per row with spe_limit; the page's SPE plot. No normalized DModX or probability.
- **STA-6** Regression coefficients in every form: scaled, raw with intercept, MLR form. *(partly in process-improve)*
  - Regression coefficients in every form: scaled and centred; raw, with the intercept b₀ = ȳ − x̄'b; Y in original units with X centred; the multiple-regression form; squared and cross terms handled; B = W(P'W)⁻¹C'; rotated toward pure profiles.
  - Already there: beta_coefficients_ on PLS (scaled and centred); coefficient_plot.
- **STA-7** VIP. *(in process-improve)*
  - VIP of each variable.
  - Already there: vip(); the squared VIP values sum to K.
- **STA-8** Jackknife uncertainty and p-values for coefficients, loadings, VIP and predictions. *(partly in process-improve)*
  - Jackknife uncertainty from the cross-validation models: intervals for scores, loadings, coefficients, predictions and VIP, and a p-value for each coefficient, loading and weight, to select variables.
  - Already there: PLS.cross_validate gives jackknife error bars for the coefficients.
- **STA-9** Error measures: RMSEE, RMSEP, bias, SEP, and a standard error for each prediction. *(partly in process-improve)*
  - Error measures: RMSE of calibration (RMSEE) and of prediction (RMSEP, pooled over segments), bias, SEP, and the multiple correlation for calibration and validation.
  - A standard error for each predicted value.
  - Already there: PLS.cross_validate returns the cross-validated RMSEP per Y. PLS.prediction_interval.
- **STA-10** Residuals raw, standardized or studentized, with a normal probability plot. *(not in process-improve)*
  - Residuals raw, standardized or studentized.
  - A normal probability plot of the residuals, showing normality and outliers beyond about ±4 SD.
- **STA-11** Correlation loadings. *(in process-improve)*
  - Correlation loadings: the correlation of each raw X and Y column with the scores.
  - Already there: correlation_loadings_plot.
- **STA-12** Contributions to scores, T², distances and predictions: a row or a group, against the average or each other. *(partly in process-improve)*
  - Contributions to the scores and T², to the distance in X and in Y, and to predicted Y: a row against the average, a row against a row, a group against the average, a group against a group.
  - Contribution weights to choose from: for example p, w*, coefficients or VIP for the scores, and RX for the distances.
  - The gap contribution: the difference between two rows, optionally weighted.
  - Already there: score_contributions, t2_contributions, spe_contributions, group_contributions (a group against a reference) and oMEDA; the page explains a lassoed group. group_contributions with a one-row group and a one-row reference.
- **STA-13** Observation risk, and the leave-one-out prediction error of each row. *(not in process-improve)*
  - Observation risk (the change in the Y residuals with the row in or out, per Y and pooled), and the prediction error of each training row from the model fitted without it.
- **STA-15** Outlier detection against limits. *(in process-improve)*
  - Outlier detection against limits: critical limits drawn on the distance and T² plots, and the rows over them listed.
  - Already there: detect_outliers (SPE and T²), spe_limit and hotellings_t2_limit; the page draws the limits.

### PRD: Prediction and classification

- **PRD-1** A prediction set from a table, the unfitted rows, a class or pasted rows; saved as a table. *(partly in process-improve)*
  - A prediction set taken from a table, the rows not fitted, the working copy, a class, a mix of these, or rows typed or pasted in. It stays current, and centres can be imported.
  - Save the prediction set as a secondary table.
  - Already there: PCA.diagnose and PLS.diagnose project any new rows; the page projects its excluded rows.
- **PRD-2** Prediction gives scores, ŷ with its error, T², distances and residuals, missing values handled. *(partly in process-improve)*
  - Prediction projects each new row through the model, with missing values handled, and gives its scores and residuals as well as ŷ; a short form gives ŷ = b₀ + x'b alone.
  - Prediction results: scores, distance in X with its probability, T², predicted Y back-transformed with standard errors and jackknife intervals, residuals, distance in Y with its probability, hierarchical Y.
  - Already there: PCA.project and PLS.project estimate scores, predictions and diagnostics for rows with missing values; PLS.predict. diagnose returns scores, T², SPE and predictions; prediction_interval gives the intervals.
- **PRD-3** Prediction statistics when Y is known: RMSEP, bias, SEP, slope. *(partly in process-improve)*
  - Prediction with reference values: when Y is known for the prediction set, RMSEP, SEP, bias, slope, offset, correlation and the corrected SEP.
  - Already there: PLS.score gives R² on the new rows.
- **PRD-4** Cross-validated predictions for test rows, with their standard errors. *(not in process-improve)*
  - Cross-validated predictions for test rows from each round's submodel, with their jackknife SE.
- **PRD-6** Classification with one model per class, with membership probabilities. *(not in process-improve)*
  - Classification with one model per class: a table of members with the probability per class model (green above 10%, red from 5 to 10%, not a member below 5%), membership decided by a leverage limit and a distance limit, the modelling power of each variable within a class.
  - **PRD-6.1** Coomans' plot, distances between class models, discrimination power. *(not in process-improve)*
    - Coomans' plot: the distance to one class model against the distance to another.
    - The distance between two class models, and the discrimination power of each variable between two classes.

### VIS: Visualization

- **VIS-1** Plot types: scatter (2D and 3D), line, column, histogram, time series, normal probability, lists. *(partly in process-improve)*
  - Plot types: scatter (2D and 3D), line, column, histogram, time series, normal probability, contour and 3D response surface, and lists.
  - Any line plot shown as a histogram; any plot made from any list, and any list from any plot; the marked table rows drawn as lines, or every X row (spectra) when nothing is marked.
  - Already there: Plotly in the page (scatter, bar, line); the library's score_plot draws 3D.
- **VIS-2** A plot builder for every plot and list: the source, the series, scaling and units. *(not in process-improve)*
  - A plot builder for every plot and list: the data source, the data type, the item (variable, component, cross-validation round), an x series and any number of series, each scaled, offset or scaled 0 to 1, in original, transformed or working-copy units.
  - Transform a series: auto- or cross-correlation, power spectrum, wavelet coefficients, EWMA smoothing (which tames a dense score trajectory).
  - The confidence level of column plots (default, none, 99%, 95%), and the number format and precision of lists.
  - **VIS-2.1** Colour and size by value, category, class, density or limit. *(partly in process-improve)*
    - Colour a scatter: continuously between two colours (with colours for missing and out of range), by categories, by score categories, by density, by predictions, by models, by what-if, by ids, by patterns, or by class (the default when classes exist).
    - Size: enlarge the points whose distance is over the critical limit.
    - Already there: views.js colours by a number (a colour scale) or by a category (a colour-blind safe palette). views.js sizes markers by a number, area-proportional.
  - **VIS-2.2** Point and axis labels from ids, and item selection. *(partly in process-improve)*
    - Labels: the id (or part of it) for points and axis labels, axis-label rotation and interval, the ids in titles, saved as the default.
    - Item selection: move items between selected and unselected; find by text, class, X, Y, lag or cross term; invert; select all.
    - Already there: The page labels the selection and the extremes, each placed away from its neighbours.
- **VIS-3** Summary plots of the model. *(partly in process-improve)*
  - Summary plots: model overview, X/Y overview, X/Y component, component contribution, summary list.
  - Already there: The page's model summary and its R² beside Q² chart.
- **VIS-4** Score plots with the T² ellipse, in 2D and 3D. *(in process-improve)*
  - Score plots: t against t and u against u with the T² ellipse, 3D, line plots with ±2 and ±3 SD, t against u with a regression line, scaled by R²X.
  - Already there: score_plot (2D and 3D, with the ellipse); the page's score plot, % variance on each axis.
- **VIS-5** Loading plots, including correlation loadings and biplots. *(partly in process-improve)*
  - Loading plots: p, w, w*, c, w*c; p(corr), c(corr) and pc(corr) in the unit circle; scaled by R²X; biplot.
  - Already there: loading_plot and correlation_loadings_plot; the page's loading plot.
- **VIS-6** Diagnostic plots: T², distances, residuals, permutation, observation risk. *(partly in process-improve)*
  - Diagnostic plots: T² with 95% and 99% limits; residuals and their normal probability plot; the distances to the model in X and Y with the limit; observation risk; the permutation plot.
  - Already there: t2_plot and spe_plot with limits; the page's SPE and T² plots.
- **VIS-7** Coefficient and VIP plots. *(in process-improve)*
  - Coefficient plots (every Y at once, or one Y with jackknife bars, by coefficient type) and VIP plots (sorted with bars, or unsorted as columns).
  - Already there: coefficient_plot, with error bars from PLS.cross_validate; vip.
- **VIS-8** Observed against predicted. *(in process-improve)*
  - Observed against predicted, with RMSEE or RMSEP, and a time series of observed and predicted Y.
  - Already there: predictions_vs_observed_plot.
- **VIS-9** Contributions plotted from any plot. *(in process-improve)*
  - Contributions plotted from any of these plots.
  - Already there: The page's contribution bars for a lassoed group, with each top variable's raw data beside them.
- **VIS-10** Contour and response-surface plots, with the optimum found. *(partly in process-improve)*
  - Contour and response-surface plots: two X on the axes between low and high limits, the other X held at low, centre, high or a value (editable live), with resolution, levels, colours and labels; the maximum, minimum or saddle point found and reported.
  - Already there: The designed-experiments surface plots and their stationary point, for least-squares models.
- **VIS-11** Linked views: marking in one plot marks in all. *(in process-improve)*
  - A linked dashboard: four linked plots (a score scatter beside line or histogram plots of three variables) and a property bar for the model, the components and the colouring variable; marking in one marks in all; right-click changes a plot's type.
  - Already there: The selection store and link.js: a selection in one view shows in every view and in the table.
- **VIS-12** Annotations and fit statistics on plots. *(partly in process-improve)*
  - Annotations: labels (500 at most), free text, shaded alarm zones, regression lines, ellipses, limits and bars at a chosen confidence, a header and a time-stamped footer.
  - Statistics printed on any 2D scatter: the least-squares line with its slope and offset, bias, correlation, RMSED and SED.
  - Already there: views.js draws the T² ellipse, origin lines, control limits and arrows.
- **VIS-13** Styles: axes, fonts, colours in lists, saved defaults. *(partly in process-improve)*
  - Cell colours in lists: out-of-limit predictions red, classification probabilities green or red, misclassifications highlighted, significant components green.
  - Axes and styles: range, ticks, decimals, scientific or time labels, log or reversed axes, a second Y axis; grids, titles, fonts, line, symbol and column styles; restyle the marked points; legends and backgrounds; save a style as the default and restore the factory styles.
  - Already there: visualization themes (register_themes, set_theme).
- **VIS-14** Every result is named, and can be plotted or listed. *(partly in process-improve)*
  - Every result has a name, and can be plotted or listed: per row (scores, T², distances, leverages, predictions and their errors, residuals, classification, the cross-validation group), per column (averages, weights, loadings, coefficients, VIP, modelling power, R²V and Q²V, residual variances, error measures) and per number of components (eigenvalues, Q², R², sums of squares, the SD of t and u).
  - Already there: Fitted models expose named DataFrames: scores_, loadings_, spe_, hotellings_t2_, beta_coefficients_, r2_per_variable_ and more.

### UI: Interaction

- **UI-1** Mark points by lasso, box or band; find by text or rule; combine marks. *(partly in process-improve)*
  - Mark points with a lasso, a box, an x band or a y band; Ctrl keeps the marks; a click on empty space clears them; whole batches can be marked.
  - Find items by wildcards (? and *, case-sensitive) in ids or comments (begins with, contains, exact), or by a rule; the result replaces, joins or intersects the marks; select all or the complement.
  - Already there: link.js binds Plotly's lasso, box and click to the selection store; a double-click clears. Named sets combine by union, intersection and difference.
- **UI-2** A marking pane: include, exclude, set a class, relabel, list, plot raw values. *(partly in process-improve)*
  - A marking pane: include, exclude, set a class, relabel; the previous and the current groups.
  - Act on the marks: set a class, label or restyle them, list them, plot their raw values; plot from the table by marking rows or columns.
  - Already there: The page's Exclude, Include and Save set buttons act on the selection. Rows selected in the table are selected in every plot; a header click selects the variable.
- **UI-3** A contribution tool: double-click a point, or two points for their difference. *(partly in process-improve)*
  - The contribution tool: double-click a point for its contributions against the average; click one point and double-click another for the difference between them; a default action picks the contribution from what is marked; a flag shows the raw value behind a bar.
  - Already there: The page's Explain this group: contributions, with the raw data of the top variables.
- **UI-4** Excluding marked points creates a new model and keeps the old one. *(partly in process-improve)*
  - Exclude or include marked points from any plot: this creates a new model, fitted at once, and keeps the old one.
  - Already there: Exclude and Include refit the one model; the excluded rows are still projected and drawn.
- **UI-5** A right-click menu on every plot and list; sortable lists. *(not in process-improve)*
  - A right-click menu on every plot and list (copy, create a list or plot, show as histogram, add to favourites or to the report, quick info, sort, plot settings, maximize, save as the default style, save, print, properties); properties open the plot builder.
  - Lists warn before listing every item; lists and bar charts sort by value, or by part of an id then value.
- **UI-6** Zoom, hover and 3D rotation. *(in process-improve)*
  - Zoom (box, x, y, subplot) and zoom out, coordinates, 3D rotation, and hover giving the id, the series and the value.
  - Already there: Plotly zoom and hover; the hovered point is enlarged in every view.
- **UI-7** Keyboard: arrow keys change the components, Page Up and Page Down the model. *(not in process-improve)*
  - Arrow keys change the components; Page Up and Page Down change the model, or the phase.
- **UI-8** A property bar and a status bar with the model, A, R² and Q². *(partly in process-improve)*
  - A property bar (model, components, column) and a status bar (model, type, A, A for prediction, R²X, R²Y, Q², adaptive).
  - Already there: The page's model title and summary line.
- **UI-9** A model window, the working copy as tabs, and a simple wizard mode. *(not in process-improve)*
  - A model window with its summary and buttons to its options and working copy; the working copy as tabs (overview, variables, observations, transform, lag, expand, scale, spreadsheet), each with "as model", the selections kept across tabs.
  - A simple mode: a wizard that states the model it will build and fits it; the tabbed working copy is the advanced mode.
- **UI-10** Panes, toolbars and favourites. *(not in process-improve)*
  - An item-information pane.
  - Toolbars (standard, shortcuts, a customizable plot toolbar), tooltips and status-bar hints.
  - Dockable panes: audit trail, advisor, favourites, item information, marked items, observations, variables, what-if, quick info.
  - Favourites: commands and plots kept in folders; run a whole folder.

### SET: Settings

- **SET-1** Settings at four levels: factory, user, project, model. *(not in process-improve)*
  - Settings at four levels: factory, global (per user), project, and model, which overrides the project; a fitted model keeps its options until reset.
- **SET-2** Fit and prediction settings: cross-validation rounds, confidence level, tolerances, units. *(partly in process-improve)*
  - Fit settings: cross-validation rounds and segment selection, iteration cap, missing-value tolerance (50%), minimum distinct values, confidence level (95%), significance level (0.05), wavelet reconstruction, centring on the mean or none, and the warning limits.
  - Prediction settings: the units of predicted Y, whether the working copy's trimming applies, the adaptive λ (0.1).
  - Already there: The estimators take these as arguments (cv, n_repeats, max_iter, tol, conf_level).
- **SET-3** Display and start-up settings: result types, labels, preferences. *(not in process-improve)*
  - Result settings: distance normalized or absolute, weighted by modelling power or not; residuals raw or standardized; R² explained or adjusted; the coefficient type; the cross-validation groups; the components for results and for predictions.
  - Label settings: the default id and its part, the ids in titles, a cap of 500 labels.
  - General settings: open the last project on start-up, folders, tab position, simple or advanced mode, printing in colour, caching; each file remembers its display preferences; restore the factory plot styles, favourites, toolbars and "don't show again" messages.

### REP: Reporting, audit and help

- **REP-1** A report writer with templates and live placeholders, written as an HTML report, a runnable notebook or a Word document. *(not in process-improve)*
  - A report writer: templates (blank, a default per model type with the sections Objective, Data, Preprocessing, Model summary, Transformation and scaling, Goodness of fit, Results, Predictions and Conclusion, or the user's own); live placeholders (update, reset, freeze); plots and lists added from their menus; links, images and files; another model's report appended or inserted; output as HTML, e-mail or print, with a chosen image size and format.
- **REP-2** A log of every change. *(partly in process-improve)*
  - A log of every change (append, compute, delete, import, insert, replace, shift, sort, transform and each analysis), readable as text.
  - Already there: The recipe is the log of every step from the source file on, with undo, each step written as Python; the project file holds the source's SHA-256.
- **REP-3** Free-text notes per table and per model. *(not in process-improve)*
  - Free-text notes per table and per model result.
- **REP-4** A help system, an advisor that explains each plot, and a tutorial with example data. *(partly in process-improve)*
  - A help system with contents, an index and full-text search, and a Help button in every dialog.
  - An advisor that explains each analysis plot, and a tutorial with example data and example results.
  - A reading guide: what score and loading plots reveal (groups, trends, outliers, similarity), and how a direction in one is explained by the same direction in the other.
  - Already there: The page opens an example data set.
- **REP-6** A glossary, a reading list, and one notation everywhere. *(partly in process-improve)*
  - A glossary of terms (time series and control, data, variables, methods, geometry, diagnostics, and the kinds of process variable), with a reading list of the method papers behind each technique.
  - One notation everywhere (N rows, K X columns, M Y columns, A components; T scores, P loadings, W weights, Q Y loadings), stated in one table.
  - Already there: The package's convention: N, K, M, A, stored as n_samples_, n_features_in_, n_targets_.

### DEP: Model export

- **DEP-1** Export a model for use outside the app. *(partly in process-improve)*
  - Export a model for use outside the app, or compact a project for deployment.
  - Already there: The project file holds the recipe, the variants and the fitted arrays; the recipe and the model are written as runnable Python.
  - **DEP-1.1** As a plain-text model file, with a header and every matrix. *(not in process-improve)*
    - A plain-text model file with everything needed for prediction and classification.
    - A file header: type and format version, model name, date and creator, method, calibration data name, numbers of rows and columns, validation method, components present and suggested, centring, calibration and test rows, cross-validation segments, the list of matrices.
    - Each matrix as a header (name, rows, columns) with its values, missing written as "m"; the prediction-with-reference statistics, when Y was known.
  - **DEP-1.2** In a mini version (coefficients only) and a full one (with the outlier statistics). *(not in process-improve)*
    - A "mini" model file: the coefficients and intercepts only, for prediction alone.
    - A "full" model file that adds what the outlier statistics need: X and Y weights and centres, residual variances, loadings and loading weights, sums of squares, the mean calibration leverage, the RMSE of calibration, the SD of the scores, and the calibration means, SDs and values.
  - **DEP-1.3** With its preprocessing steps, so the deployed model repeats them. *(partly in process-improve)*
    - One line per preprocessing step, with its parameters, so the deployed model repeats the preprocessing; a step that needs stored data refers to a matrix in the file and is flagged when that matrix is missing.
    - Already there: The recipe's steps are written as Python with their settings.

## Only when nearly free

Built only when a Definitely line brings it within reach at almost no cost.

- **DAT-3** Import from a database, through a query wizard. *(not in process-improve)*
- **PRE-3.1** Lags, differences, bins and row-wise statistics in expressions. *(partly in process-improve)*
- **PRE-3.2** Columns taken from another table or from a model's results. *(not in process-improve)*
- **PRE-4** Spectral filters, row by row, chained and recorded so new data are filtered the same way. *(not in process-improve)*
- **PRE-4.1** Scatter correction: MSC and SNV. *(not in process-improve)*
- **PRE-4.2** Savitzky-Golay smoothing and derivatives, and row normalization. *(not in process-improve)*
- **PRE-4.3** Spectroscopic conversions (absorbance, Kubelka-Munk) and baseline correction. *(not in process-improve)*
- **PRE-4.4** Orthogonal signal correction (OSC). *(partly in process-improve)*
- **PRE-4.5** Wavelet denoising and compression, for spectra and for time series. *(not in process-improve)*
- **PRE-5** Per-model transforms (log, logit, power and others), with skewed columns flagged. *(partly in process-improve)*
- **PRE-6** Lagged columns. *(not in process-improve)*
- **MOD-3** Three-way data and three-way PLS. *(not in process-improve)*
- **STA-14** A response permutation test for over-fitting. *(partly in process-improve)*
- **STA-15.1** Automatic warnings with settable limits, collected in one outlier list. *(not in process-improve)*
- **PRD-5** Adaptive predictions, with EWMA-updated centring. *(partly in process-improve)*
- **PRD-8** What-if sliders on the inputs of a prediction. *(partly in process-improve)*
- **MON-1** Control charts of scores, variables, residuals or predictions. *(partly in process-improve)*
- **MON-2** Shewhart, CUSUM and EWMA charts, with limits from training data or entered. *(in process-improve)*
- **MON-3** Time-series PLS with lagged X and Y, and finite impulse response models. *(not in process-improve)*
- **MON-4** Auto- and cross-correlation. *(partly in process-improve)*
- **MON-5** Wavelet analysis, with its plots. *(not in process-improve)*
- **MON-6** Images analysed as rows, or as pixels. *(not in process-improve)*
- **REP-2.1** A regulated audit trail: user and time on every event, an administrator lock. *(not in process-improve)*
- **REP-5** The analysis cycle and its cautions, as the app's guide. *(not in process-improve)*

## Later

- **PRD-7** PLS-DA, with a misclassification table. *(in process-improve)*
- **BAT-1** Batch data: batches by time by variables, with phases and whole-batch columns. *(in process-improve)*
- **BAT-2** Batch import checks, and preparation per phase. *(partly in process-improve)*
- **BAT-3** Evolution models per phase, with maturity as Y, and a batch working copy. *(not in process-improve)*
- **BAT-4** Batch control charts with limits from good batches; new batches projected onto them. *(in process-improve)*
- **BAT-5** An out-of-control summary per batch. *(not in process-improve)*
- **BAT-6** Batch contributions. *(partly in process-improve)*
- **BAT-7** Batch-level models from initial conditions, phase summaries and final quality. *(partly in process-improve)*
- **BAT-8** Batch what-if, and export for on-line monitoring. *(partly in process-improve)*
- PLS-DA as a model type (taken out of MOD-1).
- Drawing the chemical structure when a row id is a SMILES string (taken out of UI-10).

## Out of scope

- Designed experiments (DOE-1, DOE-2, DOE-3, DOE-4): covered by the designed-experiments app.
- **DAT-2** Import Matlab files, designed-experiment files and other chemometrics packages' files.
- **DAT-8** Send chosen columns or results to an external visual-analytics tool.
- Instrument files and integration: spectroscopy and chromatography formats, instrument parameters, instrument model files (never supported).
- Export in exchange formats (DIF, Matlab 4, JCAMP-DX) (Not needed in the review).
- Plug-ins and user extensions (Not needed in the review).
- Numbered result files; replicate precision; statistics of one class only (Not needed in the review).

## Proposed build order

Each Definitely line in one milestone, in an order where each builds on the last.

1. **The analysis loop on the existing page** (33 lines, 32 with code already there). Most of it is in process-improve already; the work is the page and the model lineage.
   PRJ-1, PRJ-3, PRJ-4, PRJ-4.1, PRJ-4.2, PRJ-6, PRJ-6.1, MOD-1, MOD-5, MOD-6, MOD-7, MOD-8, MOD-9, MOD-10, STA-1, STA-4, STA-5, STA-7, STA-11, STA-15, VIS-3, VIS-4, VIS-5, VIS-6, VIS-7, VIS-8, VIS-9, VIS-11, UI-1, UI-2, UI-3, UI-4, UI-6.
2. **Data in, and preparing it** (15 lines, 12 with code already there). Import, the table, ids and classes, and the preprocessing the models need.
   PRJ-2, PRJ-5, PRJ-7, PRJ-8, PRJ-9, PRJ-10, DAT-1, DAT-4, DAT-5, DAT-6, PRE-1, PRE-2, PRE-3, PRE-7, PRE-8.
3. **Statistics, diagnostics and prediction** (16 lines, 10 with code already there). The full results of a model, new data through it, and class models.
   MOD-2, MOD-4, STA-2, STA-3, STA-6, STA-8, STA-9, STA-10, STA-12, STA-13, PRD-1, PRD-2, PRD-3, PRD-4, PRD-6, PRD-6.1.
4. **Plots and the working environment** (16 lines, 9 with code already there). The plot builder and styles, menus, keys and panes, and settings.
   VIS-1, VIS-2, VIS-2.1, VIS-2.2, VIS-10, VIS-12, VIS-13, VIS-14, UI-5, UI-7, UI-8, UI-9, UI-10, SET-1, SET-2, SET-3.
5. **Results out** (11 lines, 7 with code already there). Reports, the change log, help, and models exported for use elsewhere.
   DAT-7, DAT-9, REP-1, REP-2, REP-3, REP-4, REP-6, DEP-1, DEP-1.1, DEP-1.2, DEP-1.3.
