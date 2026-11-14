# Factorial: what the app does

Factorial is the browser app of `process-improve`, at <https://factori.al/>. It plans a
designed experiment, takes its results and analyses them, in three steps: Design, Run and
Learn. Python runs inside the browser tab, so a study never reaches a server.

This document lists what the app does today, grouped by kind. Use it to describe the app
elsewhere. Three rules keep it true:

- It holds what the app does as built from this commit, and nothing planned. What the app
  does not do is under [Not in the app](#13-not-in-the-app).
- A change to the app that a reader of the app would notice changes this document in the same
  pull request ([`CONTRIBUTING.md`](../CONTRIBUTING.md#the-browser-app-is-tested)).
- [`tests/web/page/test_features.py`](../tests/web/page/test_features.py) fails when a design,
  a chart, a process to practise on, a limit or a count below is not what the app has.

Counts marked "(Nov 2026)" come from the build of 14 November 2026 and only grow.

1. [In one minute](#1-in-one-minute)
2. [Designs](#2-designs)
3. [Planning a study: Design](#3-planning-a-study-design)
4. [Running it: Run](#4-running-it-run)
5. [Calculations: Learn](#5-calculations-learn)
6. [Figures](#6-figures)
7. [Interaction and ease of use](#7-interaction-and-ease-of-use)
8. [Data in and out, and reproducibility](#8-data-in-and-out-and-reproducibility)
9. [Enterprise: privacy, security and operation](#9-enterprise-privacy-security-and-operation)
10. [Learning and documentation](#10-learning-and-documentation)
11. [The other pages at factori.al](#11-the-other-pages-at-factorial)
12. [Limits](#12-limits)
13. [Not in the app](#13-not-in-the-app)

---

## 1. In one minute

- **No install, no account, no cost.** A web page. It is open source under the MIT license.
- **Your data stays on your device.** Python (CPython in WebAssembly) runs in the tab. The
  server sends the app's files and nothing more.
- **Works with no network.** After one visit the app opens and runs offline. One more
  download adds the charts and the workbooks.
- **16 designs** from one form, from a full factorial to optimal, space-filling and mixture
  designs, with limits on the settings that can be run.
- **From a brief to a recommendation.** The app lists the designs that can answer the
  question, analyses the results, finds the best settings, and says what to run next.
- **Plain words.** Findings are written as sentences from the numbers alone, with no language
  model: the same results give the same words.
- **17 charts.** Each opens in a window to resize, save or copy, and 11 of them show the
  Python that draws them. A run pointed at on one chart is marked on the others and in the
  run sheet.
- **Reproducible.** Every step shows Python code that runs as it is. A study leaves as a
  workbook, a notebook, a report, a file or a link.
- **Checked against print.** The design catalog opens more than 1,000 published designs and
  experiments in the app. Where a source prints its results, the app lists them beside its
  own.

## 2. Designs

The design list holds 16 designs. The reading level sets how many of them show.

| Design | Factors | Notes |
|---|---|---|
| Full factorial (2-level) | 1 to 7 | Mixed levels with a categorical factor of 3 to 8 levels |
| Fractional factorial (2-level) | 3 to 24 | Resolution III, IV or V; minimum aberration; chosen on a grid of runs against factors |
| Plackett-Burman screening | 2 to 40 | |
| Taguchi orthogonal array | 2 to 40 | Each numeric factor at two levels or at three |
| Definitive screening (DSD) | 3 to 40 | |
| OMARS (three levels, runs you choose) | 3 to 5 | Up to 41 runs |
| Box-Behnken | 3 to 7 | |
| Central composite (CCD) | 2 to 7 | Face-centred, rotatable, orthogonal, or an axial distance you type |
| D-optimal | 2 to 8 | Made for a model, inside limits on the settings |
| I-optimal | 2 to 6 | Made for a model, inside limits on the settings |
| A-optimal | 2 to 6 | Made for a model, inside limits on the settings |
| Space-filling (Latin hypercube) | 2 to 8 | Maximin; up to 200 runs |
| Space-filling (Sobol sequence) | 2 to 8 | Up to 200 runs |
| Simplex lattice (mixture) | 3 to 6 | Degree 1 or 2 |
| Simplex centroid (mixture) | 3 or 4 | |
| Extreme vertices (mixture with bounds) | 3 to 6 | A lower and an upper proportion for each component |

Also:

- **Runs as given.** The runs of a design the app did not make, in their order and their own
  units: a table of your own, or a design from the catalog. The app analyses them with the
  largest model they estimate in full. Runs that form a regular two-level fraction state its
  resolution and defining relation.
- **A study of one factor.** Two or more levels compared, sized by the power of the test.
- **Options.** Center points, replicates (up to 10), blocks (up to 8), the model an optimal
  design is made for, and the number of runs where the design takes one.
- **Random run order** from a seed of the study's own, named in the report and the code.
- **What the design is.** Its runs and spare runs, its resolution and defining relation, its
  aliases, and what its blocks take.

## 3. Planning a study: Design

- **The brief.** What you measure (name, units, and a goal: maximise, minimise, hit a target,
  or just learn), what might change it, and what you want to learn.
- **What you know already (optional).** The typical value, the run-to-run noise and the
  smallest change worth finding. They give each design's power to find that change.
- **Up to 5 responses**, each with its own units and goal.
- **Numeric and categorical factors.** A categorical factor has 2 to 8 levels of any text,
  shown as chips to type, paste, remove and reorder.
- **Four aims.** Which factors matter; how they act together; where each effect curves; the
  whole response surface.
- **The designs that can learn the brief.** As the brief is typed, the app makes every design
  that fits and lists each with its runs, spare runs and power, and what it stands out for.
  It picks the fewest runs that leave three or more spare, and says why. A design left out
  says why.
- **Compare them in detail.** Nine measures of each design (precision of the coefficients and
  of the predictions, correlation between terms, power), fraction-of-design-space curves on
  one chart, and a correlation map of each design. A switch sets a design aside.
- **Limits on the settings.** Up to 8 inequalities in the factors' own units, such as
  `T + 25*P <= 250`. The app says what each rules out and what share of the ranges is left.
  A limit can also be proposed from the corners you mark as not runnable. A picture shows
  the settings that can be run for each pair of factors.
- **Which terms these runs tell apart.** A correlation map of the study's own runs, with
  sentences that name each aliased or correlated pair of terms.
- **Experiments like yours.** The plan beside what others in your field have published.
- **Up to 40 factors.** Interactions are modelled for up to 12.
- **Renaming.** A new name or unit for a factor or response reaches the run sheet, the
  equations, the charts and the code, without making the runs again.

## 4. Running it: Run

- **The run sheet** in its randomised order, with a column for each response. Its columns sort.
- **Results typed, pasted or uploaded.** A block pasted from a spreadsheet fills the sheet,
  matched by run number when it has one, with decimal commas read. A filled-in workbook
  (.xlsx) is read back.
- **Partial results.** The app analyses the runs that have a result.
- **Practise with simulated results.** Six processes with a known true model: Reactor yield,
  Main effects only, An interaction, A maximum inside the ranges, Nothing active, and A
  random process. A mixture has one of its own (a synergy and an antagonism). You set the
  noise, simulate again for new noise, and show the true process when you want it.
- **A curve on each run.** A response that is a whole curve, such as a spectrum or a colour
  over time, pasted under the run sheet.
- **Added runs** join the run sheet as a block of their own, and confirmation runs go at its
  end.
- **A changed plan** marks the run sheet as out of date, with one button to make the new one
  and one to put the design back.

## 5. Calculations: Learn

**The fit**

- **What the results say.** Sentences on which factor matters most and by how much in its own
  units, which effects are real, which depend on another factor, where the response curves,
  how well the model fits, and which runs to look at again.
- **The model.** As designed, main effects, main effects and two-factor interactions, the full
  quadratic, or terms you choose (interactions of up to three factors, and squares).
- **Coefficients** with 95% intervals and p-values, ordered by p-value. With simulated results
  each stands beside its true value.
- **Analysis of variance**, a lack-of-fit test when runs are repeated, R², adjusted R²,
  predicted R² and the residual degrees of freedom.
- **The response as measured, or its log, square root or inverse.** Box-Cox's interval offers
  a transform when the results call for one. Results are put back in the response's units.
- **Aliased terms** are estimated once, as their chain (`T + P·F`).
- **Saturated designs** are judged with Lenth's method.
- **Blocks** are fitted as a fixed effect. A term the blocks take is left out, with a note.
- **Categorical factors** of three levels or more are fitted as a contrast for each level and
  tested as one source. A factor nested in another is fitted within it.
- **Correlated terms.** Each coefficient's variance inflation factor, with a note when one is
  above 5.

**Checking the fit**

- **Each run, as the model sees it.** Fitted value, residual, leverage and Cook's distance. A
  run is marked by the Bonferroni outlier test.
- **Set a run aside.** The model is fitted without it. The run keeps its result, and Learn,
  the report and the code all say it was left out.
- **Simplify the model.** Stepwise selection by AICc that keeps an interaction only with both
  its main effects. Nothing is fitted until you accept, and the app then says the terms were
  picked from the same results.
- **The spread of repeated results.** Mean, standard deviation, ln s and the signal-to-noise
  ratio at each setting, a response table, and the adjustment factor for a target.

**A study of one factor**

- **Two levels.** The difference and its interval, tested four ways: the t-test, Welch's
  t-test, a randomization test and Wilcoxon's rank-sum test.
- **Paired runs.** The paired t-test, a randomization test and Wilcoxon's signed-rank test.
- **Three levels or more.** One-way analysis of variance, Tukey's and Dunnett's intervals,
  Bartlett's test and the Kruskal-Wallis test.

**Mixtures**

- **Scheffé models**: linear, quadratic or special cubic blending, with the mixture's analysis
  of variance and each component's effect along its Cox direction.
- **The best blend**, searched over the blends the bounds allow.

**From the model to a decision**

- **The best settings for your goal**, inside the ranges and the limits, with the prediction,
  its 95% confidence and prediction intervals, a warning when it leans on the model beyond
  the runs, and how tightly to hold each factor.
- **Balance the responses.** With two or more goals, the settings of the best overall
  desirability, with each response's acceptable range, importance and shape to edit, and a
  picture of where every response is acceptable.
- **What if.** A slider for each factor. Each response's prediction, its intervals and its
  leverage follow at once. Settings can be kept with the study under a name.
- **A target curve.** For a curve response, the settings whose predicted curve comes closest.
- **Confirmation runs.** One button adds three at the settings found, and the app judges them
  when their results are in.
- **The next step.** What the results call for, with one button: runs added to the design
  (a fold-over, axial runs, center runs, runs chosen for the model, or every run again),
  confirmation runs, or a new study with its ranges moved along the steepest ascent.

**Data you have already**

- **Analyse data you have.** A table pasted, or a .csv or .xlsx file, of a design run
  elsewhere or of a process's records. The app guesses the factors and the responses, reads
  units from the headings, and lets you correct it. It then says which factors moved
  together and what unplanned runs cannot show.
- **Printed in the source.** A study opened from the catalog lists each number its source
  prints beside the app's own, and whether they agree.

## 6. Figures

**The 17 charts**

| Chart | Shown for |
|---|---|
| Coefficients | Every fit; switches to the book's Pareto plot |
| The runs at each level | A study of one factor |
| Fitted response surface | Two factors of your choice, the others held at the best settings |
| True response surface | Simulated results, on the fitted surface's colour scale |
| Fitted response over the blends | A mixture of three components |
| True response over the blends | A mixture, simulated |
| Fitted response along each Cox direction | A mixture of four components or more |
| Residuals against fitted values | Every fit |
| Half-normal plot of the effects | A saturated two-level design |
| Each factor's effect on the prediction | More charts; with a 95% band on each line |
| Average result at each setting | More charts |
| Interaction of the two factors | More charts |
| Fitted response surface, in three dimensions | More charts |
| Predictions at the corners | More charts |
| Residuals against the normal distribution | More charts |
| Residuals in the order the runs were made | More charts |
| Where the predictions are least certain | More charts |

Other pictures: the settings you can run, for each pair of factors; the fraction-of-design-space
curves and correlation maps of the designs compared; the correlation map of the study's
runs; how tightly to hold each factor; where every response is acceptable; and a predicted
curve with its 95% bands.

**Looking closely**

- **A window for each figure.** Expand opens a figure over the page. It takes a shape (16:9,
  4:3, square or all the room) or the size its corner is dragged to, with a mouse, a finger or
  the arrow keys, and keeps that shape for the next visit. Plotly's tools zoom and turn it.
- **Step through the figures** with the window's arrows or the left and right keys.
- **Save or copy.** A PNG or an SVG in the shape shown, on white under its title, or an image
  on the clipboard to paste into a slide. An image saved from a dark page has the light
  theme's colours.
- **The Python that draws it.** The fitted response surface, the residuals, the half-normal
  plot and the eight under More charts (11 of the 17) show code that runs as it is. The
  others are drawn by the page, and the analysis's code gives their numbers.
- **One run, marked everywhere.** Rest on a run on a chart or in a table and it is ringed on
  every chart that draws the runs and shaded in the run sheet. A press or a tap keeps it;
  Escape lets it go.
- **Reading a chart.** Resting on a point names its run. A chart of several lines reads every
  line at one setting together. Each factor's effect is read with its interval.

**Drawn to be read**

- Axes and titles are in your words and units. A long title wraps.
- Every chart fits its card and is drawn again when its room changes.
- Light and dark themes. On a dark page a surface starts as dark as the page, with no glare.
- The numbers on a surface's lines have a halo, so they read on every band.
- Lines differ by pattern as well as colour. A run's marker reads on every colour of a surface.
- Each chart is one named picture to a screen reader, beside a table that holds its numbers.
- No chart offers to send itself to Plotly's servers.

## 7. Interaction and ease of use

- **Three steps, one page.** A bar names the study and Design, Run and Learn, each marked with
  how far it has got.
- **Three ways in.** Start a new study, explore a worked example, or browse the design
  catalog. A fourth button analyses data you have.
- **A reading level.** Beginner, Intermediate or Adventurous. Each level adds designs, charts
  and options, and says how many more a higher level has. A level leaves out no result and no
  warning.
- **Words defined on request.** The glossary holds 78 terms. A term on the page has a dotted
  underline and opens its definition, in the book's words, on a hover, a press or the
  keyboard.
- **A phone, a tablet or a desk.** The layout, the tables and the charts fit each.
- **Light, dark or the system's colours**, kept in the browser.
- **The keyboard and a screen reader.** A link past the header, sortable headings, labelled
  controls, and text that meets WCAG AA contrast in both themes.
- **Tables that sort**, and a coefficients table that downloads as .csv.
- **A page that stays free.** Python runs in a worker, so the page scrolls and types while it
  computes. A long search can be stopped.
- **Errors in one sentence**, with what to change. A name the analysis would refuse is refused
  as it is typed, with a name that works.
- **Every study is kept** in the browser as you work, with a draft of a brief not yet made.
  Your studies lists them, to open, share or delete.
- **Feedback** from the header, the footer or beside a result, with a list of what is sent.

## 8. Data in and out, and reproducibility

- **Show the Python code.** Every step has a script of `process-improve` calls that runs as it
  is and gives the page's numbers: the design, the comparison, the analysis, the best
  settings, the balance, the predictions and each figure. Tests run the code and compare.
- **Workbook (.xlsx).** The run sheet to fill in and read back. Two downloads of one study are
  the same file.
- **Run sheet (.csv)**, for any spreadsheet or program.
- **Notebook (.ipynb)** of the study's code.
- **Report (.html).** One page that needs nothing else: the brief, the design, the results, the
  analysis with its charts, the best settings, the next step and the code. A stamp gives the
  versions, the seed, the SHA-256 of the study's inputs and a link that opens the same study.
- **A link to a study**, with its results or without. The study is packed after the `#`, which
  a browser sends to no server.
- **The study file (.json).** One study or all of them, to move between browsers. It is plain
  JSON and carries a SHA-256 of each study.
- **Old studies keep opening.** A link or a file saved by an earlier version opens in a later
  one. An earlier version refuses a newer study in a sentence, and opens no part of it.
- **Share** hands a study to the device's share sheet where the browser has one.
- **Links by name.** The book's 19 examples open with `?book=<name>`, and 3 published
  examples with `?example=<name>`.

## 9. Enterprise: privacy, security and operation

What an organisation asks before it lets people use a tool.

**Where the data goes**

- **Nowhere.** Factors, results and analyses stay in the browser. The server has no database,
  no accounts, no cookies and no tracking script.
- **What a server sees.** One access-log line per request, with the address cut to its network
  before it is written, kept for 366 days to count visits.
- **Feedback is the one thing sent**, only on Send, and only what the dialog lists. The relay
  emails it and keeps nothing.

**Running it yourself**

- **A folder of plain files.** The app is built with one command and served by any web server,
  an internal one included.
- **Offline.** A copy of the app, Python and its packages is kept in the browser (about 170
  MB), and one button adds the charts and the workbooks (about 12 MB more). The footer says
  what the copy holds, and removes it on request.
- **Pinned versions.** The Python release and each package are pinned, and the About page
  shows the versions a copy loads.

**Safe by construction**

- A workbook's formulas are never evaluated, and a model is never read from a file as text.
- A link is checked field by field before any of it is used.
- Text a spreadsheet would run as a formula is written to .csv so that it stays text.
- A table of your own is held to 2 MB, 60 columns and 2,000 runs.
- The site is served over HTTPS with strict transport security, and refuses to be framed.

**Evidence it is right**

- **About 3,500 automated tests of the app** (Nov 2026): the page's logic, every call into
  Python, the page in a real browser, and the app under real WebAssembly.
- **Numbers checked against print.** The app's results for the book's 19 examples are held to
  the numbers the book prints. The reproduction page reports how closely the library gives
  the numbers 22 sources print.
- **Only a tested build is published.** A build is sent to factori.al after it passes its
  smoke tests, and is tested there again, and every week.
- **Deploys can be undone** with one action, to the build before or to a named commit.
- **The code is scanned** (CodeQL) on every pull request and every week.
- **A usage report for the operator.** Visits to each page by day, week or month, from the
  access log, with nothing of a reader's in it.

## 10. Learning and documentation

- **Links into the book.** Each design, each measure and each defined word links to its place
  in *Process Improvement using Data* (<https://learnche.org/pid>).
- **The worked example** opens at once, and can be run again in the tab.
- **Practice on a known process** shows the true model beside the fitted one.
- **How Factorial works** (the About page): what runs where, what reaches a server, the
  versions loaded, and how to build and serve the app yourself.
- **The concept map**: see below.
- **The package's documentation**: <https://kgdunn.github.io/process-improve/>.

## 11. The other pages at factori.al

- **The design catalog** (`/catalog/`). More than 1,500 entries (Nov 2026): published
  experiments and designs from 22 sources, 1956 to 2026, and designs the library makes.
  Search in words, filter by family, field, runs, factors and model, and see each on a map of
  runs against factors. A design's page links to where its source prints it, projects its
  runs onto two or three factors, and draws its correlation map. Up to four designs are
  compared side by side. More than 1,100 designs open in Factorial with one click. There is
  a page for each of 30 fields and for each source, a list of errata, a feed, and forms to
  propose a published design or share a case study.
- **Reproduced from the books** (`/catalog/reproduced/`). How closely the library reproduces
  each number its sources print, in four tiers, with the reason for each difference.
- **The concept map** (`/map/`). 237 ideas of designed experiments and 489 links between them,
  in 13 topics, each with its definition, the book's line and a link into the book. It shares
  the app's reading level.
- **The multivariate page** (`/mva/`). PCA and PLS on an example or your own file: a data
  table with each variable's distribution in its heading, linked score, loading, SPE and T²
  plots, and "Why are these rows different?". The app does not link to it yet
  ([#1227](https://github.com/kgdunn/process-improve/issues/1227)).

## 12. Limits

| What | Most |
|---|---|
| Factors in a study | 40 |
| Factors whose interactions are modelled | 12 |
| Responses in a study | 5 |
| Levels of a categorical factor | 8 |
| Limits on the settings | 8 |
| Blocks in a design the app makes | 8 |
| Blocks in runs as given | 32 |
| Replicates of each run | 10 |
| Runs in a table of your own | 2,000 |
| Columns in a table of your own | 60 |
| Runs of a space-filling design | 200 |
| Runs of an OMARS design | 41 |
| Runs at each level of a one-factor study | 50 |
| Points along a curve | 1,000 |
| Components of a mixture | 6 |

A first visit downloads Python and its packages. In Chromium the app was ready in about 11
seconds on a first visit and about 8 seconds after (Nov 2026).

## 13. Not in the app

Do not describe the app as having these.

- Accounts, sign-in, shared workspaces, or a record of who changed what. There is no server
  to hold them.
- Split-plot designs for hard-to-change factors, and supersaturated designs
  ([#1225](https://github.com/kgdunn/process-improve/issues/1225)).
- An assistant that turns a typed description into a brief
  ([#1224](https://github.com/kgdunn/process-improve/issues/1224)).
- A guided plan for a whole campaign of studies
  ([#1226](https://github.com/kgdunn/process-improve/issues/1226)).
- A language other than English.
- Browsers other than Chromium in its automated tests.
