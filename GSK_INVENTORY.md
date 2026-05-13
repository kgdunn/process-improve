# GSK Teaching Material — Phase A Inventory

> **Status:** Phase A research artefact. Drives Phase B (rebuild as Python self-study case studies inside `process-improve`, with a few items possibly routed to Chapter 7 of `pid-book`).
> **Date of audit:** 2026-05-13.

## 1. Context

Two GSK-related repositories on GitHub hold the original teaching material from Kevin Dunn's 2011–2012 Latent-Variable Methods course at GSK Zebulon. They are static archives — both have been untouched for over a decade. Before any of this content is rebuilt as Python case studies in `process-improve`, we needed to settle which repo holds the canonical version of each piece, what is unique to each, and what is missing entirely.

| Repo | Role | Format | Last commit on default branch |
|------|------|--------|---|
| [`kgdunn/gsk-teaching`](https://github.com/kgdunn/GSK-teaching) | MATLAB LVM/PLS/PCA toolkit + FMC worked example + datasets + 5 reading PDFs | MATLAB `.m`, `.mat`, `.xls`, `.pdf` | `87cfb3b` — 2011-05-16 |
| [`kgdunn/gsk-course-notes`](https://github.com/kgdunn/GSK-course-notes) | Lecture notes, case-study write-ups, exercises | LaTeX beamer (`.tex`) | `303921c` — 2012-05-21 |

**Headline finding.** The two repos are **complementary**, not duplicate. `gsk-course-notes` is the more recent (May 2012 vs May 2011) and is the canonical source for every narrative — case studies, exercises, topical modules. `gsk-teaching` is the only place that holds (a) the datasets in MATLAB binary form, and (b) runnable analysis code (FMC only).

## 2. Canonical-source matrix

| Item | Canonical narrative | Runnable code | Dataset | Phase B routing flag |
|------|---------------------|---------------|---------|----------------------|
| **FMC** batch case | `gsk-course-notes/examples/batch-fmc.tex` | `gsk-teaching/example_FMC.m` | `gsk-teaching/datasets/FMC.mat` (1.70 MB) | `process-improve` — multiblock batch PLS |
| **DuPont Nylon** batch case | `gsk-course-notes/examples/batch-dupont.tex` | — | `gsk-teaching/datasets/DuPont.mat` (440 KB) | `process-improve` — batch PCA outlier diagnosis |
| **SBR rubber** batch case | `gsk-course-notes/examples/batch-sbr.tex` | — | `gsk-teaching/tests/SBRDATA.mat` (766 KB) | `process-improve` — batch PLS fault diagnosis |
| **Kamyr digester** exercise | `gsk-course-notes/exercises/kamyr-digester.tex` | — | `gsk-teaching/tests/kamyr-digester-subset.xls` (51 KB) | `process-improve` — soft-sensor / time-lag PLS |
| **Food texture** exercise | `gsk-course-notes/exercises/food-texture.tex` | — | inline + connectmv.com link | **Candidate for `pid-book` Ch 7** — small intro PCA |
| **Wafer thickness** exercise | `gsk-course-notes/exercises/wafer-thickness.tex` | — | connectmv.com link | **Candidate for `pid-book` Ch 7** — PCA outliers, SPE |
| LVM framework code (`mblvm.m`, `mbpls.m`, `mbpca.m`, block classes, `lvmplot.m`) | — | `gsk-teaching/*.m` | — | Reference only; `process-improve` already has its own Python LVM library |
| 11 topical modules (PCA, NIPALS, MLR/PCR/PLS, monitoring, soft-sensors, classification, preprocessing, multiblock, etc.) | `gsk-course-notes/gsk/*.tex` | — | — | Pedagogical scaffolding; see §6 |
| 13 class files + 38 slide files | `gsk-course-notes/classes/`, `gsk-course-notes/slides/` | — | — | Presentation layer; not carried forward |
| 5 reading PDFs (multiblock-method papers) | — | — | `gsk-teaching/readings/` | Reference list only |

## 3. Case-study summaries

### 3.1 FMC — multiblock batch PLS

- **Source:** [`examples/batch-fmc.tex`](https://github.com/kgdunn/GSK-course-notes/blob/main/examples/batch-fmc.tex) (course-notes) + [`example_FMC.m`](https://github.com/kgdunn/GSK-teaching/blob/main/example_FMC.m) (teaching) + [`datasets/FMC.mat`](https://github.com/kgdunn/GSK-teaching/blob/main/datasets/FMC.mat).
- **Process / business context.** Agricultural-chemical drying batch process. Wet "cake" (solid + embedded solvent) is charged, dried in three recipe phases (solvent collection, temperature ramp, cool-down). Solvent is collected in an external side tank; chemical changes occur in the solid phase during drying. Operators can adjust a few set-points.
- **Dataset shape.**
  - 59 batches initially; **13 excluded** for missing initial-chemistry data → **46 used**.
  - Batch disposition labels: Good 1–33, Abnormal 34–61, High residual solvent 62–71.
  - Block structure (`example_FMC.m`):
    - `Zchem = X⁽¹⁾`: initial-condition chemistry block (cake properties)
    - `Zop  = X⁽²⁾`: operations / alignment-warping block
    - `X    = X⁽³⁾`: 10 batch trajectories, aligned within each of 3 phases
    - `Y`: critical quality attributes (CQAs)
- **Fault / analytical question.** Multivariate characterisation of product quality; effect of initial conditions on quality; trajectory alignment within phases; troubleshoot poor-quality batches; final-quality prediction; stagewise batch monitoring.
- **Learning objectives.** Multiblock data layout (`Z`-blocks for raw materials, recipe, operator/shift, ambient conditions, idle times). Phase-wise trajectory alignment with a time-warping trajectory carried as a variable. Working up from PCA-on-Y → PLS-chemistry → PLS-operating → multiblock PLS → batch MB-PLS with trajectories.
- **Methods used.** PCA, PLS, multiblock PLS, batch PLS, contribution plots, SPE.
- **Recommended Phase B action.** **Translate `example_FMC.m` to Python**, reusing `process-improve`'s existing MB-PLS implementation. Convert `FMC.mat` → CSV/parquet. Reuse the LaTeX narrative essentially verbatim — it is the most complete case study of the three and the only one with a runnable reference implementation.

### 3.2 DuPont Nylon — batch PCA outlier diagnosis

- **Source:** [`examples/batch-dupont.tex`](https://github.com/kgdunn/GSK-course-notes/blob/main/examples/batch-dupont.tex). Dataset: [`datasets/DuPont.mat`](https://github.com/kgdunn/GSK-teaching/blob/main/datasets/DuPont.mat). No MATLAB script.
- **Process / business context.** Industrial Nylon production at DuPont. The motivating constraint: lab analysis of final quality takes **12 hours**, so feedback-based adjustment is impossible; long hold-ups before disposition is known. (Cites Nomikos PhD thesis as the original source.)
- **Dataset shape.** `N = 55` batches × `K = 10` tags (temperature, pressure, flow) × `J = 100` time intervals. Data are scaled for confidentiality.
- **Fault / analytical question.** Known problematic batches: **38, 40, 41, 42, 50, 51, 53, 54, 55**. Build a batch PCA model that flags these via scores and SPE.
- **Storyline (walk-through).**
  1. Initial 2-component PCA, R²_X cumulative ≈ 55.9 %. Batches 50–55 distort the model.
  2. SPE flags batch **49** — but raw data alone wrongly points to `Flow-1`; the SPE contribution plot ($JK = 1000$ contributions, grouped by tag) reveals the true cause is small deviations in heating/cooling and pressure systems at $t = 55$–67. Nomikos reported batch 49 had barely acceptable final quality.
  3. Score outliers (batches 50–55) investigated via $p_1$ loading and contribution plots. Batch 54 has high $t_1$.
  4. **Exclude 49–55, rebuild.** A second cluster appears in $t_2$–$t_3$. Investigate batch 39 — turns out 37, 39, 43–48 are **not bad batches**, just operated differently.
  5. **Exclude 37, 39, 43–48 too** and rebuild a third model — score distribution is now more even.
  6. **Key teaching point — observability.** Batches 38, 40, 41, 42 are known to have poor CQA, but no cause is visible in the trajectories. The measurements don't contain the necessary information. *"The measurements must contain the information required to make a classification."*
- **Learning objectives.** Batch PCA workflow, iterative outlier-and-rebuild loop, distinguishing "bad" from "different but acceptable", SPE-vs-score contributions, observability requirement.
- **Methods used.** Batch PCA, SPE, score / loading contributions over $J \times K$.
- **Recommended Phase B action.** Narrative is ready to reuse from `.tex`. **Write Python implementation from scratch** using `DuPont.mat` (converted to CSV/parquet). Heavy use of contribution plots — make sure the `process-improve` plotting story supports the $JK = 1000$ bar plots and tag-grouped summaries used in the slides.

### 3.3 SBR rubber — batch PLS fault diagnosis on simulated data

- **Source:** [`examples/batch-sbr.tex`](https://github.com/kgdunn/GSK-course-notes/blob/main/examples/batch-sbr.tex). Dataset: [`tests/SBRDATA.mat`](https://github.com/kgdunn/GSK-teaching/blob/main/tests/SBRDATA.mat). No MATLAB driver script (but the `.tex` contains MATLAB snippets in a comment block showing the analysis recipe).
- **Process / business context.** Styrene-butadiene rubber (SBR) batch reactor. Data are **simulated** from a first-principles mechanistic model — useful because the simulation contains a deliberately injected fault that we know the model should find. (Also from Nomikos PhD thesis.)
- **Dataset shape.** `N = 53` batches × `K = 6` tags × `J = 200` time steps. Tags: reactor temperature, cooling-water temperature, reactor-jacket temperature, latex density, conversion, energy released. `Y`-space has 5 quality variables: composition, particle size, branching, cross-linking, polydispersity.
- **Fault / analytical question.** Two batches (34 and 37) had the **same injected fault** — 30 % greater organic impurity in the butadiene feed — but starting at **different times**.
- **Storyline.**
  1. Build a 2-component PLS. $R^2_{X,1}=24.5\%$, $R^2_{X,2}=12.7\%$; $R^2_{Y,1}=65.3\%$, $R^2_{Y,2}=6.9\%$.
  2. Score plot flags batches **34** and **37** as the unsuccessful ones — encouraging.
  3. SPE on the entire-batch data fails to flag the fault (it is averaged out).
  4. Weights $w_1, w_2$ reveal: batch 37 has low $t_1$ because of below-average latex density and conversion **throughout the batch**.
  5. Raw-data and contribution plots for batch 37 confirm the interpretation — fault from the very start.
  6. Batch 34 has high $t_2$ from cooling-water, jacket-temperature, and energy released — the **same fault starting mid-batch**.
  7. Score plot demonstrates the key insight: identical faults appear in different score-plot locations when they occur at different times.
- **Learning objectives.** Batch PLS for fault detection; same fault → different signature depending on onset time; using $w$, contributions, and raw-data overlays in sequence; cross-checking quality-prediction obs-vs-pred plots to confirm batches 34/37 also have poor predicted quality.
- **Methods used.** Batch PLS, score plots, SPE, weights $w$, contribution plots, observed-vs-predicted $Y$.
- **Recommended Phase B action.** Narrative reusable from `.tex`. **Write Python from scratch** using `SBRDATA.mat`. The MATLAB snippets embedded in the `.tex` are a usable algorithmic spec — variable names like `tagNames`, dimensions (53, 200, 6), and the `lvm({'X', batchX, 'Y', Y_data}, 2)` call sequence map cleanly to a Python rewrite.

## 4. Exercise summaries

### 4.1 Kamyr digester — soft-sensor exercise with known time lags

- **Source:** [`exercises/kamyr-digester.tex`](https://github.com/kgdunn/GSK-course-notes/blob/main/exercises/kamyr-digester.tex). Dataset: [`tests/kamyr-digester-subset.xls`](https://github.com/kgdunn/GSK-teaching/blob/main/tests/kamyr-digester-subset.xls) (51 KB).
- **Process context.** Pulp / paper kraft mill in Alberta — continuous Kamyr digester. Critical quality variable $y$ = **Kappa number** (higher = more residual lignin, e.g. cardboard; lower = bleachable pulps), measured at the end of the process. Time delays from some $X$ variables to $y$ exceed **3 hours**, which makes feedback control hard.
- **Dataset shape.** Slow variables sampled once per hour; faster variables provided as hourly averages. Known-lag columns already shifted (e.g. `ChipLevel4` is chip level lagged 4 h; `BlackFlow2` is black-liquor flow lagged 2 h). Rows are therefore already "aligned".
- **Aim.** (1) Build a good predictive PLS model for Kappa number. (2) Understand which variables drive Kappa, so the company can reduce its variability.
- **Suggested steps (verbatim from `.tex`):** outlier cleanup → loadings to understand variable relationships → confirm via raw-data and coefficient plots → quantify $R^2_y$ and RMSEE → obs-vs-pred and obs-and-pred time-series → add 1-hour and then 2-hour lags of Kappa to $X$ → discuss implementation → test-set split for RMSEP.
- **Methods used.** PLS, lag-augmented $X$-space, time-series prediction, RMSEE / RMSEP, cross-validation.
- **Recommended Phase B action.** Convert `.xls` → CSV/parquet. Implement as a guided self-study exercise in `process-improve`. **Open question:** is there a public-domain Kamyr dataset that supersedes this 51 KB subset? Check `datasets.connectmv.com` redirect; otherwise keep this file.

### 4.2 Food texture — small intro PCA

- **Source:** [`exercises/food-texture.tex`](https://github.com/kgdunn/GSK-course-notes/blob/main/exercises/food-texture.tex). Dataset: not in `gsk-teaching`; available at `datasets.connectmv.com`.
- **Process context.** Pastry quality assessed by **5 attributes**: percentage oil, density, crispiness (7 = soft → 15 = crispy), fracture angle, hardness (force required to break).
- **Dataset shape.** Centering vector $\bar{x} = [17.2, 2857.6, 11.5, 20.9, 128.2]$; SD vector $[1.6, 124.5, 1.78, 5.47, 31.1]$. At least 36 observations (pastry "B758" is row 36 in the worked example).
- **Storyline.** Standard centering + scaling preprocessing → 2-component PCA: PC1 = 60.6 %, PC2 = 25.9 % (total 86.5 %). Per-variable $R^2$ after 2 components: oil 81.2 %, density 86.0 %, crispy 90.9 %, fracture 83.4 %, hardness 91.0 %. Loadings $p_1 = [0.46, -0.48, 0.53, -0.50, 0.15]$. Worked manual calculation of $t_1$ for pastry B758 = 3.59. Includes (commented-out) extension on sample 33, $t_1 = -4.2$, demonstrating low-$t_1$ characteristics.
- **Learning objectives.** Hands-on understanding of centering & scaling, score calculation by hand, loading interpretation, orthogonality of $p_1, p_2$ → independent adjustment of hardness without disturbing other attributes.
- **Methods used.** PCA, manual score calculation.
- **Recommended Phase B action.** **Strong candidate for `pid-book` Chapter 7 rather than `process-improve`** — small, pedagogical, manual-arithmetic-friendly. Decide in Phase B.

### 4.3 Wafer thickness — PCA outlier hunt with iterative refitting

- **Source:** [`exercises/wafer-thickness.tex`](https://github.com/kgdunn/GSK-course-notes/blob/main/exercises/wafer-thickness.tex). Dataset: link to `datasets.connectmv.com/info/silicon-wafer-thickness`.
- **Process context.** Silicon wafer manufacturing. Nine thickness measurements per wafer, taken at fixed locations (see figure in `.tex`).
- **Dataset shape.** 184 observations; the first ~100 used for model-building, remainder as test data. 9 variables.
- **Suggested steps (verbatim).** Build PCA on first 100 → score plot → contribution-tool outlier investigation → verify outliers in raw data → exclude & refit → repeat until clean → interpret $p_1$ → interpret $R^2$ and $Q^2$ → interpret $p_2$ → time-series of $t_1$ and $t_2$ → run all 184 observations as test data → check whether previously-excluded outliers still show up, and whether new ones appear in 101–184.
- **Learning objectives.** Iterative outlier removal as a workflow; testing model on held-out data; using $t_1, t_2$ time-series to detect process drift.
- **Methods used.** PCA, SPE, contributions, $t$ time-series plots, model-vs-test discipline.
- **Recommended Phase B action.** **Strong candidate for `pid-book` Chapter 7 rather than `process-improve`** — short, focused, ideal companion to the food-texture exercise. Decide in Phase B.

## 5. Datasets to copy/convert (Phase B inputs)

All live in `kgdunn/gsk-teaching` at the `87cfb3b` commit on the default branch. Phase B will need to convert `.mat`/`.xls` → CSV or parquet.

| Path | Size | SHA (blob) | Used by |
|------|------|-----------|---------|
| `datasets/FMC.mat` | 1 702 776 B | `6ca6301e9aa80d8cfe994c2d7564dff51cfef2e9` | FMC case |
| `datasets/DuPont.mat` | 440 025 B | `8f778319511cd43d51283a65a4ca8e3da3fccde5` | DuPont case |
| `tests/SBRDATA.mat` | 765 976 B | `d3d4fd5161dc42929b9f5d416f28c9254705efcf` | SBR case |
| `tests/SBR-expected.mat` | 29 692 B | `f579145f60926bbc0c285b36298e6652dc55b588` | SBR expected-output reference for testing |
| `tests/kamyr-digester-subset.xls` | 52 736 B | `7232f006adf3c0afcceb18ddae8edba7d3c26c36` | Kamyr exercise |
| `tests/LDPE-PCA.mat` | 9 054 B | `136386308a53f03ece62ef9ff770601f35e7237f` | (unit-test fixture; possible bonus PCA exercise) |
| `tests/LDPE-PLS.mat` | 27 665 B | `3b9862c968258eb5c1e38183a6b19ac6fe4958b3` | (unit-test fixture; possible bonus PLS exercise) |

Food-texture and wafer-thickness datasets are not in `gsk-teaching`; they live at `datasets.connectmv.com` per the `.tex` source. Check whether those URLs still resolve; if not, recover the data from elsewhere before Phase B starts.

## 6. Topical modules (scaffolding)

These 11 `.tex` files in `gsk-course-notes/gsk/` are the pedagogical scaffolding around the case studies. Not full case studies themselves — but they encode how Kevin Dunn introduced each concept. They are reference reading for Phase B authors (and possibly cross-referenced from `process-improve` docs), not content to be rebuilt verbatim.

| File | Lines | One-line summary |
|------|-------|------------------|
| `want-from-data.tex` | 316 | Framing: what objectives do we have from a dataset? |
| `data-types.tex` | 276 | Continuous / categorical / batch / spectral data shapes |
| `lv-concept.tex` | 167 | Latent-variable intuition |
| `pca-intro.tex` | 477 | PCA introduction |
| `pca-nipals.tex` | 601 | NIPALS algorithm walk-through |
| `MLR-PCR-PLS.tex` | 1 137 | Multiple linear regression → PCR → PLS |
| `monitoring.tex` | 789 | Process monitoring & control (includes Paxil CR example added 2012-05-14) |
| `soft-sensors.tex` | 338 | Soft-sensor modelling |
| `classification.tex` | 928 | LDA / PLS-DA / classification |
| `preprocessing.tex` | 410 | Centering, scaling, missing data, block scaling |
| `multiblock.tex` | 247 | Multiblock layouts (background for the FMC case) |

## 7. Not carried forward

- **`gsk-teaching/*.m` framework code** (`mblvm`, `mbpls`, `mbpca`, `block`, `block_base`, `block_batch`, `lvmplot`, `lvm`, `lvm_opt`, `unit_tests`, etc.). `process-improve` already provides Python equivalents — these are reference only. `example_FMC.m` is the one exception: its top-level driver script is worth porting.
- **`gsk-course-notes/classes/`** (13 LaTeX class drivers) and **`gsk-course-notes/slides/`** (38 slide topics). Presentation layer; the underlying content lives in `examples/`, `exercises/`, and `gsk/` and is what we carry forward.
- **`gsk-teaching/readings/`** (5 PDFs: Berglund & Wold; Hoskuldsson; Kohonen; Naes; Wangen & Kowalski). Cite in `process-improve` references if relevant; do not host PDFs.

## 8. Phase B routing recommendation

| Item | Recommended destination |
|------|-------------------------|
| FMC | `process-improve` — flagship multiblock batch case study |
| DuPont Nylon | `process-improve` — batch PCA outlier-hunt case study |
| SBR rubber | `process-improve` — batch PLS fault-diagnosis case study |
| Kamyr digester | `process-improve` — soft-sensor / time-lag PLS exercise |
| Food texture | `pid-book` Ch 7 (small intro PCA, hand-calculation) |
| Wafer thickness | `pid-book` Ch 7 (PCA outliers, SPE, contributions) |

The food-texture and wafer-thickness exercises are pedagogically simpler and pair well with the Chapter 7 narrative of the PID book. The four batch / process examples carry the heavy LVM machinery that `process-improve` exists to demonstrate. **Final routing decisions are Phase B's call.**

## 9. Provenance

- `gsk-course-notes` head:  `303921cea91b0f661d08016b7f238357ca91f9f6` (2012-05-21).
- `gsk-teaching` head:  `87cfb3bbb02fa623bd19bd20e2813bff4a0cf5d4` (2011-05-16).
- All file paths, sizes, and blob SHAs above were read directly from those commits via the GitHub MCP server on 2026-05-13.
