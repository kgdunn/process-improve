# Predicting TPLS blends that use a new material

TPLS links the quality `Y` of a blend to its formula `F`, to the properties `D` of the materials
in it, and to the process conditions `Z` (Garcia-Munoz, 2014). A *new material* is one that no
training blend used: a new row of `D` (its properties are measured) and a new column of `F` (its
fraction in the blends to predict). `TPLS.diagnose` refuses such a column. This folder shows why
the obvious workarounds fail, five ways that work, and how well each one does on simulated data
with a known truth and on the bundled pyphi data set.

**In short.** A new material can be predicted from its properties alone, with three changes to
how TPLS is used:

1. Score the new material through the D block as a new observation, and give it an F loading
   of zero. Nothing else in the fitted model changes.
2. Centre F instead of autoscaling it, so that a material's effect depends on its properties
   and not on how its usage happened to vary in training.
3. Do not deflate the D block. This deflation, more than the new material, is what defeats
   every workaround: it caps the TPLS fit at R2 0.60 to 0.78 in simulations where the truth
   is at 0.97 to 0.99.

With these, leave-one-material-out error falls from 5.6 to 7.3 times the noise (every workaround
on standard TPLS) to 1.7 to 2.9 times in the three scenarios with ideal mixing. The best method
(approach 3a, mixing-rule PLS) is then within 8% of the error for a material that was in
training. Measuring one blend with the new material cuts the remaining error by a further 18%,
two blends by 29%.

## Why it fails today

1. **`diagnose` refuses the column.** Every F column must be a training material.
2. **The obvious workaround drops the material, and moves every other prediction.** Appending
   the new material to D, with an all-zero column in F, and refitting looks right. But that
   column has no variance, so the default autoscaling of F gives it a NaN scale, and TPLS
   excludes it: at prediction the new material's fraction is multiplied by zero, as if it were
   not in the blend. The refit also changes the model, because the new row changes the centring
   and scaling of D and enters the D-block deflation: in the simulation, adding one material
   that no blend uses moved the predictions of the blends that do not use it by up to 0.78 Y
   standard deviations.
3. **Autoscaled F ties a material's effect to how it was used, not to what it is.** With F
   divided by the standard deviation `sigma_j` of each material's fraction, the effect of
   material `j` on Y is `b(d_j) / sigma_j`, where `b(d_j)` depends on its properties alone. A
   material used in many blends at varied fractions is damped; one used once is amplified. A
   new material has no `sigma` at all. Splitting one lot into two identical half-lots is a
   direct test: it changes the predictions of an autoscaled model, and leaves a centred model
   exactly unchanged (checked on the pyphi data). There the usage spread `sigma_j` of the 9
   Group 2 lots spans a factor of 20, so in the autoscaled model two lots with identical
   properties would get effects 20 times apart. The lot effects in that data set are small, so
   the change is small (0.002 Y standard deviations); in the simulation with uneven usage the
   autoscaled model's error on new materials is 15% higher than the centred model's.
4. **The D-block deflation caps the fit.** After each component TPLS deflates D by `r v'`,
   which removes that material direction from D for good. When the super score also carries Z,
   or a second group of materials, Y needs that direction again in a later component, with a
   different Y loading, and it is gone. The fit stops improving well short of what the data
   allow (table below). This hurts every prediction, not only those of new materials.

In-sample R2 with 8 components, mean of 6 simulated data sets of 80 blends (`fit_ceiling` in
`study.py`). *Truth* is the R2 of the noise-free values, the most any model should reach.

| Data | TPLS (default) | TPLS, centred F | TPLS, centred F, no D deflation | PLS on [Z, F D] | Truth |
|---|---:|---:|---:|---:|---:|
| Group A only | 0.943 | 0.948 | 0.948 | 0.948 | 0.949 |
| Group A + Z | 0.702 | 0.759 | 0.984 | 0.984 | 0.985 |
| Groups A + B | 0.782 | 0.599 | 0.966 | 0.963 | 0.969 |
| Groups A + B + Z | 0.771 | 0.722 | 0.968 | 0.977 | 0.988 |

With one group and no Z every variant reaches the truth. Add Z or a second group and the
D-deflated models stop between 0.60 and 0.78, while the same model without the deflation keeps
up with PLS on the mixture properties.

The deflation also makes TPLS fragile. Adding to D one material that no blend uses, and
refitting, moved the predictions of the blends by up to 0.78 Y standard deviations (autoscaled
F) and 0.65 (centred F); without the deflation, by 0.04.

## Why a new material can be predicted at all

TPLS treats the materials (rows of D) exactly as a PCA or PLS model treats observations. For
component `a` the material scores are `r_a = D_a s_a / (s_a' s_a)`, after which D is deflated,
`D_(a+1) = D_a - r_a v_a'`. A new material is scored by running the same recursion on its own
preprocessed property row: it is a new observation of the D block, and it gets a T2 and an SPE
there too.

The blend scores are `t_a = f_a r_a / (r_a' r_a)`, with the F row deflated by
`f_(a+1) = f_a - t_a p_a'`. Unrolled, the F loadings `p` enter only through the scalars
`p_b' r_a`, which are sums over the materials. Give the new material an F loading of zero:

* these scalars do not change, so no existing prediction changes;
* the new material adds `f_new r_new,a` to every component, the same rule every training
  material follows;
* zero is the value TPLS itself computes, `p = F' t / t't`, for a column of F that is zero in
  every training blend.

With F centred (not scaled) the prediction is then affine in the fractions, and every material's
coefficient is one function of its properties, `b(d)` (`material_coefficients` computes it): the
model's own property-to-effect map, the same for old and new materials.

## Five ways to predict with a new material

All five are implemented on a fitted `TPLS` in `new_materials.py` and `methods.py`.

### 1. Project the new material through the D block

Score the new material through `S` and `V` like a new observation, give it a zero F loading, and
run the usual prediction (`project_materials`, `predict`). Centred F is the consistent choice
(reason 3) and matters when usage is uneven; the D-block deflation has to go (reason 4,
`TPLSWithoutDDeflation`). This is the TPLS-native answer: no refit, and the T2 and SPE of the
material in the D block say how far it is from the materials the model has seen.

### 2. Pseudo-recipe: write the new material as a mixture of known ones

Find non-negative weights, summing to one, over the nearest training materials that reproduce
the new material's properties (`pseudo_recipe`). A blend that uses `x` of the new material then
uses `x * alpha_k` of each of those materials instead (`substitute`), and the unchanged model
predicts it. This works with any fitted TPLS, the default autoscaled one included, and the
reconstruction residual says whether the new material lies inside the region spanned by the
known ones. On a centred, linear model it gives the same answer as approach 1 whenever that
residual is zero.

### 3. Mixing rules: collapse the T to an L

Describe each blend by the fraction-weighted mean of every property of each group (the ideal
mixing rule), then regress Y on `[Z, means]` with PLS. A new material needs nothing special: its
properties enter the mean. With one group of materials and no Z, TPLS with centred F and without
the D-block deflation *is* this PLS (the predictions agree to 1e-8), which makes each a check on
the other; with Z, TPLS is the multiblock version and reaches the same fit. Non-ideal mixing
(dissimilar materials that do not blend linearly) adds the second central moments of the
blend's materials, in the space of the leading principal components of each group's properties.

### 4. Hierarchical: one property effect for all, plus an offset per known material

Joint-Y PLS gives several sites one latent space, defined by what they share, and lets each site
keep its own loadings. Here every material shares one property-to-effect map (approach 1), and
each known material keeps an offset: the part of its effect its measured properties do not
explain, estimated from the residuals of its blends and shrunk towards zero by a ridge
regression whose penalty GCV chooses; for that penalty the ridge estimate is the best linear
unbiased predictor of a random material effect. The
fitted shrinkage gives `tau2`, the variance of material effects that D cannot explain, which
says how far zero-shot prediction can be trusted. A new material gets offset zero and a
prediction variance of `tau2 * f_new^2`; its blends still gain, because the offsets of the
other materials in them are known.

### 5. Few-shot calibration: product transfer to the new material

In product transfer between sites, a few runs at the new site calibrate it inside the joint
model. Here: measure one to three blends that use the new material, and update its offset with
the same prior as in approach 4,
`delta = sum(f_i e_i) / (sum(f_i^2) + sigma2 / tau2)`, where `e_i` are the zero-shot residuals of
those blends. The rest of its blends are then predicted with that offset.

## Results

### Simulation, new material, nothing measured with it

Leave-one-material-out: for each of the 14 group-A materials in turn, the blends that use it are
removed from training, the material is removed from D and F, and those blends are predicted from
its properties alone. The numbers are the root mean squared error of prediction divided by the
noise standard deviation, so 1.0 is the best any model can do. Each is the mean of 6 simulated
data sets of 80 blends, at the best number of components in {4, 6, 8} for that method.

| Method | Ideal mixing | Uneven usage | Hidden trait | Non-ideal mixing |
|---|---:|---:|---:|---:|
| *Standard TPLS (D deflated)* | | | | |
| Ignore the material | 6.20 | 7.16 | 6.31 | 7.33 |
| Nearest known material | 5.78 | 7.06 | 5.87 | 6.83 |
| Zero-padded refit | 5.82 | 6.70 | 5.89 | 6.87 |
| 1. Projection, autoscaled F | 5.58 | 6.79 | 5.77 | 6.82 |
| 1. Projection, centred F | 6.60 | 6.11 | 6.24 | 7.21 |
| 2. Pseudo-recipe | 5.63 | 6.67 | 5.70 | 6.67 |
| *Without the D deflation* | | | | |
| 1. Projection, autoscaled F | 1.92 | 2.47 | 2.54 | 4.00 |
| 1. Projection, centred F | 2.17 | 2.14 | 2.63 | 4.03 |
| 2. Pseudo-recipe | 2.43 | 2.86 | 2.90 | 4.21 |
| 3a. Mixing-rule PLS | **1.76** | **1.73** | 2.35 | 3.79 |
| 3b. Mixing-rule PLS + moments | 3.08 | 3.33 | 3.48 | **3.51** |
| 4. Hierarchical offsets | 1.83 | 1.99 | **2.14** | 3.76 |
| *Reference: the material seen in training* | 1.86 | 1.60 | 2.28 | 3.58 |

The reference is approach 1 (centred F, no D deflation) predicting each blend from all the other
blends, so the material has been used in training. Scenarios: *ideal mixing*, properties explain
every material effect; *uneven usage*, some materials are used far more than others; *hidden
trait*, part of each material's effect is invisible in D; *non-ideal mixing*, blends of
dissimilar materials depart from the fraction-weighted average.

* **The D deflation decides it.** On standard TPLS no method gets below 5.6 times the noise.
  Without it, approaches 1, 2, 3a and 4 are between 1.7 and 2.9 in the first three scenarios.
* **Approach 3a loses almost nothing to novelty.** It is within 8% of the reference in every
  scenario, and below it with ideal mixing: 3a is more accurate than approach 1 in every
  column, enough to beat approach 1 even when approach 1 has seen the material.
* **Centred F matters when usage is uneven** (2.14 against 2.47 for autoscaled F). With even
  usage the median-scale guess of approach 1 is harmless.
* **Offsets help when D misses something.** With a hidden trait, approach 4 (2.14) beats
  approach 1 (2.63): the hidden effects of the known materials in each blend are learned. The
  new material's own hidden effect stays unknown until it is measured (approach 5).
* **Non-ideal mixing costs every model.** The moments (3b) help there and cost the most when
  mixing is ideal (3.08 against 1.76): add them only when blends of dissimilar materials are
  known to behave differently.

### Simulation, after measuring one to three blends with the new material

Hidden-trait scenario, 8 components, mean of 6 data sets, 3 random draws of the measured blends
per material. The error is on the new material's other blends.

| Method | 1 blend | 2 blends | 3 blends |
|---|---:|---:|---:|
| 4. Zero-shot, nothing measured | 2.15 | 2.15 | 2.15 |
| 5. Update the new material's offset | **1.77** | **1.53** | **1.47** |
| Refit approach 4 with the measured blends | 2.85 | 2.00 | 1.83 |
| Refit the default TPLS with the measured blends | 10.85 | 7.22 | 6.93 |

One measured blend cuts the error by 18%, two by 29%. Refitting is worse than updating, and
with one blend worse than not measuring at all: a refit re-estimates every parameter from data
that say little about most of them, while the update changes only what the new blends inform,
which is the logic of product transfer. Refitting the default TPLS is the worst of all, because
autoscaling turns the one non-zero entry of the new material's F column into a value eight
standard deviations from its mean.

### Is the new material inside what the model has seen?

The new material's Hotelling's T2 and SPE, in a 3-component PCA of the other materials'
properties, flag it before any blend is made (`domain.py`). In the simulation one material was
moved 4 trait standard deviations away from the others (6 data sets, 8 components):

| Material | Approach 1 | 3a | 4 | T2 / 95% limit | SPE / 95% limit |
|---|---:|---:|---:|---:|---:|
| Far from the others | 3.14 | 1.80 | 2.89 | 1.80 | 1.51 |
| One of the other 13 | 1.92 | 1.69 | 1.48 | 0.17 | 0.62 |

(medians; the first three columns are RMSEP / noise SD). The T2 limit flags the far material in
every data set and crosses for 7% of the ordinary ones, close to the nominal 5%. The SPE limit,
taken from the training materials' own residuals, is too tight (31% of ordinary materials cross
it); a limit from leave-one-material-out residuals would be the fix. The far material is
predicted with 64% more error than the others by approach 1 and 95% more by approach 4, but
only 7% more by 3a: in this simulation the truth stays linear outside the range, which a real
process need not do.

### The pyphi data

Leave-one-lot-out on the T-shaped data set bundled with the package (105 blends, 5 groups, 230
lots; from pyphi by Garcia-Munoz). Q2 is computed on standardised Y, so 1 is perfect and 0 is
predicting the mean; each method uses the number of components, from 1 to 4, that is best over
all lots.

| Method | Group 1 (162 lots) | Group 2 (9) | Group 3 (22) | Group 4 (19) | Group 5 (18) | All lots |
|---|---:|---:|---:|---:|---:|---:|
| Ignore the material | 0.205 | -0.030 | 0.144 | 0.028 | 0.041 | 0.103 |
| 1. Projection, centred F, no D deflation | 0.207 | -0.029 | 0.142 | 0.026 | 0.039 | 0.102 |
| 2. Pseudo-recipe | 0.196 | -0.029 | 0.144 | 0.028 | 0.041 | 0.100 |
| 3a. Mixing-rule PLS | 0.207 | -0.043 | 0.158 | 0.076 | 0.098 | **0.121** |
| 3b. Mixing-rule PLS + moments | -0.032 | -0.215 | -0.776 | -0.049 | -0.059 | -0.185 |
| 4. Hierarchical offsets | 0.243 | -0.051 | 0.168 | 0.028 | 0.062 | 0.120 |

This data set cannot rank the methods. Every TPLS variant scores 0.10, the same as ignoring the
new lot (`results/pyphi_leave_one_lot_out.csv` has them all): the properties of a lot say
almost nothing about Y here that the process conditions do not. What the data do show: the
projection runs unchanged on five groups and on a D block with missing values, 3a and 4 are
again the best, and the moments overfit, because most blends use one lot of each group, so
their spread is zero in most rows.

## Three rules that follow

* **A model that can take new materials sees a blend only as a distribution of material
  properties.** Two tests must leave every prediction unchanged: splitting a lot into two
  identical half-lots, and adding to D a material that the blend does not use. A model that
  passes both depends on the formula only through the fractions placed on the materials'
  property vectors. Fraction-weighted sums of functions of the properties (the mixing rules of
  approach 3) are the linear models of that kind, and TPLS with centred F is the simplest of
  them. The tests are cheap; run them on any T-shaped model before trusting it with a new
  material. TPLS with autoscaled F fails the first; the zero-padded refit fails the second.
* **Choose the number of components by leaving materials out, not blends.** Leave-one-blend-out
  rewards a model for what it learns about each known material, which is no help for a new one.
* **Check the material before the prediction.** A new material beyond the T2 limit of the
  training materials is an extrapolation, whatever the method.

## What could change in the package

These are for discussion; each changes public behaviour.

* `TPLS.diagnose(X, d_new=...)`: accept F columns for materials given in `d_new`, scored as in
  `predict` here (approach 1), and return their T2 and SPE in the D block.
* An F preprocessing that centres without scaling, beside `skip_f_matrix_preprocessing`. It keeps
  a material's effect a function of its properties alone, which autoscaling does not, and it
  centres the blend scores, which raw fractions do not.
* An option to skip the D-block deflation, or a change of default after a check against the
  results of Garcia-Munoz (2014), since it caps the fit whenever Z or a second group is present.
* `TPLS.material_coefficients(d)`: the effect per unit fraction of any material, old or new.

## Not done

* A kernel version of approach 3 (a Gaussian-process prior over material effects, with a
  kernel between blends built from a kernel between materials); it is the natural extension
  of the moments for strongly non-linear mixing.
* Calibrating the few-shot offset in the latent space (A parameters) instead of in Y
  (one per quality), which would need fewer measured blends when Y has many columns.
* Substitution: with `b(d)` known, the fraction of a new material that reproduces the quality
  of the current formula is a small least-squares problem, the analogue of product transfer.

## Reproduce

```bash
cd examples/multivariate/tpls_new_materials
python study.py     # about 40 minutes on 4 cores; writes results/
python domain.py    # applicability domain and the far material, after study.py
```

`simulate.py` documents the simulated truth; `study.py` the leave-one-material-out protocol.
