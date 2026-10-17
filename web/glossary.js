// What the page's words mean, each said once: the definition a dotted word opens (terms.js). The
// definitions follow the book, Process Improvement using Data, in its own words where it has them.
// A definition that depends on the study, such as a design's or a comparison measure's, is given
// with its word where the page draws it.

export const GLOSSARY = {
  // The brief
  response: "The result of each run, which you measure, such as a yield, a strength or a time.",
  typical: "Where the response sits now, at the usual settings. Practice results are simulated around it.",
  noise:
    "How much the response varies when one run is repeated at the same settings: the standard deviation of those " +
    "repeats.",
  change:
    "The smallest difference in the response, from a factor's low setting to its high one, that matters to you. " +
    "Each design's power is its chance of finding a change this size.",
  factor:
    "Something you can change to influence the response, set differently from run to run, such as a temperature, " +
    "a speed or a supplier.",
  categorical:
    "A factor with two labels rather than a range of numbers, such as supplier A and B: the first is coded -1, " +
    "the second +1. Only the two-level designs take one.",
  limit:
    "Some combinations of settings are unsafe, infeasible or not of interest. A limit rules them out: an " +
    "inequality in the factors' own units, such as T + 25*P <= 250, which every run must meet.",
  "main-effect":
    "How much the response changes, on average over the other factors, when one factor goes from its low setting " +
    "to its high one.",
  interaction: "Two factors interact when the effect of one depends on the level of the other.",
  square:
    "A factor's squared, or quadratic, term: it lets the factor's effect curve, so the best setting can lie " +
    "inside its range rather than at one end.",
  quadratic:
    "The intercept, the main effects, every two-factor interaction and every square: the simplest model that can " +
    "place a maximum, a minimum or a saddle inside the region.",
  coefficient:
    "The change in the response for a one-unit change in a term's coded value, the other terms held constant. " +
    "The intercept is the response predicted at the center of every range.",

  // The design
  "center-points":
    "Runs with every factor at the middle of its range. Repeated, they estimate the run-to-run noise, and they " +
    "test whether the response curves.",
  axial:
    "Runs that move one factor at a time away from the center along its own axis, the others at their middles: " +
    "they let a central composite design estimate the squares.",
  resolution:
    "How clearly a fractional factorial separates its effects. At III, main effects are aliased with two-factor " +
    "interactions; at IV, two-factor interactions with each other; at V, only with three-factor ones.",
  replicates:
    "Each run but the center runs is made this many times, each a whole new run, set up, made and measured, so " +
    "the noise can be estimated from the repeats.",
  blocks:
    "Groups of runs that share conditions the study does not control, such as one day or one batch of raw " +
    "material: a shift between groups is taken up by a block effect, not a factor's. 1 for none.",
  aliasing:
    "Two terms are aliased, or confounded, when the design changes them together, so their effects cannot be " +
    "told apart: what is estimated is their sum.",

  // The runs
  "std-order":
    "The run's place in the order the design is listed in before it is randomized; for a two-level factorial, " +
    "the standard (Yates) order.",
  "run-order":
    "The order to make the runs in, chosen at random, so that disturbances you cannot measure show up as noise " +
    "rather than biasing a factor's effect.",
  "true-process":
    "The process the practice results are simulated from. Its coefficients are known, so the analysis can be " +
    "checked against them.",

  // The analysis
  coded:
    "Each factor centered and scaled, as its setting less the center, over half its range: it runs from -1 at " +
    "its low setting to +1 at its high one.",
  "model-terms":
    "Tick the terms to fit; the intercept is always in. A term the design cannot tell apart from another is " +
    "estimated with it, as their sum.",
  "std-error": "How precisely the coefficient is estimated: the standard deviation of its estimate.",
  interval:
    "The estimate plus or minus about two standard errors: intervals made this way hold the true coefficient " +
    "95% of the time.",
  "p-value":
    "The probability of an estimate at least this far from zero if the true coefficient were zero. A small value " +
    "is evidence that it is not.",
  "true-value":
    "What the coefficient estimates for the process the results were simulated from: the estimate the same fit " +
    "gives without noise.",
  vif:
    "The variance inflation factor: how many times larger a coefficient's variance is because its term is " +
    "correlated with the other terms in the runs fitted. 1 means uncorrelated; above 5, the standard error is more " +
    "than doubled.",
  effect: "The change in the response from a factor's low setting to its high one: twice its coefficient.",
  pse: "The pseudo standard error: a robust estimate of the standard error of an effect, from the effects themselves.",
  me: "The margin of error, the individual 95% cutoff: an effect whose size exceeds it is significant on its own.",
  sme: "The simultaneous margin of error: the stricter cutoff, for scanning all the effects at once.",
  anova:
    "The analysis of variance: how much of the variation in the results each source explains, and how much is " +
    "left in the residuals.",
  df: "Degrees of freedom: how many independent pieces of information each source takes.",
  "sum-of-squares": "The variation a source accounts for, as a sum of squared differences.",
  "f-value":
    "A source's mean square over the residual mean square: it compares the variance the source explains with " +
    "the variance left in the residuals.",
  r2: "The variation the model explains, as a fraction of the total variation in the results.",
  "r2-adjusted":
    "R² computed with each sum of squares divided by its degrees of freedom: a term that adds little lowers it.",
  "r2-predicted":
    "How well the model predicts each run when that run is left out of the fit. Well below R², it warns of a " +
    "model that fits the noise.",
  "residual-df":
    "The runs left after fitting: they estimate the run-to-run noise, and every test and interval rests on them.",
  "lack-of-fit":
    "The part of the residuals the model failed to explain that is not just noise. A large F, with a small " +
    "p-value, is evidence that the model is missing a term.",
  "pure-error":
    "The spread among runs repeated at the same settings: the run-to-run variability of the system, without " +
    "reference to any model.",
  "response-surface":
    "The fitted model drawn over two factors, as contours of the predicted response, with the other factors at " +
    "their centers.",
  residuals:
    "Each run's residual, its result less the model's fitted value, against that fitted value: with a good " +
    "model they show no structure.",
  "half-normal":
    "The effects' sizes, in order, against the quantiles of the half-normal distribution: effects that are " +
    "noise fall along a straight line through the origin.",
  pareto:
    "The coefficients' sizes as bars, largest first and the intercept left out: orange for a positive " +
    "coefficient, blue for a negative one.",
  fds:
    "Each design's prediction variance over the factors' region, sorted: a point at f along the bottom means a " +
    "fraction f of the region is predicted at least this well.",
  "correlation-map":
    "The correlation |r| between each pair of terms over the design's runs: 0 means the two can be estimated " +
    "independently, 1 that they cannot be told apart at all.",

  // The best settings
  "confidence-interval":
    "The range that holds the mean response at these settings, the average of many runs there, 95% of the time.",
  "prediction-interval":
    "The range that one new run at these settings falls in 95% of the time. It is wider than the confidence " +
    "interval, since one run carries the run-to-run noise too.",
  leverage:
    "How far settings sit from the runs, as the model sees them: the variance of the prediction there, in units of " +
    "the noise variance. Above every run's, the prediction leans on the model beyond the runs.",
  "confirmation-run":
    "A run made at the best settings after the analysis, to test the model there. It does not enter the fit: its " +
    "result is compared with what the model predicts.",
  "stationary-point":
    "Settings where the fitted surface is flat in every direction: its top, its bottom, or a saddle between.",
  "steepest-ascent":
    "The direction in which the predicted response rises fastest, each factor changed in proportion to its slope; " +
    "steepest descent is the opposite direction.",
  "sensitivity":
    "How much the predicted response changes when one factor alone moves a little from the best settings: a factor " +
    "that moves it most must be held most tightly.",

  // Balancing several responses
  desirability:
    "A response's score for its goal, d, from 0 where it is not acceptable to 1 where it is as good as it gets, " +
    "rising across the range you accept. Derringer and Suich's measure.",
  "overall-desirability":
    "D, the geometric mean of every response's desirability, each counted by its importance. It is 0 where any " +
    "response is not acceptable, so a balance never trades one response away entirely.",
  importance: "How much a response counts in the overall desirability, from 1 to 5, beside the others.",
  "desirability-shape":
    "How fast d rises towards the good end of its range. Evenly, in a straight line; more, a result counts well " +
    "only near the good end; less, most of the range counts nearly as well as the good end.",
  overlay:
    "Each response's acceptable boundary drawn over the same two factors, the others held at the balanced " +
    "settings. Where the regions they leave overlap, every response is acceptable.",
};

/** Phrases marked wherever the page's sentences or the library's notes use them, and the entry each opens. */
export const MARKED = {
  aliased: "aliasing",
  "coded units": "coded",
  "random run order": "run-order",
  "center of the ranges": "center-points",
  "R²": "r2",
  "Adjusted R²": "r2-adjusted",
  "Predicted R²": "r2-predicted",
  "Residual degrees of freedom": "residual-df",
  "Lack of fit": "lack-of-fit",
  "pure error": "pure-error",
  "variance inflation factor": "vif",
  PSE: "pse",
  ME: "me",
  SME: "sme",
  "confidence interval": "confidence-interval",
  "prediction interval": "prediction-interval",
  leverage: "leverage",
  "confirmation run": "confirmation-run",
  "confirmation runs": "confirmation-run",
  "stationary point": "stationary-point",
  "steepest ascent": "steepest-ascent",
  "steepest descent": "steepest-ascent",
  "overall desirability": "overall-desirability",
  desirability: "desirability",
};
