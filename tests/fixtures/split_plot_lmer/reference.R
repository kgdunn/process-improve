# Writes reference.json: lme4, lmerTest and pbkrtest on two split plots. See README.md.
# Run from the repository root, in the "mixed" target of tools/r/Dockerfile.
suppressPackageStartupMessages({
  library(lmerTest)
  library(pbkrtest)
  library(jsonlite)
})
here <- "tests/fixtures/split_plot_lmer"

variance_components <- function(model) {
  vc <- as.data.frame(VarCorr(model))
  list(whole_plot = vc$vcov[vc$grp != "Residual"], residual = vc$vcov[vc$grp == "Residual"])
}

coefficients <- function(model) {
  satterthwaite <- coef(summary(model, ddf = "Satterthwaite"))
  kenward_roger <- coef(summary(model, ddf = "Kenward-Roger"))
  terms <- rownames(satterthwaite)
  lapply(setNames(terms, terms), function(term) list(
    estimate = satterthwaite[term, "Estimate"],
    std_error = satterthwaite[term, "Std. Error"],
    df_satterthwaite = satterthwaite[term, "df"],
    df_kenward_roger = kenward_roger[term, "df"]
  ))
}

term_tests <- function(model) {
  satterthwaite <- anova(model, type = 3, ddf = "Satterthwaite")
  kenward_roger <- anova(model, type = 3, ddf = "Kenward-Roger")
  terms <- rownames(satterthwaite)
  lapply(setNames(terms, terms), function(term) list(
    F = satterthwaite[term, "F value"],
    df = satterthwaite[term, "NumDF"],
    df_satterthwaite = satterthwaite[term, "DenDF"],
    df_kenward_roger = kenward_roger[term, "DenDF"],
    p_value = satterthwaite[term, "Pr(>F)"]
  ))
}

# Seven whole plots of 2 to 5 runs; A changes only between whole plots, B within them.
unbalanced <- read.csv(file.path(here, "unbalanced.csv"))
unbalanced$plot <- factor(unbalanced$plot)
unbalanced_fit <- lmer(y ~ A * B + (1 | plot), data = unbalanced, REML = TRUE)

# Box, Hunter and Hunter's corrosion experiment: sum-to-zero contrasts, as the Python
# analysis uses for a categorical factor, so the Type III tests are the usual ones.
options(contrasts = c("contr.sum", "contr.poly"))
corrosion <- read.csv("src/process_improve/datasets/experiments/corrosion.csv")
for (column in c("Heat", "Temperature", "Coating")) corrosion[[column]] <- factor(corrosion[[column]])
corrosion_fit <- lmer(Resistance ~ Temperature * Coating + (1 | Heat), data = corrosion, REML = TRUE)

versions <- c("lme4", "lmerTest", "pbkrtest")
reference <- list(
  generated_with = c(list(R = paste(R.version$major, R.version$minor, sep = ".")),
                     setNames(lapply(versions, function(p) format(packageVersion(p))), versions)),
  unbalanced = list(
    variance_components = variance_components(unbalanced_fit),
    coefficients = coefficients(unbalanced_fit)
  ),
  corrosion = list(
    variance_components = variance_components(corrosion_fit),
    tests = term_tests(corrosion_fit)
  )
)
write_json(reference, file.path(here, "reference.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat("wrote", file.path(here, "reference.json"), "\n")
