# Writes reference_lmertest3.json: SensMixed's final unbalanced models, refitted with a
# current lme4 and lmerTest. See README.md. Run from the repository root, in the "mixed"
# target of tools/r/Dockerfile, after reference.R.
suppressPackageStartupMessages({
  library(lmerTest)
  library(jsonlite)
})
here <- "tests/fixtures/sensmixed_mam"
sensmixed <- fromJSON(file.path(here, "reference.json"), simplifyVector = FALSE)

tvbo <- read.csv(file.path(here, "tvbo.csv"))
for (factor_column in c("Assessor", "TVset", "Repeat", "Picture")) {
  tvbo[[factor_column]] <- factor(tvbo[[factor_column]])
}
unbalanced <- tvbo[-unlist(sensmixed$dropped_rows), ]
attributes <- names(tvbo)[5:ncol(tvbo)]
# lmerTest 2.0 refits with these contrasts before its Type I tests.
options(contrasts = c("contr.SAS", "contr.poly"))

refit <- function(attribute) {
  data <- unbalanced
  data$x <- as.vector(scale(predict(lm(reformulate("TVset * Picture", attribute), data)), scale = FALSE))
  random <- setdiff(names(sensmixed$cases$unbalanced[[attribute]]$variance_components), "Residual")
  model <- lmer(
    reformulate(c("TVset * Picture", "Assessor:x", sprintf("(1 | %s)", random)), attribute),
    data = data,
    control = lmerControl(optimizer = "bobyqa", optCtrl = list(rhobeg = 1e-3, rhoend = 1e-12))
  )
  tests <- anova(model, type = 1, ddf = "Satterthwaite")
  rownames(tests)[rownames(tests) == "Assessor:x"] <- "Scaling"
  lapply(setNames(rownames(tests), rownames(tests)), function(term) list(
    den_df = tests[term, "DenDF"],
    f_value = tests[term, "F value"]
  ))
}

reference <- list(
  generated_with = list(
    R = paste(R.version$major, R.version$minor, sep = "."),
    lmerTest = as.character(packageVersion("lmerTest")),
    lme4 = as.character(packageVersion("lme4"))
  ),
  unbalanced = lapply(setNames(attributes, attributes), refit)
)
write_json(reference, file.path(here, "reference_lmertest3.json"), auto_unbox = TRUE, digits = I(12), pretty = TRUE)
