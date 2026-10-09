# Writes reference.json: SensMixed's mixed assessor model on the TVbo panel. See README.md.
# Run from the repository root, in the "sensmixed" target of tools/r/Dockerfile.
suppressPackageStartupMessages({
  library(SensMixed)
  library(lmerTest)
  library(jsonlite)
})
here <- "tests/fixtures/sensmixed_mam"

tvbo <- read.csv(file.path(here, "tvbo.csv"))
for (factor_column in c("Assessor", "TVset", "Repeat", "Picture")) {
  tvbo[[factor_column]] <- factor(tvbo[[factor_column]])
}
attributes <- names(tvbo)[5:ncol(tvbo)]
tvbo$Product <- factor(paste(tvbo$TVset, tvbo$Picture, sep = "_"))

group_part <- function(random_terms) {
  sub(" *\\)$", "", sub("^\\(1 *\\| *", "", random_terms))
}

# The steps of SensMixed:::.stepAllAttrMAM, written out to keep what sensmixed() discards:
# the final model's variance components, scaling coefficients and REML criterion.
mam_steps <- function(data, products, replication, attribute) {
  random <- if (is.null(replication)) "Assessor" else list(individual = "Assessor", replication = replication)
  initial <- suppressMessages(SensMixed:::createLMERmodel(
    structure = list(product_structure = 3, error_structure = "ASS-REP"), data = data,
    response = attribute, fixed = list(Product = products, Consumer = NULL), random = random,
    corr = FALSE, MAM = TRUE, mult.scaling = FALSE, calc_post_hoc = FALSE, oneway_rand = FALSE
  ))
  candidates <- group_part(lmerTest:::getRandTerms(formula(initial)))
  nonzero <- suppressWarnings(SensMixed:::elimZeroVar(initial))
  kept <- group_part(lmerTest:::getRandTerms(formula(nonzero)))
  keep <- c(paste(c(products, "Assessor"), collapse = ":"), "Assessor")
  stepped <- suppressMessages(lmerTest::step(nonzero, fixed.calc = FALSE, keep.effs = keep, reduce.random = TRUE))
  final <- as(stepped$model, "merModLmerTest")
  scaling <- fixef(final)[grepl("x.scaling.private", names(fixef(final)))]
  gamma <- setNames(rep(0, nlevels(data$Assessor)), levels(data$Assessor))
  gamma[sub(":x.scaling.private", "", sub("^Assessor", "", names(scaling)))] <- scaling
  components <- as.data.frame(VarCorr(final))
  list(
    zero_variance = setdiff(candidates, kept),
    random = stepped$rand.table,
    anova = suppressMessages(anova(final, type = 1)),
    beta = as.list(1 + gamma - mean(gamma)),
    variance_components = as.list(setNames(components$vcov, components$grp)),
    reml_criterion = unname(REMLcrit(final))
  )
}

rows <- function(table, columns) {
  lapply(setNames(rownames(table), trimws(rownames(table))), function(term) {
    as.list(setNames(unlist(table[term, columns]), names(columns)))
  })
}

run_case <- function(data, products, replication) {
  official <- suppressWarnings(suppressMessages(sensmixed(
    attributes, prod_effects = products, replication = replication, assessor = "Assessor", data = data, MAM = TRUE
  )))
  lapply(setNames(attributes, attributes), function(attribute) {
    steps <- mam_steps(data, products, replication, attribute)
    reported <- official$step_res[[attribute]]
    anova_table <- steps$anova
    rownames(anova_table)[grepl("x.scaling.private", rownames(anova_table))] <- "Scaling"
    # The written-out steps must reproduce sensmixed() itself.
    stopifnot(
      isTRUE(all.equal(as.matrix(anova_table), as.matrix(reported$anova.table), tolerance = 1e-10)),
      isTRUE(all.equal(as.matrix(steps$random), as.matrix(reported$rand.table), tolerance = 1e-10))
    )
    list(
      anova = rows(anova_table, c(sum_sq = "Sum Sq", mean_sq = "Mean Sq", num_df = "NumDF",
                                  den_df = "DenDF", f_value = "F.value", p_value = "Pr(>F)")),
      random = rows(steps$random, c(chi_sq = "Chi.sq", chi_df = "Chi.DF", step = "elim.num", p_value = "p.value")),
      zero_variance = I(steps$zero_variance),
      beta = steps$beta,
      variance_components = steps$variance_components,
      reml_criterion = steps$reml_criterion
    )
  })
}

# Five rows removed, the same for every attribute: no product cell empties.
dropped <- c(3, 50, 77, 120, 160)
averaged <- aggregate(tvbo[attributes], by = tvbo[c("Assessor", "TVset", "Picture")], FUN = mean)

reference <- list(
  generated_with = list(
    R = paste(R.version$major, R.version$minor, sep = "."),
    SensMixed = as.character(packageVersion("SensMixed")),
    lmerTest = as.character(packageVersion("lmerTest")),
    lme4 = as.character(packageVersion("lme4"))
  ),
  dropped_rows = dropped,
  cases = list(
    one_way = run_case(tvbo, "Product", "Repeat"),
    factorial = run_case(tvbo, c("TVset", "Picture"), "Repeat"),
    unbalanced = run_case(tvbo[-dropped, ], c("TVset", "Picture"), "Repeat"),
    no_replicate = run_case(averaged, c("TVset", "Picture"), NULL)
  )
)
write_json(reference, file.path(here, "reference.json"), auto_unbox = TRUE, digits = I(12), pretty = TRUE)
