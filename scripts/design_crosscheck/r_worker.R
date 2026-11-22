# Build designs with FrF2, DoE.base, rsm and daewr: one JSON request per line in, one "@@"-prefixed JSON line out.
# compare.py starts this script. Packages outside R's default library are found through R_LIBS_USER.
suppressPackageStartupMessages({
  library(jsonlite); library(FrF2); library(DoE.base); library(rsm); library(daewr)
})

# Signed generators, as FrF2 takes them: list(c(1, 2), -c(1, 3)) for D = AB, E = -AC.
signed <- function(q) Map(function(p, s) s * unlist(p), q$positions, q$signs)

# A DoE.base "design" object as numbers, with its block labels when it has them.
frame <- function(d) {
  df <- as.data.frame(d)
  fn <- names(factor.names(d))
  m <- sapply(df[fn], function(col) as.numeric(as.character(col)))
  bn <- design.info(d)$block.name
  list(matrix = unname(m), blocks = if (is.null(bn)) NULL else as.integer(df[[bn]]))
}

# Factor columns as level codes 0, 1, 2, ...
codes <- function(d) unname(sapply(as.data.frame(d), function(col) as.integer(as.factor(col)) - 1L))

xcols <- function(d, k) unname(as.matrix(as.data.frame(d)[paste0("x", seq_len(k))]))

handle <- function(q) {
  k <- q$k
  switch(q$kind,
    fraction = {  # FrF2: a full or regular fractional two-level factorial, blocked on request
      a <- list(nfactors = k, randomize = FALSE)
      if (!is.null(q$runs)) a$nruns <- q$runs
      if (!is.null(q$resolution)) a$resolution <- q$resolution
      if (!is.null(q$positions)) a$generators <- signed(q)
      if (!is.null(q$block_words)) a$blocks <- unlist(q$block_words)
      if (!is.null(q$n_blocks)) a$blocks <- q$n_blocks
      if (!is.null(q$block_words) || isTRUE(q$allow_2fi)) a$alias.block.2fis <- TRUE
      frame(do.call(FrF2, a))
    },
    pb = frame(pb(nruns = q$runs, nfactors = k, randomize = FALSE)),
    fullmixed = list(matrix = codes(fac.design(nlevels = unlist(q$levels), randomize = FALSE))),
    oa = {
      d <- oa.design(nlevels = unlist(q$levels), randomize = FALSE)
      list(matrix = codes(d), origin = paste(design.info(d)$origin, collapse = " "))
    },
    ccd = {  # rsm
      nb <- k - length(q$positions)
      a <- list(
        basis = as.formula(paste("~", paste0("x", seq_len(nb), collapse = " + "))),
        n0 = unlist(q$n0), alpha = q$alpha, inscribed = isTRUE(q$inscribed),
        randomize = FALSE, oneblock = isTRUE(q$oneblock)
      )
      if (nb < k) {
        a$generators <- lapply(seq_along(q$positions), function(i) {
          rhs <- paste0("x", unlist(q$positions[[i]]), collapse = "*")
          as.formula(sprintf("x%d ~ %s%s", nb + i, if (q$signs[[i]] < 0) "-" else "", rhs))
        })
      }
      if (!is.null(q$cube_blocks)) a$blocks <- as.formula(paste("Block ~", q$cube_blocks))
      d <- do.call(ccd, a)
      df <- as.data.frame(d)
      list(matrix = xcols(d, k), blocks = if ("Block" %in% names(df)) as.integer(as.factor(df$Block)) else NULL)
    },
    bbd = list(matrix = xcols(bbd(k, n0 = q$n0, block = FALSE, randomize = FALSE), k)),
    dsd = list(matrix = unname(as.matrix(DefScreen(m = k, c = 0, center = 0, randomize = FALSE)))),
    stop("unknown request kind: ", q$kind)
  )
}

con <- file("stdin")
open(con)
while (length(line <- readLines(con, n = 1)) > 0) {
  out <- NULL
  junk <- capture.output(  # the packages print progress to stdout: keep the reply line clean
    out <- tryCatch(
      suppressWarnings(suppressMessages(handle(fromJSON(line, simplifyVector = FALSE)))),
      error = function(e) list(error = conditionMessage(e))
    )
  )
  cat("@@", toJSON(out, auto_unbox = TRUE, digits = NA, null = "null"), "\n", sep = "")
  flush(stdout())
}
