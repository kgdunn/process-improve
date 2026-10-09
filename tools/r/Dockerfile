# syntax=docker/dockerfile:1
#
# R with the mixed-model packages the test suite uses as an external reference. Two
# targets; see tools/r/README.md for building them and running scripts in them.
#
# Each target installs packages from a dated snapshot of CRAN (Posit Package Manager),
# so a rebuild gets the same package versions.
#
# Behind a proxy that intercepts TLS, pass its CA bundle as an optional build secret:
#   docker build --secret id=ca_bundle,src=/path/to/ca-bundle.crt ...

# --- mixed: lme4, lmerTest (Satterthwaite) and pbkrtest (Kenward-Roger) on current R ---
FROM rocker/r-ver:4.4.2 AS mixed
RUN --mount=type=secret,id=ca_bundle,required=false \
    if [ -f /run/secrets/ca_bundle ]; then \
        export SSL_CERT_FILE=/run/secrets/ca_bundle CURL_CA_BUNDLE=/run/secrets/ca_bundle; \
    fi \
    && install2.r --error --skipinstalled --ncpus 4 jsonlite lme4 lmerTest pbkrtest \
    && Rscript -e 'for (p in c("jsonlite", "lme4", "lmerTest", "pbkrtest")) cat(p, format(packageVersion(p)), "\n")'
WORKDIR /w

# --- sensmixed: SensMixed's mixed assessor model, in the environment it was released for ---
# SensMixed 2.1-0 (March 2018) was archived from CRAN in 2020 and no longer runs on a
# current stack: R >= 4.3 rejects its `&&` on vectors, and its random-effect tables break
# with later lme4. So this target is R 3.4.4 with CRAN as of 2018-04-01 (lme4 1.1-15,
# lmerTest 2.0-36), built from source. Note that `Rscript -e` in R 3.4 takes one line only.
FROM rocker/r-ver:3.4.4 AS sensmixed
ARG CRAN_SNAPSHOT=https://packagemanager.posit.co/cran/2018-04-01
RUN --mount=type=secret,id=ca_bundle,required=false \
    if [ -f /run/secrets/ca_bundle ]; then \
        export SSL_CERT_FILE=/run/secrets/ca_bundle CURL_CA_BUNDLE=/run/secrets/ca_bundle; \
    fi \
    && echo "options(repos = c(CRAN = '${CRAN_SNAPSHOT}'))" >> /usr/local/lib/R/etc/Rprofile.site \
    && Rscript -e "options(Ncpus = 4); install.packages(c('jsonlite', 'lmerTest', 'SensMixed'))" \
    && Rscript -e 'for (p in c("jsonlite", "lme4", "lmerTest", "SensMixed")) cat(p, format(packageVersion(p)), "\n")'
WORKDIR /w
