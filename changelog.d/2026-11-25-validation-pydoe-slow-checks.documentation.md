- **pydoe's 14 slowest checks are counted.** pydoe's `fracfact_opt` scores every generator set, up to 4.9 hours
  for 11 factors in 128 runs, so 14 checks of 10 and 11 factors had not finished in 3 minutes. Each search was run
  once, without that limit, and all 14 agree: pydoe 1.7.0 now has 3,176 checks, all as printed. The comparison
  script searches each size once and shares it between the checks of that size.
