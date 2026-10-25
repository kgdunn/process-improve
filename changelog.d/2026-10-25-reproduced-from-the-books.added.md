- **Reproduced from the books: a page of the design catalog at https://factori.al/catalog/reproduced/.** It shows,
  for every case of every source in the reference catalogue, how closely process-improve reproduces the numbers
  the source prints, in four tiers: fully reproduced (every recomputed number agrees to the source's last printed
  digit), within 1%, within 5%, and unable to reproduce. A table counts the cases and the numbers in each tier. A
  table by source groups the sources by their form (books, journal papers, and handbooks, encyclopedias and
  courses), with each group's totals and the share of each source's printed numbers that agree. A chart shows how
  far off the differing numbers are. Each tier opens as a list of its cases, grouped by source, and each case lists
  the numbers that differ with the printed value, the library's value, the relative difference, where the source
  prints it and the reason the catalogue records. The page can be searched (a title, a quantity, a word of the
  reasons such as "misprint"), filtered by source, form of source and kind of case, and its address keeps the
  view, down to one case.
- **`design_catalog.catalog_reproduction`** returns that classification for every case, or for one source, and
  `REPRODUCTION_TIERS` holds each tier's largest relative difference. **`design_catalog.source_form`** reads the
  form of a source from what the catalogue records of it (`SOURCE_FORMS`). The browser app's build writes both to
  `catalog/reproduction.json`.
