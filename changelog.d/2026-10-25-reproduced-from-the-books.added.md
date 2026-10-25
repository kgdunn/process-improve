- **Reproduced from the books: a page of the design catalog at https://factori.al/catalog/reproduced/.** It shows,
  for every case of every source in the reference catalogue, how closely process-improve reproduces the numbers
  the source prints, in four tiers: fully reproduced (every recomputed number agrees to the source's last printed
  digit), within 1%, within 5%, and not reproduced. A table counts the cases and the numbers in each tier, a bar
  per source breaks its cases down by tier, and a chart shows how far off the differing numbers are. Each tier
  opens as a list of its cases, grouped by source, and each case lists the numbers that differ with the printed
  value, the library's value, the relative difference, where the source prints it and the reason the catalogue
  records. The page can be searched (a title, a quantity, a word of the reasons such as "misprint"), filtered by
  source and kind of case, and its address keeps the view, down to one case.
- **`design_catalog.catalog_reproduction`** returns that classification for every case, or for one source, and
  `REPRODUCTION_TIERS` holds each tier's largest relative difference. The browser app's build writes it to
  `catalog/reproduction.json`.
