- **factori.al's usage report counts each page, by day, week or month, and the server keeps a
  year of its log.** `factorial-usage` gives each page a column (the app, the concept map, the
  design catalog's pages, the multivariate page, About, and any other page), groups the rows by
  day, week or month (`--by`), and draws a line of bars for each page over the period. The access
  log is kept for 366 days, up from 90, so a year can be set beside the one before; a daily job
  that `setup.sh` installs keeps each finished day's counts, numbers alone, in
  `/var/lib/factorial/usage.csv` for longer still. The About page says both.
