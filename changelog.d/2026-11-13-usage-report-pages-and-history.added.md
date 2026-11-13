- **factori.al's usage report counts each page, by day, week or month, and keeps the counts.**
  `factorial-usage` gives each page a column (the app, the concept map, the design catalog's
  pages, the multivariate page, About, and any other page), groups the rows by day, week or month
  (`--by`), and draws a line of bars for each page over the period. A daily job that `setup.sh`
  installs keeps each finished day's counts, numbers alone, in `/var/lib/factorial/usage.csv`, so
  usage can be followed after the 90 days the access log is kept. The About page says so.
