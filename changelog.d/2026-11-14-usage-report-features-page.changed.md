- **`factorial-usage` counts the page of what Factorial does in a column of its own.** A visit
  to `/features.html` was counted under Other; it now has the column Features, after About. The
  kept counts (`/var/lib/factorial/usage.csv`) need no change by hand: the file is written whole
  each day, and a day kept before the column has zero in it. Run `setup.sh` on the server to
  install the new report.
