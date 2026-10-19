- **Run sheets out to the lab and back.** `run_sheet(design, factors, units=..., labels=...,
  responses=...)` writes the runs as a table to save as CSV or Excel: in actual or coded units,
  under the factors' names, `X01`..`Xnn` or labels of your own, with empty response columns.
  `read_run_sheet(source, factors, labels=...)` reads a filled sheet (a CSV or Excel file, the
  bytes of a workbook, or a DataFrame) back into the factors' own names and actual units. It
  tells coded from actual units, matches columns without regard to case or leading zeros, and
  warns about settings outside the ranges. Both are MCP tools, `write_run_sheet` and
  `read_run_sheet`, which exchange the sheet as CSV text or a base64 workbook.
