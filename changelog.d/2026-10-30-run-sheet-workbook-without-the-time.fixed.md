- **The `write_run_sheet` workbook no longer holds the time it was written.** With
  `file_format="xlsx"`, openpyxl wrote the time of the call into the workbook's created and modified
  dates and into the date of every member of its zip archive, so two calls a few seconds apart gave
  different `xlsx_base64`. Each of those dates is now 1 January 1980, so the same runs give the same
  bytes; the cells are as before.
