- **`reshape_to_long` refuses two labels that differ only in surrounding whitespace (#1003).** It strips each
  label, so `"P1"` and `"P1 "` (a space typed into a spreadsheet) became one panelist: silently when their mean
  scores agreed, and otherwise through a round-trip failure that blamed the column mapping. A `ValueError` now
  names the two labels, for panelists, products and attributes alike.
