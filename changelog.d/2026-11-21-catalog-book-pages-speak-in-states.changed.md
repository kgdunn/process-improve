- **Factorial: the book pages and the lists of cases speak in the reproduction page's words** (#1356). A
  source's page under `catalog/books/` draws every number it prints as the bar of its row on
  [Reproduced from the books](https://factori.al/catalog/reproduced/), in the same colors, and names each
  example "Matches the book", "Differs, and we know why", "Multiple choice, wrong on purpose" or "Differs, reason
  not known yet", where it said Matches, Very close, Close or Differs. An example takes the last of these that one of
  its numbers is under, so a misprint no longer reads as a match, and a reason not known yet is never hidden behind
  a known one. Its button, which opens the source's cases on the reproduction page, says so: "Search the book's
  cases", where it said "Show the numbers". Behind "Search every case" the cases are listed by the same four names,
  not by tier, and a source's row counts the cases that match in every number the same way. A printed number is
  named in plain words, such as "a sum of squares" or "an F ratio", where it was named by the catalog's own word
  (`anova sum sq`, `f regression`). `catalog_reproduction` still returns each case's tier.
