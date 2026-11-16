- **factori.al: Python starts when the CDN refuses one of its packages at first.** Pyodide only logs a package
  whose download failed, and the page then stopped with "Could not start Python" at the first import of what was
  missing. The start now checks that every package it named arrived, with all each depends on, and asks again
  for those that did not: three times, after 1, 3 and 9 seconds, saying so in the status line. What did arrive
  is kept, so each try asks for less. Packages loaded later, for fitting and for charts, were asked for again
  already.
