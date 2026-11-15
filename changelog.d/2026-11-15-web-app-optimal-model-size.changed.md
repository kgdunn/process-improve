- In Factorial the D-, I- and A-optimal designs are limited by the size of their model and not by a number of
  factors. They take 2 to 40 factors (D-optimal took 2 to 8, I- and A-optimal 2 to 6), for a model of up to 91
  coefficients, the quadratic model in 12 factors: any model in up to 12 factors, and the main effects in up to
  40. A larger model is refused, and the form does not offer it.
- A search that takes long is the reader's to start. Timed under Pyodide, a search takes longer the more
  coefficients its model has and the more factors there are, and no longer for more runs: 14 s at most for a
  D-optimal design, and up to a minute and a half for an I- or A-optimal design of 40 factors. When a search is
  expected to take 5 seconds or more, the design form says how long before the run sheet is made ("up to about
  1.5 minutes"), and the list of designs that fit the brief leaves that design out until the reader asks for
  the search, with the wait. The list used to hold every optimal design back from five factors for a model
  beyond main effects, a quick D-optimal search among them; `api_candidates` now gives each waiting design's
  `seconds`.
- With limits on the settings a study still has up to 8 factors, now as a limit of its own
  (`MAX_LIMITED_FACTORS`), and the I- and A-optimal designs are made for 7 and 8 of them.
