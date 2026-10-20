- **`analyze_omars` finishes in seconds on designs with many factors.** Its second-order subset
  search fitted every subset of each size, and a 10-factor design with no heredity has 29 million
  subsets of six terms, which took hours. The search stays exhaustive at every size with at most
  200,000 subsets, so its results there are unchanged. Past that it continues with a beam search
  (every one-term extension of the 100 best subsets of the size before), refined by sequential
  replacement. `details["subset_search"]` says which sizes were searched exhaustively.
