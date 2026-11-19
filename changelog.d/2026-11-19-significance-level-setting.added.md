- **Browser app (factori.al): a study's significance level.** "Settings" now chooses the significance level a study is
  judged at: 0.01, 0.05 (as before) or 0.10. The confidence of every interval is one minus it, so the page shows
  99%, 95% or 90% intervals. Every test, every interval, Lenth's margins of error and a design's power follow the
  level, the page's words name it, and the Python code shown passes it to the library. A study judged at 0.01 or
  0.10 is saved in study format 13, which a page from before the setting refuses, so no study is silently judged
  at 0.05.
