- **A copy of factori.al that is removed is less able to come back.** After a build retired the
  worker, one of the copy's caches was once found still there (#1248). The worker looked for each
  kept file in its cache by the cache's name, which in Chromium makes the cache: a look made as
  the copy went brought its name back, for the worker to delete again. It now looks in every
  cache at once, which makes none, so nothing is left if the worker is stopped first. And a
  reader who removed the copy has any cache that came back deleted at their next visit.
