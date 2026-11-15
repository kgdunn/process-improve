- A build of Factorial that retires its service worker (`--no-service-worker`) now leaves nothing behind in the
  visit that retires it. In Chromium, a look the worker had begun in the copies as the page deleted them could bring
  one of them back, empty, some tens of milliseconds later, and it stayed until the next visit. The page now
  removes the copies once more when Python has started, or has failed to, by which time no such look is left
  ([#1248](https://github.com/kgdunn/process-improve/issues/1248)).
