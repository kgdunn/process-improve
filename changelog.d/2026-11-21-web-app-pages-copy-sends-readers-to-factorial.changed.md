- **The browser app's copy on GitHub Pages sends a reader on to https://factori.al/** (#1237). A page at
  `kgdunn.github.io/process-improve/app/` goes to the same page at factori.al, with its query and its `#` part,
  where the browser keeps no study at that address. Studies are kept in a browser's storage for one address and do
  not move with a redirect, so a browser that keeps some stays, and the page says how many, and how to take them
  along: Your studies, Download all, then drop that file into Your studies at factori.al. The copy's cached scripts
  could run with the new Python for ten minutes after a deploy; factori.al's cannot. The weekly smoke test and the
  check after a deploy now test factori.al only.
