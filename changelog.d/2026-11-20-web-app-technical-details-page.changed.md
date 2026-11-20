- **The browser app's page on how Factorial works has a name that says what it holds**:
  `technical-details-how-factorial-works.html`, where it was `about.html`, with its own canonical address and share
  card. The old address forwards to it: the server answers it with a permanent redirect that keeps a link's query
  (`deploy/factorial/factorial.caddy`), and a small page at the old address forwards wherever the server has none.
