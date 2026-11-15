- **`factorial-usage` counts the visits a tagged link brought.** A link posted somewhere can name the
  place, as `https://factori.al/?ref=linkedin`, and the report lists its visits as `?ref=linkedin`
  under "Where readers came from", apart from the rest. That counts a place whose app opens links in
  a browser of its own, which names no site. A tag is up to 24 letters, digits, `-` or `_`, read in
  lower case; any other is not counted. It is a word for a place, never for a person. The app drops
  the tag from its address once open, so a reload, or the address copied on, is not counted again.
  The access log already holds each request's address, so visits made before this are counted too.
  Run `setup.sh` on the server to install the new report.
