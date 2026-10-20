- **`to_pseudocomponents` and `from_pseudocomponents`: L-pseudocomponents of a mixture with lower
  bounds** (`process_improve.experiments`, #788 A10). x'_i = (x_i - L_i)/(T - sum L) maps the
  region the lower bounds leave back onto the whole simplex, and x_i = L_i + (T - sum L) x'_i maps
  it back, for proportions (T = 1) or percentages (T = 100); a DataFrame keeps its index and
  columns, with the bounds given by name. The pure first pseudocomponent of the pid-book's region
  (bounds 0.2, 0.2 and 0) is the blend (0.8, 0.2, 0.0), and Goos and Jones's rolling-mill runs
  convert as the reference catalogue codes them.
