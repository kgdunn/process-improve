- **`lm(...).get_parameters()` is annotated and documented as returning a pandas `Series`**, indexed by term,
  which it always returned; the annotation said `DataFrame`, so a type checker accepted DataFrame-only calls such
  as `.to_dict(orient="records")` that fail at run time.
