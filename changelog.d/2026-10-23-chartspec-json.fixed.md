- **`ChartSpec.to_dict()` is JSON-safe, as documented (#513).** It converted enums only, so a layer's data
  built from a DataFrame kept its numpy scalars and `json.dumps` raised on an `np.int64`. It now converts
  them as every tool's output is converted (`tool_spec.clean`): numpy values become Python values, and
  NaN and infinity become `None`.
