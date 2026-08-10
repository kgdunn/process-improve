# Example inputs

The files behind `references/worked-example.md`. Run them in order from the
`skills/doe-designer/` directory.

| File | What it is |
|---|---|
| `strategy_spec.json` | Input to `recommend_strategy`: 7 factors, one response, 40-run budget |
| `design_spec.json` | Input to `generate_design`: the same factors, resolution IV, 16 runs |
| `screen_coded.csv` | The generated design in coded units, ready for `verify_design.py` |
| `screen_results.csv` | The same design with a simulated Yield column, ready to analyse |

```bash
# 1. Plan
python scripts/doe_tool.py call recommend_strategy --input examples/strategy_spec.json

# 2. Generate
python scripts/doe_tool.py call generate_design --input examples/design_spec.json --output design.json

# 3. Verify (this is the step that must not be skipped)
python scripts/verify_design.py examples/screen_coded.csv --require-resolution 4

# 4. Compare two candidates and rank them
python scripts/verify_design.py examples/screen_coded.csv other.csv --compare

# 5. Analyse
python - <<'PY'
import json, pandas as pd
frame = pd.read_csv("examples/screen_results.csv")
json.dump({"design_matrix": frame.to_dict("records"), "response_column": "Yield",
           "model": "interactions",
           "analysis_type": ["effects", "lenth_method", "residual_diagnostics"]},
          open("analysis_spec.json", "w"))
PY
python scripts/doe_tool.py call analyze_experiment --input analysis_spec.json --output analysis.json

# 6. Plot
python scripts/render_plot.py --analysis analysis.json --type half_normal --output half_normal.png
```

The Yield column in `screen_results.csv` is simulated, from a process where
only Temperature and Catalyst matter (plus a small interaction between them).
That is what makes it useful as a teaching case: you already know the answer,
so you can see whether the analysis recovers it, and you can watch the
resolution IV aliasing produce three identical interaction estimates.
