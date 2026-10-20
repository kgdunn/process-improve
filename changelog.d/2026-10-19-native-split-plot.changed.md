- **`generate_design(..., hard_to_change=[...])` builds the split-plot design itself.** With
  `backend="auto"` (the default) or `"exchange"`, a split-plot design now comes from the built-in
  coordinate exchange (`metadata["backend"] == "split_plot_exchange"`), whether or not pyoptex
  is installed; `backend="pyoptex"` still asks for pyoptex. Before, the design came from pyoptex
  when it was installed, and without it `hard_to_change` was ignored with a warning. With
  constraints, a candidate set, fixed runs, blocks or the E-, G- and K-criteria, it is still
  ignored with a warning that names the reason. The whole-plot variance ratio stays at 0.5.
