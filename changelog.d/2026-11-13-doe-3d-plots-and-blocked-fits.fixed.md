- **DOE plots: the 3D plots name their axes, a cube plot is a cube, and a blocked fit can be drawn.**
  `surface_3d` and `cube_plot` now title their axes in Plotly's `layout.scene`, the response axis
  included; the titles were written to `layout.xaxis` and `layout.yaxis`, which a 3D trace does not
  read. A cube plot draws its twelve edges as lines, in a grey that shows on a white page, and its
  corners as points with their values: a line used to zigzag through the eight corners in the order
  they were listed. A corner's value is written to four significant figures, not two decimals. A
  design-quality plot (`prediction_variance`, `fds_plot`, `power_curve`) given a blocked fit leaves
  the fit's block contrasts out of its model and is drawn for the factors' terms, as
  `evaluate_design` scores a design whose model leaves its blocks out; it used to stop with
  `formula references unknown name(s) ['Block1']`. (#1229)
