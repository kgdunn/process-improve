- **A component that `MBPLS` or `PLS` fits after Y is exhausted leaves the predictions as they were.** From
  the component that the `SpecificationWarning` "from component k on there is nothing left in Y that X can
  explain" names, each component takes the largest direction left in X, and its Y-loading is now 0. It was
  Y's regression on the component's score: about zero while X had variance left, but once every column of X
  was spent the score was round-off, and the regression the ratio of two round-off errors. So an `MBPLS` with
  more components than its blocks have columns moved its `predictions_`, `predict` and R²Y: on 30 rows of
  four columns in two blocks, six components moved the predictions by 0.016 from those of four, with
  Y-loadings of about 1e13 for the last two. `PLS` refuses a component once X has no variance left, so its
  predictions moved by much less: 7.5e-10 on the designed experiment in its tests. (#1074)
