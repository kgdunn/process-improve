- **A column or row at `2**1023` (about 9e307) or above keeps its scaling and its length.**
  `MCUVScaler` took a column whose largest absolute value is `2**1023` or more for a column with no
  spread, and left it unscaled with no warning. On LDPE with one column brought to 1.3e308, `PCA`
  on autoscaled data gave `inf` scores, `MBPCA` and `MBPLS` gave NaN, and `PLS` raised "no
  variance left". `PRM`'s row lengths, and the L1 median it centres X at, were NaN for a row that
  far out. Each divided by the power of two above the largest value, which there is `2**1024` and
  overflows. Now such a column is scaled as in any other units, and these models are those of the
  data in units of 1, to 1e-14; a row has its length: `[1e308, 0]` is 1e308 long, and
  `[1e308, 1e308]` is 1.41e308 long, where `numpy.linalg.norm` gives `inf`. A standard deviation
  or a length beyond the largest float, 1.8e308, is `inf` with numpy's overflow warning, as numpy
  gives it. Data below `2**1023` give the same numbers, bit for bit. (#1149)
