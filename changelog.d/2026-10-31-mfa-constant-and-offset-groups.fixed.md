- **`MFA` judges a group's columns by their values, not by their size.** A group whose values sat
  far from zero relative to their spread (lab results 1e7 above zero with an sd of 1, or times in
  seconds since 1970 that vary by minutes) was refused for having "no variation", as its first
  eigenvalue was compared with the mean square of the raw values; the offset is centred away
  before the group is weighted, so it now fits as it does near zero. And with `scale=True` a
  constant column was refused only when its standard deviation came out exactly 0; 0.1 in twelve
  rows gives 1.4e-17, and the column joined its group as one of -1s, uncentred. A column is now
  constant when all its values are equal, as `MCUVScaler` reads one, and a group has no variation
  when every column in it is constant. (#1029)
