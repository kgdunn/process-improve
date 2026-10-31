- **`FAMD` refuses a constant numeric column whatever its standard deviation rounds to.** It
  refused one only when its population standard deviation came out exactly 0, which depends on
  the value and the number of rows: 0.1 in twelve rows gives 1.4e-17. Divided by that, the column
  became one of -1s, uncentred, which added an inertia of 1 to the total and an axis of its own,
  and cut every share. A column is now constant when all its values are equal, as `MCUVScaler`
  reads one, in any units. (#1029)
