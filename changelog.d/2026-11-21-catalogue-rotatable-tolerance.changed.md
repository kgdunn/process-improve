- **The catalog's `rotatable` check takes a tolerance** (#790): the spread allowed on each sphere, as a share of its
  mean, for a design printed rounded. Box and Draper's Exercise 12.12 is now checked: blocks 1 and 2 are rotatable
  for the second-order model, and all three blocks, whose heptagon is printed to three decimals (a spread of 6.1e-4
  of the mean), for the third-order one at a tolerance of 1e-3.
