- **The components of three-level interactions.** `interaction_components(data, response,
  factors)` splits each interaction of three-level factors into its components of two degrees of
  freedom, in arithmetic modulo 3, with Montgomery's names (`AB2C`) and Yates's (`X(ABC)`), and
  gives each component's sum of squares. It reproduces Davies's cotton dyestuff (Example 9-4:
  X = 6191, Z = 1002, W = 2939 from the second replicate and Y = 664 from the first) and
  Montgomery's syrup loss (W = 3804.11), and in a full factorial an interaction's components add
  up to its sum of squares.
