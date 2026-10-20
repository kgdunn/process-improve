- **Factorial finds a mixture's best blend, and balances its responses over the blends.** For a
  mixture with a goal, Learn searches the region of blends for the one the fitted Scheffé model
  predicts best (`optimize_responses` with a mixture `DesignRegion`: each component between its
  bounds, the proportions summing to 1), where it used to say the blends were not searched. It shows
  each component's proportion and whether it sits at a bound (and whether the goal improves past a
  bound you set), the prediction with its 95% confidence and prediction intervals, the leverage,
  and how much the prediction moves when each component changes along its Cox direction.
  Confirmation runs go at the blend weighed out to a thousandth, still summing to 1, with the
  component that takes the rounding named. A star marks the blend on the triangle of three
  components; with four or more, the Cox traces are drawn through the best blend and cross at the
  star. With two responses or more that have goals, the balance of the responses is found over the
  blends too, each response with its own Scheffé model, and for three components the overlay of the
  acceptable ranges is drawn on the triangle. The Python panels search the same region and find the
  same blends.
