- **A ratio of components as a mixture constraint.** A mixture region now takes a ratio of two linear
  expressions against a number, such as `"A / B <= 2.5"`, `"0.5 <= A / (B + C) <= 2"` or `"A / B >= 0.4"`.
  It holds the limit that multiplying through by the denominator gives (`"A <= 2.5 * B"`), in
  `generate_design`, in a design of components beside process factors, and in `DesignRegion`, whose
  `linear_forms()` states that limit. The component bounds must keep the denominator above zero: where it can
  reach zero the constraint is refused with the limit to write in its place, because that limit also admits
  the blends where the ratio has no value. Such a constraint used to be refused as not linear.
