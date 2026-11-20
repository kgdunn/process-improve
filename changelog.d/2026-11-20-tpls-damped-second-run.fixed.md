- **TPLS converges on a component whose iteration swings** (#1163). The loop of passes is not a power iteration.
  Once D is deflated, the F block's score `t_f = F r / r'r` grows as `1/|r|` in some directions, the super weight
  swings between the blocks, and `u` can alternate between two states for ever. The component was then the state
  of the last pass, which differed between platforms. A component that stops at `max_iter` is now iterated again
  from its start, damped: each pass that turns `u` less in two passes than in one halves how far later passes move
  it. Damping keeps the fixed point, so the damped run converges to a solution of the same equations. A fit that
  converged before is unchanged; one that did not converge either way warns as before.
