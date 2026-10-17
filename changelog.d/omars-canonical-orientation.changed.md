- **`generate_omars` returns an exhaustively searched design in one fixed orientation.** For three and
  four factors at small run sizes the search enumerates every design and picks the best. Designs that differ
  only in factor order and factor signs tie exactly on every criterion, so rounding in the floating-point
  scores picked among them, and the design returned could differ between platforms. The winner is now
  returned in its canonical orientation, which is the same on every platform. It scores as before on every
  criterion, but its columns can be permuted and sign-flipped relative to 1.99.0.
