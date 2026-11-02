- **Plackett-Burman designs now have 92, 116, 156, 172, 184 and 188 runs where those are the
  smallest size.** No Hadamard matrix of these orders was built, so 88 to 91 factors gave a 96-run
  design with no warning, and likewise at the other five sizes. The orders now come from the
  Goethals-Seidel array of four circulant matrices (Williamson matrices for 92, 116, 156 and 172,
  Djokovic's rows for 188, and 184 as 92 doubled), and every multiple of 4 up to 232 is built.
  Supersaturated designs gain the same orders. Where a size is still missing (the first is 236), the
  design now raises a `UserWarning` that names the smallest size and the size built. (#967)
