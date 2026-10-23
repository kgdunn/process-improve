- **`nan_to_zeros` returns a new array and leaves its argument alone (#513).** It wrote the zeros
  into the array it was given and returned that same array, so a view of the caller's data was
  overwritten. A caller that relied on the argument being changed must now use the return value.
