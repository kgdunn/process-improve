- **`MBPCA` and `MBPLS` run their first pass whatever the tolerance.** The loop tested the
  tolerance before its first pass, and a centred seed changes by about its own size in that pass,
  so a `tol` above about 1 ended the loop before it began, and `fit` raised `KeyError`. The first
  pass now always runs. The dense path, which iterates as the masked path does, also checks `tol`
  as that path does: a `tol` of 10 or more, or one at machine precision or below, is refused with a
  `ValueError` for `algorithm="dense"` too. Such a small `tol` used to run to `max_iter` and warn.
  (#1028)
