# Multiblock Audit (DRAFT)

This document audits the two MATLAB multiblock latent-variable repositories
(`kgdunn/multiblock-latent-variable-methods-matlab` and
`kgdunn/multiblock-latent-variable-methods-gsk`) and recommends what to port
into `process-improve` as Python.

## Status

**Stub.** Plan approved; full audit content to follow in subsequent commits to
this PR once the plan has been reviewed.

## Headline finding

- Both MATLAB repos are dormant (matlab: Jun 2012, gsk: Apr 2011).
- The matlab repo's `mbpca.m` was checked in mid-debug (last commit message:
  *"Something wrong with deflation in PCA?"*); the deflation loop contains a
  bare un-commented line of text that would fail to parse. The matlab MB-PCA
  is **not** the right port source as-is.
- The gsk repo's `test_mbpca.m` and `test_mbpls.m` contain working algorithms
  with explicit assertions against the Westerhuis equations, and are the
  recommended algorithm references.
- Best-of-both: take **architecture, batch class, and plotting** from the
  matlab repo; take **MB-PCA/MB-PLS algorithms and validation oracles** from
  the gsk repo; reuse the partial Python port `build_lvm.py` as a sanity check
  against the existing single-block PCA/PLS in
  `process_improve/multivariate/methods.py`.
