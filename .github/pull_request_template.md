## Summary

<!-- What does this PR change, and why? One to three bullet points. -->

## Test plan

<!-- How was this change verified? -->

- [ ]

## Checklist

- [ ] Changelog entry added as a fragment, `changelog.d/<slug>.<type>.md`, not in `CHANGELOG.md` itself, with `<type>` the heading it belongs under (`added` for new public surface, `removed` for a removal, otherwise `changed` / `fixed` / `deprecated` / `security` / `documentation`). The type picks the release's version level, so it matters. Internal-only changes may not need an entry at all.
- [ ] **No** version bump: `pyproject.toml` and `CITATION.cff` are set at release time, not in a PR
- [ ] Tests added or updated where relevant
- [ ] `ruff check .` passes
