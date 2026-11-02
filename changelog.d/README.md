# Changelog fragments

A pull request with a user-facing change adds its changelog entry here, as a file of its own, and
does not edit `CHANGELOG.md`. Two open pull requests then never touch the same lines, so they do
not conflict with each other on the changelog.

Name the file `<yyyy-mm-dd>-<slug>.<type>.md`:

- `<yyyy-mm-dd>` is the day you write the fragment, such as `2026-10-18`. The fragments then list in
  the order they were written, and at release each heading files them in that order.
- `<slug>` is any short name in lower case, digits and hyphens, such as the branch name.
- `<type>` is the heading the entry belongs under: `added` for new public surface, `removed` for a
  removal, otherwise `changed`, `fixed`, `deprecated`, `security` or `documentation`. The type
  picks the release's version level, as the heading does (see "Choosing the level at release" in
  [`CONTRIBUTING.md`](../CONTRIBUTING.md#choosing-the-level-at-release)).

The file holds the entry as one or more Markdown bullets, written as they will read in the
changelog, with no heading. For example, `changelog.d/2026-10-18-glm-analysis.added.md`:

```markdown
- **Generalized linear models in `analyze_experiment`.** `analysis_type="glm"` fits a count, a
  number of successes out of `trials`, or a positive amount whose spread grows with its size.
```

`python tools/changelog.py check` says what is wrong with a fragment; the test suite runs it on
every pull request. At release, `python tools/changelog.py build` files every fragment under its
heading in `## [Unreleased]` and deletes it.
