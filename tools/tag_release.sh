#!/usr/bin/env bash
# Tag a release that is already on main, and push the tag, which starts publish.yml.
#
#   cd ~/process-improve
#   DRY_RUN=1 tools/tag_release.sh 2.6.0  # check only: nothing is tagged or pushed
#   tools/tag_release.sh 2.6.0            # check, tag, push, then watch the publish run
#
# The tag goes on the commit of main that first carried the version (the merge of the release pull request), so
# what was merged after it is not in the release. Every check prints its result, and the script stops at the
# first one that fails, saying which. Run it as a script: it never closes the shell it was started from.
set -u

# The script works on the clone it is in, wherever it is started from.
cd "$(dirname "$0")/.." || { echo "STOP: cannot go to the repository of this script" >&2; exit 1; }

VERSION="${1:-}"
if [ -z "$VERSION" ]; then
  echo "Usage: tools/tag_release.sh <version>, for example 2.6.0" >&2
  exit 2
fi
TAG="v$VERSION"

step() { printf '%-58s' "$1"; }
ok() { echo "ok${1:+: $1}"; }
stop() {
  echo "FAILED"
  echo "STOP: $1" >&2
  exit 1
}

step "Fetching main and the tags"
git fetch --quiet origin main --tags || stop "git fetch failed"
ok

step "Finding the commit of main that first carried $VERSION"
SHA="$(git log --first-parent --reverse --format=%H -S"version = \"$VERSION\"" origin/main -- pyproject.toml | head -n 1)"
[ -n "$SHA" ] || stop "no commit of main sets version = \"$VERSION\" in pyproject.toml: is the release pull request merged?"
ok "$(git log -1 --format='%h %s' "$SHA")"

# Each file is read whole into a variable and searched there. A pipe into `grep -q` would close early on a long
# file, and with `pipefail` that reads as a failure although the text was found.
has() { # has <file> <pattern>: whether the file, as that commit holds it, has a line matching the pattern
  local text
  text="$(git show "$SHA:$1")" || return 2
  grep -q -- "$2" <<<"$text"
}

step "pyproject.toml holds version $VERSION"
has pyproject.toml "^version = \"$VERSION\"\$" || stop "pyproject.toml at $SHA is not at $VERSION"
ok

step "CITATION.cff holds version $VERSION"
has CITATION.cff "^version: $VERSION\$" || stop "CITATION.cff at $SHA is not at $VERSION"
ok

step "CHANGELOG.md has the heading [$VERSION]"
has CHANGELOG.md "^## \[$VERSION\]" || stop "CHANGELOG.md at $SHA has no '## [$VERSION]' heading"
ok

step "The tag $TAG does not exist here"
if git rev-parse --quiet --verify "refs/tags/$TAG" >/dev/null; then
  stop "the tag $TAG exists in this clone already (git tag -d $TAG removes a local one that was never pushed)"
fi
ok

step "The tag $TAG does not exist on GitHub"
if git ls-remote --exit-code --tags origin "$TAG" >/dev/null 2>&1; then
  stop "the tag $TAG is on GitHub already: the release was tagged before"
fi
ok

if [ -n "${DRY_RUN:-}" ]; then
  echo "Every check passed. DRY_RUN is set: nothing was tagged or pushed."
  exit 0
fi

step "Tagging $SHA as $TAG"
git tag -a "$TAG" "$SHA" -m "$TAG" || stop "git tag failed"
ok

step "Pushing the tag"
git push --quiet origin "$TAG" || stop "git push failed: the tag is made here but not on GitHub (git tag -d $TAG undoes it)"
ok

echo "The tag is pushed, and publish.yml has started."
if command -v gh >/dev/null 2>&1; then
  sleep 10
  RUN="$(gh run list --workflow publish.yml --limit 1 --json databaseId --jq '.[0].databaseId' 2>/dev/null)"
  if [ -n "$RUN" ]; then
    echo "Watching run $RUN (Ctrl-C stops the watching, not the run):"
    gh run watch "$RUN" --exit-status || stop "the publish run did not succeed: gh run view $RUN --log-failed"
  fi
else
  echo "Follow it at https://github.com/kgdunn/process-improve/actions/workflows/publish.yml"
fi
echo "Then check PyPI: https://pypi.org/project/process-improve/$VERSION/"
