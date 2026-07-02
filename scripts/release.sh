#!/bin/bash -e
# Set the release version everywhere, commit, and tag vX.Y.Z (does not push).
# Usage: scripts/release.sh X.Y.Z

cd "$(dirname "$0")/.."

VER="$1"
[[ "$VER" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "usage: $0 X.Y.Z" >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "error: work tree not clean" >&2; exit 1; }
git rev-parse "v$VER" >/dev/null 2>&1 && { echo "error: tag v$VER already exists" >&2; exit 1; }

echo "$VER" > version.txt
sed -i.bak "s/^__version__ = \".*\"/__version__ = \"$VER\"/" python/src/pvfmm/__init__.py
sed -i.bak "s/^version = \".*\"/version = \"$VER\"/" julia/Project.toml
rm -f python/src/pvfmm/__init__.py.bak julia/Project.toml.bak

git commit -am "Release v$VER"
git tag -a "v$VER" -m "PVFMM v$VER"
echo "Tagged v$VER. Push with: git push origin HEAD v$VER"
