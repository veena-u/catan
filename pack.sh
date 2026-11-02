#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
commit=$(git rev-parse HEAD)
[ -z "$(git status --porcelain)" ] || commit="$commit-uncommitted"
echo "$commit" > COMMIT
COPYFILE_DISABLE=1 tar czf /tmp/catan.tgz --exclude=./target --exclude=./runs --exclude=./gpu-results \
  --exclude=./baseline.tar.gz --exclude=./.git --exclude='*/__pycache__' .
rm COMMIT
echo "packed /tmp/catan.tgz from $commit"
