#!/usr/bin/env bash
# Provision /workspace/deep-gan on tig-gpu from a git bundle pushed up from a
# local worktree. Never touches /workspace/wgan-synthetic, which is a shared
# checkout other agents may be using.
set -euo pipefail

REMOTE="${REMOTE:-tig-gpu}"
WORK_DIR="/workspace/deep-gan"
CACHE_DIR="/workspace/data-cache"
BUNDLE="/tmp/deep-gan.bundle"
# Errors here rather than three lines down: on a detached HEAD this is empty,
# and `git bundle create <file> ""` fails with a message about revision syntax
# that says nothing about the actual problem.
BRANCH="$(git branch --show-current)"
: "${BRANCH:?not on a branch (detached HEAD) -- check out the branch you want to push before running this}"

echo "==> bundling ${BRANCH}"
git bundle create "${BUNDLE}" "${BRANCH}"
scp -q "${BUNDLE}" "${REMOTE}:/tmp/deep-gan.bundle"

echo "==> unpacking on ${REMOTE}"
ssh "${REMOTE}" bash -s <<REMOTE_SCRIPT
set -euo pipefail
if [ -d "${WORK_DIR}/.git" ]; then
    cd "${WORK_DIR}"
    git checkout --detach
    git fetch /tmp/deep-gan.bundle "${BRANCH}:refs/heads/${BRANCH}" --force
    git checkout --force "${BRANCH}"
else
    git clone -b "${BRANCH}" /tmp/deep-gan.bundle "${WORK_DIR}"
fi
cd "${WORK_DIR}"
# Do NOT "pip install -r requirements.txt" here: /venv/main is a shared
# interpreter other agents on this box are actively using, and requirements.txt
# is unpinned (numpy, torch, plotly, ...) -- a resolver run against the whole
# file can silently upgrade a neighbour's torch mid-run. h5py is the only
# package this branch actually adds beyond what the box already has (it
# backs the HDF5 reader in src/data/fetch.py), so install just that one
# package and nothing else.
/venv/main/bin/pip install -q h5py
mkdir -p "${CACHE_DIR}"
REMOTE_SCRIPT

echo "==> fetching DEEP data (shared cache, downloads only once)"
ssh "${REMOTE}" "cd ${WORK_DIR} && /venv/main/bin/python -m src.data.fetch deep \
    --cache-dir ${CACHE_DIR} --out-dir data"

echo "==> done: ${REMOTE}:${WORK_DIR}"
