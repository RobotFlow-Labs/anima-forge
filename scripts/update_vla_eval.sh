#!/bin/bash
# Update vla-evaluation-harness to latest version
set -e

REPO_DIR="repositories/vla-evaluation-harness"

if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning vla-evaluation-harness..."
    mkdir -p repositories
    git clone https://github.com/allenai/vla-evaluation-harness.git "$REPO_DIR"
else
    echo "Updating vla-evaluation-harness..."
    cd "$REPO_DIR"
    git pull origin main
    cd -
fi

echo "vla-eval at $(cd "$REPO_DIR" && git rev-parse --short HEAD)"
