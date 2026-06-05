#!/bin/bash
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo '{"async": true, "asyncTimeout": 300000}'

# Install Julia if not present
if ! command -v julia &> /dev/null; then
  echo "Installing Julia..."
  curl -fsSL https://install.julialang.org | sh -s -- --yes
  export PATH="$HOME/.juliaup/bin:$PATH"
fi

export PATH="$HOME/.juliaup/bin:$PATH"

# Instantiate the Julia project to install all dependencies
cd "${CLAUDE_PROJECT_DIR:-.}"
julia --project=. -e 'using Pkg; Pkg.instantiate()'