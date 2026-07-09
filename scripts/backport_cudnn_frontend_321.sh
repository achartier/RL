#!/bin/bash
# Backport cuDNN-frontend PR #321 ("Migrate cute.core.ThrMma / cute.make_fragment") onto the
# installed cuDNN-frontend 1.25.0 so its cutedsl DSA kernels run against cutlass-dsl 4.6.0.
# PR #321 (commit 74efc0d, merged to develop 2026-06-24) is NOT in any release; 1.25.0 is the
# latest on PyPI and its [cutedsl] extra pins cutlass-dsl==4.5.0. Both migrated symbols also exist
# at cutlass.cute top-level in 4.5.0, so the patched frontend runs on BOTH 4.5.0 and 4.6.0.
# Run once against the mcore venv AFTER `uv sync` (idempotent).
# Usage: backport_cudnn_frontend_321.sh /path/to/mcore-venv
set -euo pipefail
VENV="${1:?usage: $0 <mcore-venv-dir>}"
SP="$VENV/lib/python3.13/site-packages"
[ -d "$SP/cudnn" ] || { echo "no cudnn in $SP"; exit 1; }
FILES=$(grep -rln 'cute\.core\.ThrMma\|cute\.make_fragment(' "$SP"/cudnn/ 2>/dev/null || true)
if [ -z "$FILES" ]; then echo "already patched (no old symbols)"; exit 0; fi
echo "patching:"; echo "$FILES" | sed "s#$SP/##"
for f in $FILES; do
  sed -i 's/cute\.core\.ThrMma/cute.ThrMma/g; s/cute\.make_fragment(/cute.make_rmem_tensor(/g' "$f"
done
REM=$(grep -rl 'cute\.core\.ThrMma\|cute\.make_fragment(' "$SP"/cudnn/ 2>/dev/null | wc -l)
echo "#321 backport applied; old symbols remaining: $REM"
