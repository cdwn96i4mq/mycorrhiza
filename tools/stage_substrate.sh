#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-data}"
mkdir -p "${DEST}"
echo "fetch the sources listed in cultures.txt into ${DEST}, then run prepare with the shared-gene set"
grep -E 'https?://' ../cultures.txt || true
