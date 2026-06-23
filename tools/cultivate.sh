#!/usr/bin/env bash
set -euo pipefail
RECIPE="${1:-conf/symbiont.gin}"
EPOCHS="${2:-100}"
OUT="${3:-runs/symbiont.pt}"
torchrun --standalone --nproc_per_node=4 -m mycorrhiza.forager cultivate -c "${RECIPE}" -e "${EPOCHS}" -o "${OUT}"
