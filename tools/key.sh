#!/usr/bin/env bash
set -euo pipefail
RECIPE="${1:-conf/symbiont.gin}"
python -m mycorrhiza.forager key -c "${RECIPE}"
python -m mycorrhiza.forager forage -c "${RECIPE}"
