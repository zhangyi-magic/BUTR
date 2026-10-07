#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

config="${1:-configs/models/BUTR-L-AITOD.yml}"
if (( $# )); then shift; fi
exec torchrun --nproc_per_node="${NPROC_PER_NODE:-8}" \
  --master_port="${MASTER_PORT:-7789}" train.py --config "$config" --seed 0 "$@"
