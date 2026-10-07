#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if (( $# < 2 )); then
  echo "Usage: bash dist_test.sh CONFIG CHECKPOINT [extra train.py arguments]" >&2
  exit 2
fi
config="$1"
checkpoint="$2"
shift 2
exec torchrun --nproc_per_node="${NPROC_PER_NODE:-8}" \
  --master_port="${MASTER_PORT:-7775}" train.py --config "$config" \
  --test-only --resume "$checkpoint" "$@"
