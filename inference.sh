#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if (( $# < 3 )); then
  echo "Usage: bash inference.sh CONFIG CHECKPOINT IMAGE_OR_VIDEO [extra inference arguments]" >&2
  exit 2
fi
config="$1"
checkpoint="$2"
input="$3"
shift 3
exec python tools/inference/torch_inf.py --config "$config" --resume "$checkpoint" \
  --input "$input" "$@"
