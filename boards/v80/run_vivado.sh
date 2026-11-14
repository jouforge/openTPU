#!/usr/bin/env bash
# Standalone batch entry; the external AMR trees are read-only inputs.
set -euo pipefail
BOARD_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$BOARD_DIR/../.." && pwd)
STEP=${1:-preflight}
case "$STEP" in
    prepare|preflight|project|generate|elaborate|synth|impl|pdi) ;;
    *) echo "usage: $0 [prepare|preflight|project|generate|elaborate|synth|impl|pdi]" >&2; exit 2 ;;
esac
PYTHON=${PYTHON:-python3}
VIVADO=${VIVADO:-vivado}
OUT_DIR=${OUT_DIR:-$ROOT/build/vivado_v80/platform}
# Reject dependency paths before even creating the output directory or lock file.
: "${AMR_ROOT:?set AMR_ROOT to the AMR checkout}"
: "${AMR_DESIGN_ROOT:?set AMR_DESIGN_ROOT to the designs checkout}"
OUT_DIR=$(realpath -m "$OUT_DIR")
case "$OUT_DIR/" in
    "$(realpath "$AMR_ROOT")/"*|"$(realpath "$AMR_DESIGN_ROOT")/"*)
        echo "OUT_DIR must be outside the external platform checkouts" >&2; exit 1 ;;
esac
mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd)
exec 9>"$OUT_DIR/.flow.lock"
flock -n 9 || { echo "V80 flow already uses $OUT_DIR" >&2; exit 1; }
VIVADO=$(command -v "$VIVADO")
"$PYTHON" "$BOARD_DIR/scripts/prepare.py" --out-dir "$OUT_DIR" --target "${TARGET:-v80}" --vivado "$VIVADO"
if [[ "$STEP" == prepare ]]; then exit 0; fi
mkdir -p "$OUT_DIR/reports"
cd "$OUT_DIR"
"$VIVADO" -mode batch -notrace -log "$OUT_DIR/reports/$STEP.log" \
    -journal "$OUT_DIR/reports/$STEP.jou" -source "$BOARD_DIR/vivado/run.tcl" \
    -tclargs "$OUT_DIR/config.tcl" "$STEP"
