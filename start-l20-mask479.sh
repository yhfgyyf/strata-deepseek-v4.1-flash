#!/usr/bin/env bash
set -euo pipefail

# Dual L20, cumulative mask479; Ctrl+C shuts down this foreground server.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -eq 0 ]; then
  echo "Usage: $0 --config /path/to/local-l20.json [--check-config]" >&2
  exit 2
fi

export CUDA_DEVICE_DEFAULT_STACK_SIZE_MAX=0
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export STRATA_DS41_ADAPTIVE_POLICY=maya
export STRATA_DS41_BLOCK_SEGMENTS=40
export STRATA_DS41_BLOCK_HUGE_MIN_PERCENT=100
export STRATA_DS41_DECODE_SEGMENTS=2
export STRATA_DS41_PIPELINE_PEER=1
export STRATA_DS41_NO_HUGE=1
export STRATA_DS41_PIPELINE_SERIAL=0
export STRATA_DS41_PIPELINE_CONTIGUOUS=0
export STRATA_DS41_PIPELINE_P2P=1
export STRATA_DS41_DECODE_CPU_MISSES=0
export STRATA_L20_Q5_N6=1
export STRATA_L20_N1_WARP=1
export STRATA_DS41_HC_PROJECT_REUSE=1
export STRATA_DS41_MOE_INPUT_REUSE=1
export STRATA_DS41_SHARED_Q8_REUSE=1
export STRATA_DS41_SM89_GROUPED_MOE=0
export STRATA_Q5_OUTPUT_A_BATCH=1
# Keep the validated mask479 kernels; the single-GPU decode options are not validated on L20.
export STRATA_DS41_OPT_MASK=0

unset STRATA_GROUPED_TRACE_DIR STRATA_L20_PROBE_DIR STRATA_L20_REFERENCE_DIR STRATA_L20_TRACE
exec "${STRATA_PYTHON:-python3}" "$ROOT/tools/deepseek41/serve_experimental.py" "$@"
