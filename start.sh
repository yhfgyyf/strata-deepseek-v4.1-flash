#!/usr/bin/env bash
set -euo pipefail
strata_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -eq 0 ]; then
  set -- gui
fi
exec "${STRATA_PYTHON:-python3}" "$strata_root/tools/deepseek41/launcher.py" "$@"
