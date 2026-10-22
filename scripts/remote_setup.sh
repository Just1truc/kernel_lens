#!/usr/bin/env bash
#
# Provision a remote GPU host for the KernelLens evaluation.
#
#   bash scripts/remote_setup.sh
#
# Targets a CUDA 12.x/13.x image (RunPod, Vast.ai, Thunder Compute). Idempotent:
# safe to re-run. Override with REPO_URL / WORKDIR / KERNEL_LENS_CACHE_DIR.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Just1truc/kernel_lens.git}"
WORKDIR="${WORKDIR:-$HOME/kl}"
export KERNEL_LENS_CACHE_DIR="${KERNEL_LENS_CACHE_DIR:-$HOME/.kernel_lens_cache}"

echo "== layout =="
echo "  repo      : $WORKDIR"
echo "  cache     : $KERNEL_LENS_CACHE_DIR"

# --- nvcc ------------------------------------------------------------------
# Needed to compile the generated plugins. CUDA images normally ship it; if the
# toolkit is installed but not on PATH, find it.
if ! command -v nvcc >/dev/null 2>&1; then
  for d in /usr/local/cuda*/bin; do
    [ -x "$d/nvcc" ] && export PATH="$d:$PATH" && break
  done
fi
if command -v nvcc >/dev/null 2>&1; then
  echo "== nvcc =="; nvcc --version | tail -2
else
  echo "!! nvcc not found. Install a CUDA toolkit (cuda-nvcc-* + cuda-cudart-dev-*)"
  echo "   or use an image that has one before continuing." >&2
fi
python3 -c "import ctypes.util, os, sys; \
p=os.environ.get('PATH',''); \
print('   cuda headers:', 'found' if any(os.path.exists(os.path.join(d,'..','include','cuda_runtime_api.h')) for d in p.split(':') if d) else 'not on PATH')" || true

# --- python dependencies ---------------------------------------------------
# TensorRT is pinned below 11: the plugin generator emits the
# IPluginV2DynamicExt interface that 10.x uses, and the package refuses 11+.
echo "== python deps =="
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet onnx onnxruntime-gpu "tensorrt>=10,<11" pytest
python3 -m pip install --quiet transformers accelerate

python3 - <<'PY'
import importlib, sys
for m in ("torch", "triton", "onnx", "onnxruntime", "tensorrt"):
    try:
        mod = importlib.import_module(m)
        print(f"   {m:12} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"   {m:12} MISSING ({e})")
sys.exit(0)
PY

# --- the package -----------------------------------------------------------
echo "== package =="
if [ -d "$WORKDIR/.git" ]; then
  git -C "$WORKDIR" fetch --quiet origin && git -C "$WORKDIR" reset --hard --quiet origin/main
else
  git clone --quiet "$REPO_URL" "$WORKDIR"
fi
python3 -m pip install --quiet -e "$WORKDIR"
python3 -c "import kernel_lens; print('   kernel_lens', kernel_lens.__version__)"

# --- sanity ----------------------------------------------------------------
echo "== gpu =="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

echo
echo "Ready. Run the matrix with:"
echo "  KERNEL_LENS_CACHE_DIR=$KERNEL_LENS_CACHE_DIR bash $WORKDIR/scripts/remote_bench_all.sh"
echo
echo "Keep the cache on a persistent volume: it holds the built ONNX graphs and"
echo "TensorRT engines, which dominate the runtime."
