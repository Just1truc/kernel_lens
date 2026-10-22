#!/usr/bin/env bash
#
# Run the full evaluation matrix and collect JSON.
#
#   bash scripts/remote_bench_all.sh
#
# Results land in $OUT (default ./results). Individual failures are recorded and
# do not abort the run, so a partial matrix is still useful.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export KERNEL_LENS_CACHE_DIR="${KERNEL_LENS_CACHE_DIR:-$HOME/.kernel_lens_cache}"
OUT="${OUT:-$ROOT/results}"
mkdir -p "$OUT"

echo "=== environment ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true
python3 -c "import torch,triton,tensorrt,onnxruntime as o; \
print('torch',torch.__version__,'| triton',triton.__version__,'| TRT',tensorrt.__version__,'| ORT',o.__version__)" || true
df -h "$OUT" | tail -1

run() {  # run <logfile> <cmd...>
  local log="$1"; shift
  echo; echo "=== $* ==="
  "$@" >"$log" 2>&1
  local rc=$?
  tail -n 6 "$log"
  [ $rc -ne 0 ] && echo "   (exit $rc — see $log)"
  return 0
}

# --- correctness -----------------------------------------------------------
run "$OUT/conformance.log"  python3 tests/test_conformance.py
run "$OUT/parity.log"       python3 -m pytest tests/test_research_kernels.py -q -p no:warnings --no-header
run "$OUT/folding.log"      python3 tests/benchmark_folding_vs_triton.py
run "$OUT/e2e.log"          python3 tests/benchmark_e2e_llama.py

# --- real pretrained models ------------------------------------------------
# Both precisions: ONNX Runtime is strongest in fp32, TensorRT in fp16.
# Above roughly 1 GB of engine the host may refuse to deserialize; those rows
# simply come back empty rather than stopping the sweep.
for spec in \
  "facebook/opt-125m 512 float32" \
  "facebook/opt-125m 512 float16" \
  "facebook/opt-350m 512 float32" \
  "facebook/opt-350m 512 float16" \
  "facebook/opt-1.3b 512 float32" \
  "facebook/opt-1.3b 512 float16" \
  "facebook/opt-2.7b 512 float32" \
  "facebook/opt-2.7b 512 float16" \
  "facebook/opt-6.7b 512 float16" ; do
  set -- $spec
  tag="$(basename "$1")_$3"
  run "$OUT/real_${tag}.log" python3 tests/benchmark_real_model.py \
      --model "$1" --seq "$2" --dtype "$3" --backends onnx,tensorrt \
      --out "$OUT/real_${tag}.json"
done

# --- summary ---------------------------------------------------------------
echo; echo "=== summary ==="
python3 - "$OUT" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(out, "real_*.json"))):
    d = json.load(open(f))
    row = {"model": d.get("model"), "dtype": d.get("dtype"),
           "eager_ms": d.get("eager_ms"), "triton": d.get("triton_modules")}
    for b in ("onnx", "tensorrt"):
        k = d.get(f"kernel_lens_{b}")
        if isinstance(k, dict):
            row[b] = (round(k["ms"], 2), round(k["top1_agreement"] * 100, 2))
    rows.append(row)
for r in rows:
    def fmt(b):
        v = r.get(b)
        return f"{v[0]:.2f}ms/{v[1]:.2f}%" if v else "-"
    print(f"{r['model']:22} {r['dtype']:8} eager {r['eager_ms']:7.2f}ms  "
          f"ORT {fmt('onnx'):>16}  TRT {fmt('tensorrt'):>16}")
json.dump(rows, open(os.path.join(out, "summary.json"), "w"), indent=2)
print(f"\n{len(rows)} model runs -> {out}/summary.json")
PY
echo "Done. Artefacts in $OUT"
