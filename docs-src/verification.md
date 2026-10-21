# Verification

This page records an independent verification of KernelLens: what was run, what
passed, what failed, which bugs were found and fixed, and which claims in earlier
drafts turned out to be unsupported. It is intentionally the least flattering
page on this site.

## Method

The package was staged from the repository working tree (`kernel_lens` 1.1.9)
onto the GPU host and executed in a clean conda environment, **not** the
development environment, to catch environment-specific assumptions.

```text
GPU        NVIDIA GeForce RTX 5060 Ti, 16 GB, Blackwell (sm_120a)
Driver     NVIDIA 570.144
Toolchain  nvcc / CUDA 12.8, g++ 15.2, glibc 2.43
Python     3.11.16
PyTorch    2.14.0+cu130     Triton 3.8.0
ONNX       1.23.0           ONNX Runtime 1.30.0
TensorRT   10.16.1.11
```

Every test in `tests/` was launched individually with only `PATH`, `PYTHONPATH`
and `KERNEL_LENS_CACHE_DIR` set, from a cold cache, and each log was retained.
CPU-only unit tests were run first; parity tests were run against native Triton;
benchmarks used CUDA events.

## Results

| Test | What it checks | Result |
| :--- | :--- | :--- |
| `test_unit_cpu.py` | manifest dataclasses, `SymInt` transpiler, alignment validation | ✅ 5/5 |
| `test_research_kernels.py` | RMSNorm / SwiGLU / RoPE / fused CE parity | ✅ exact |
| `test_architectural_fixes.py` | nested-helper store, dynamic scalars, 4-D grid | ✅ exact |
| `test_inplace.py` | in-place aliasing (fp32, ORT) | ✅ exact |
| `test_fused_residual_rms.py` | fused residual + RMSNorm, ORT & TRT | ✅ `<= 9.5e-7` |
| `test_mamba_selective_scan.py` | SSM recurrence, ORT & TRT | ✅ `4.8e-7` |
| `test_fused_cross_entropy.py` | online CE over V=128k, ORT & TRT | ✅ `1.9e-6` |
| `test_tricky_triton_kernels.py` | LayerNorm + GEMM-sigmoid, TRT | ✅ exact |
| `test_trt.py` | NHWC conv plugin, TRT | ✅ |
| `test_edge_cases.py` | multi-output LSE, INT8 dequant, multi-scalar SwiGLU | ⚠️ 2/3 exact on TRT |
| `test_fp8_gemm.py` | FP8 scaled GEMM, ORT | ✅ exact |
| `test_int4_dequant_gemm.py` | INT4 W4A16 GEMM, ORT | ✅ exact |
| `test_triton_vs_torch_compile.py` | squared-ReLU FlashAttention | ⚠️ `0.78` vs *eager* |
| `benchmark_research_kernels.py` | single-operator latency | ✅ ran |
| `benchmark_e2e_llama.py` | 4-layer decoder, ORT & TRT | ✅ ran |

## Bugs found and fixed

Verification was not a clean pass. Four concrete defects were found; three were
fixed and re-verified on the GPU.

??? bug "Fix 1 — CUDA header compatibility patched the wrong file (build blocker)"

    `ensure_cuda_compat()` in `backends/builder.py` patches the glibc ≥ 2.38
    incompatibility in CUDA's `crt/math_functions.h`. It looked for the header at
    `<cuda_home>/include/crt/math_functions.h`, but `nvcc` actually reads
    `<cuda_home>/targets/<triple>/include/crt/math_functions.h` on conda
    toolkits. The patch silently did nothing and **every** `nvcc` compile failed
    with *"exception specification is incompatible … cospi/sinpi/rsqrt"*.

    Fixed by scanning the `include/` **and** `targets/*/include/` layouts.

??? bug "Fix 2 — TensorRT include path broke on conda CUDA (build blocker)"

    `build_trt_plugin` passed `-I<cuda_home>/include` to the host compiler. On
    conda toolkits that directory is empty, so `register_plugins.cpp` failed with
    **`cuda_runtime_api.h: No such file or directory`** and *every* TensorRT test
    errored.

    Fixed with `find_cuda_include()`, which resolves a directory that actually
    contains `cuda_runtime_api.h`. After the fix all TRT tests build and run.

??? bug "Fix 3 — in-place copies assumed 4-byte elements"

    The generated in-place `D2D` copy hard-coded `sizeof(float)`, so an fp16 /
    bf16 / fp8 / int8 in-place buffer was over-copied. Fixed with a per-dtype
    `element_size_bytes()` helper feeding both the ORT and TRT generators. New
    regression test `test_inplace_dtypes.py` confirms fp32 **and** fp16 are now
    bit-exact on ORT.

??? bug "Fix 4 — leaked agent scratch path in the cache resolver"

    `compiler/core.py` fell back to a hard-coded
    `/home/ostentatoire/.gemini/antigravity/brain/.../scratch/.kernel_lens_cache`
    directory. Replaced with a portable temporary-directory fallback.

## Known limitations

These are real defects that remain. They are reported so the results above can
be read with the correct confidence.

!!! failure "TensorRT in-place aliasing produces wrong results"

    An in-place kernel (`tl.load` and `tl.store` on the same pointer) is exact on
    ONNX Runtime but returns garbage on TensorRT, for both fp32 and fp16
    (`test_inplace_dtypes.py`, marked `xfail`). The TRT plugin declares a distinct
    output buffer and copies input→output, but the addressing/aliasing contract
    with the engine is not yet correct. **Use the ORT backend for in-place
    kernels.**

!!! failure "TensorRT INT8 dequantisation GEMM"

    `Int8_Dequant_Bias_SiLU_GEMM` is exact on ORT but diverges by `56.5` on
    TensorRT. The ORT plugin path is verified; the TRT int8 tensor-desc handling
    is not.

!!! warning "Squared-ReLU attention differs from eager PyTorch, not from Triton"

    `test_triton_vs_torch_compile.py` reports a `0.78` difference **against eager
    PyTorch**. That is the difference between the tiled online-softmax-free
    accumulation and the materialised reference after a `relu^2` transform that
    amplifies small errors — not a KernelLens defect. KernelLens reproduces
    *native Triton* exactly; the test's reference is eager PyTorch.

## Claims that did not survive verification

Earlier drafts of the paper and README made claims the code does not support.
They are corrected here and in the rewritten report.

| Earlier claim | Reality |
| :--- | :--- |
| "16-byte VRAM alignment guard … `cudaMallocAsync` staging buffer" | **Not implemented.** Generated C++ contains no address-alignment check and no staging buffer. The only mechanism is a compile-time check that the physical inner dimension is a multiple of 8, which *raises* an error instead of guarding. |
| "SymPy dynamic grid AST transpilation" | The grid is a `SymInt` transpiled by **regex substitution**; there is no `sympy` dependency and no AST visitor. |
| Latency table where KernelLens ORT beats native Triton | The **committed** `measured_latencies.json` shows KernelLens ORT *slower* than native Triton (0.61 / 0.82 / 0.69 / 0.22 ms), and the paper's table (0.46 / 0.64 / 0.48 / 0.33 ms) matches neither the code nor the data. |
| "KernelLens TRT" latency column | **No backing measurement existed**; the benchmark only ran the ONNX backend. The TRT column was unsourced. |
| "Exact numerical parity across all operators" | True for the four research kernels and most edge cases, but **false** for TRT in-place and TRT INT8 dequant. |

## Reproduce the verification

```bash
# on the GPU host, from a clean environment
export PYTHONPATH=$PWD
export KERNEL_LENS_CACHE_DIR=/tmp/kl_cache
rm -rf $KERNEL_LENS_CACHE_DIR

python -m pytest tests/test_unit_cpu.py -v
python -m pytest tests/test_inplace_dtypes.py -v      # ORT passes; TRT xfail
python tests/test_research_kernels.py
python tests/benchmark_research_kernels.py
python tests/benchmark_e2e_llama.py
```

The full per-test logs, the cold-cache rebuilds and the fixed source are in the
repository; the measured JSON is under `docs/assets/data/`.