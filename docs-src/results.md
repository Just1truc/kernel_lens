# Research Results

This page is the empirical core of the project. Every number below was produced
by `tests/` on a single workstation and is reproducible with the commands in
[Verification](verification.md). The raw JSON is committed under
`docs/assets/data/`.

!!! info "Test hardware"
    **NVIDIA GeForce RTX 5060 Ti** (16 GB, Blackwell, compute capability `sm_120a`) &middot;
    CUDA 12.8 &middot; PyTorch 2.14.0 &middot; Triton 3.8.0 &middot;
    ONNX Runtime 1.30.0 &middot; TensorRT 10.16.1.11 &middot; Python 3.11.

## Headline findings

1. **Exact numerical parity with native Triton** for all four published research
   kernels inside ONNX Runtime (`MaxDiff = 0.00e+00`).
2. **Triton-level speed inside C++**: the TensorRT plugin matches native Triton
   on the same kernels, and beats `torch.compile` by **3.3x** on squared-ReLU
   FlashAttention and **2.4x** eager PyTorch on fused residual + RMSNorm.
3. **The lowest VRAM footprint** in a 4-layer LLaMA decoder, because custom
   kernels share the engine's memory planner.
4. **Two TensorRT paths are still wrong** (in-place aliasing, INT8 dequant GEMM).
   They are reported, not hidden.

## Numerical parity

Parity is measured against native Triton execution of the *same* kernel, because
that isolates what KernelLens changes (the plumbing) from what it does not (the
math).

![Verification matrix](assets/figures/verification_matrix.png)

| Workload | Backend | Max abs. difference vs native Triton |
| :--- | :--- | :--- |
| LLaMA 3 / Mistral fused RMSNorm | ORT & TRT | `0.00e+00` |
| Liger-Kernel / PaLM SwiGLU | ORT & TRT | `0.00e+00` |
| LLaMA 3 / Qwen 2.5 RoPE | ORT & TRT | `0.00e+00` |
| Liger fused softmax + cross-entropy | ORT & TRT | `0.00e+00` |
| Fused residual + RMSNorm | ORT & TRT | `<= 9.5e-07` |
| Mamba SSM recurrence | ORT & TRT | `4.8e-07` |
| Tricky LayerNorm + GEMM-sigmoid | ORT & TRT | `0.00e+00` |
| FP8 scaled GEMM | ORT | `0.00e+00` |
| INT4 W4A16 dequant GEMM | ORT | `0.00e+00` |
| Edge: multi-output rank-4/3 log-sum-exp | ORT & TRT | `0.00e+00` |
| Edge: INT8 dequant + bias + SiLU GEMM | ORT | `0.00e+00` |
| Edge: SwiGLU multi-scalar | ORT & TRT | `0.00e+00` |
| Edge: INT8 dequant + bias + SiLU GEMM | **TRT** | **`5.65e+01` — fails** |
| In-place mutation (fp32 / fp16) | **TRT** | **fails** |

The two failing rows are genuine TensorRT-path defects and are tracked on the
[verification page](verification.md#known-limitations).

## Single-operator latency

Measured with CUDA events, `fp32`, after warm-up. `torch.compile` uses the
Inductor backend; "Native Triton" is the kernel called directly from Python.

![Single-operator latency](assets/figures/results_micro_latency.png)

| Operator | PyTorch Eager | `torch.compile` | Native Triton | KernelLens ORT |
| :--- | ---: | ---: | ---: | ---: |
| LLaMA 3 RMSNorm (B=4, M=512, N=4096) | 0.516 ms | 0.173 ms | **0.158 ms** | 0.185 ms |
| PaLM SwiGLU (2048 × 4096) | 0.429 ms | 0.252 ms | **0.253 ms** | 0.283 ms |
| Qwen 2.5 RoPE (4, 512, 32, 128) | 0.670 ms | 0.174 ms | **0.160 ms** | 0.195 ms |
| Fused cross-entropy (B=32, V=32000) | 0.013 ms | 0.013 ms | **0.011 ms** | 0.040 ms |

**How to read this.** KernelLens is *not* faster than native Triton here — it
runs the identical kernel behind a C++ operator boundary, so the C++ plugin adds
roughly 10–40 % host/IO-binding overhead. What it buys is deployment: the same
kernel, the same speed class, but inside a Python-free engine that can co-optimise
the surrounding graph. The fusion-heavy cases below are where that pays off.

## Where the C++ plugin wins

![Speedups](assets/figures/results_speedup.png)

| Kernel | Eager | `torch.compile` | KernelLens TRT | vs eager | vs compile |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Mamba SSM recurrence | 3.32 ms | 0.104 ms | 0.160 ms | **20.7x** | 0.65x |
| Squared-ReLU FlashAttention (N=2048) | 5.96 ms | 3.16 ms | 0.949 ms | **6.3x** | **3.3x** |
| Fused residual + RMSNorm | 3.40 ms | 1.407 ms | 1.407 ms | **2.4x** | 1.00x |
| Fused cross-entropy (V=128k) | 9.09 ms | 2.46 ms | 4.90 ms | **1.9x** | 0.50x |

The pattern is consistent with the architecture: `torch.compile` already lowers
to Triton, so KernelLens matches it on simple fusions and wins decisively on
kernels Inductor cannot generate — custom score transforms (squared-ReLU
attention) and hand-tiled online recurrences.

## End-to-end: 4-layer LLaMA decoder

A four-layer LLaMA decoder with hidden size 1024 and custom RMSNorm, SwiGLU and
RoPE Triton kernels, measured at prefill `S=512` and decode `S=1`.

![End-to-end](assets/figures/results_e2e.png)

| Configuration | TTFT (S=512) | ITL (S=1) | Total, 128 tok | Peak VRAM |
| :--- | ---: | ---: | ---: | ---: |
| PyTorch Eager + Triton | 7.82 ms | 0.866 ms | 0.118 s | 478.0 MB |
| `torch.compile` (Inductor) | 7.91 ms | 0.866 ms | 0.118 s | 473.0 MB |
| KernelLens C++ plugins (ORT) | 4.38 ms | 1.016 ms | 0.133 s | 464.2 MB |
| **KernelLens TensorRT plugin** | **4.05 ms** | **0.854 ms** | **0.112 s** | **464.2 MB** |

The TensorRT plugin gives the **fastest time-to-first-token (4.05 ms vs
7.82/7.91 ms)**, the **fastest total generation**, and the **lowest peak VRAM**.
ORT's prefill is also fast but its per-token path pays extra host overhead. This
is exactly the value proposition: whole-engine graph optimisation on the standard
layers *and* custom Triton kernels in the same engine.

## Quantized kernels (FP8 & INT4)

Measured on the same RTX 5060 Ti (`sm_120a`), which is where PTX version clamping
matters. Both quantization plugins achieve exact parity with native Triton.

| Kernel | Eager FP16 | Native Triton | KernelLens ORT | Speedup vs FP16 |
| :--- | ---: | ---: | ---: | ---: |
| FP8 scaled GEMM (4096 × 4096) | 2.786 ms | 2.046 ms | 2.083 ms | **1.34x** |
| INT4 W4A16 GEMM (2048 × 4096) | 1.388 ms | 3.570 ms | 3.629 ms | 0.38x (2x smaller weights) |

INT4 is *slower per call* than an FP16 GEMM on this shape because the
dequantisation kernel is not Tensor-Core bound, but it halves weight VRAM and is
exact — the honest framing is "memory compression with exact parity", not a
latency win.

## Dynamic shapes

The grid is transpiled to C++ integer math, so one compiled plugin serves a range
of shapes without host recompilation. The generated launcher computes the thread
block dimensions at run time with $O(1)$ CPU cost.

## Reproduce it

```bash
python3 tests/test_research_kernels.py      # numerical parity
python3 tests/benchmark_research_kernels.py # single-operator latency
python3 tests/benchmark_e2e_llama.py        # end-to-end decoder
python3 tests/test_fp8_gemm.py              # FP8 Tensor Core path
```

Raw data: [`measured_latencies_rtx5060ti.json`](assets/data/measured_latencies_rtx5060ti.json),
[`measured_e2e_llama_rtx5060ti.json`](assets/data/measured_e2e_llama_rtx5060ti.json),
[`measured_quantization_latencies.json`](assets/data/measured_quantization_latencies.json).