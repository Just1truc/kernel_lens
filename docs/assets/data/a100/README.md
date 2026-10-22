# Cross-architecture verification: NVIDIA A100-SXM4-80GB

Second architecture, run to check that the recovered Triton -> C++ engine
contract is not specific to the Blackwell consumer card used for the main
results. All runs used the same package revision.

## Configuration

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB (Ampere, compute capability 8.0) |
| Driver | 610.43.02 (CUDA UMD 13.3) |
| CUDA toolkit | 13.3 (nvcc V13.3.73) |
| TensorRT | 10.16.1.11 |
| ONNX Runtime | 1.31.0 (CUDA + TensorRT execution providers) |
| PyTorch | 2.12.1+cu130 |
| Triton | 3.7.1 |
| Host | 8 vCPU, 64 GB RAM, 100 GB disk |

## Conformance (14 module patterns) — 14/14

Both backends, maximum absolute difference vs native Triton. All values are
exactly 0.00e+00, i.e. bit-identical to the Triton reference.

| Case | ONNX Runtime | TensorRT |
|---|---|---|
| two_kernels | 0.00e+00 | 0.00e+00 |
| same_kernel_twice | 0.00e+00 | 0.00e+00 |
| two_outputs | 0.00e+00 | 0.00e+00 |
| dynamic_scalar | 0.00e+00 | 0.00e+00 |
| strided | 0.00e+00 | 0.00e+00 |
| loop | 0.00e+00 | 0.00e+00 |
| conditional | 0.00e+00 | 0.00e+00 |
| nested_helper | 0.00e+00 | 0.00e+00 |
| inplace | 0.00e+00 | 0.00e+00 |
| grid3d | 0.00e+00 | 0.00e+00 |
| autotuned | 0.00e+00 | 0.00e+00 |
| fp16 | 0.00e+00 | 0.00e+00 |
| bf16 | n/a (ORT cannot bind bf16) | 0.00e+00 |
| dynamic_batch | 0.00e+00 | 0.00e+00 |

Parity suite (`tests/test_research_kernels.py`): **4/4 passed** on Ampere.

## Folding plus kernels (fp32)

Two-layer block, D=512, H=8, FFN=2048, S=4096, batch 1.

| Configuration | Latency | Relative |
|---|---|---|
| A. Folded only — eager, standard ops | 6.973 ms | 0.33x |
| A. Folded only — torch.compile | 3.851 ms | 0.60x |
| B. Kernels only — eager Triton | 2.300 ms | 1.00x |
| B. Kernels only — torch.compile | 2.241 ms | 1.03x |
| C. KernelLens — ONNX Runtime | 6.082 ms | 0.38x |
| C. KernelLens — TensorRT | 2.998 ms | 0.77x |

The direction of the result is unchanged — folding plus kernels still beats
folded-only — but the margin is much smaller on Ampere than on the Blackwell
consumer card (1.28x vs 1.99x), and the ONNX Runtime lane is *slower* than the
folded-only `torch.compile` baseline here. Both effects are consistent with the
paper's finding that the benefit of keeping kernels is workload- and
engine-dependent rather than universal.

## End-to-end 4-layer LLaMA decoder

| Configuration | TTFT (S=512) | ITL (S=1) | Total (128 tok) | Peak VRAM |
|---|---|---|---|---|
| PyTorch eager + Triton | 7.01 ms | 0.95 ms | 0.127 s | 453.2 MB |
| torch.compile (Inductor) | 6.66 ms | 0.61 ms | 0.084 s | 446.2 MB |
| KernelLens — ONNX Runtime | 4.21 ms | 3.40 ms | 0.436 s | 440.4 MB |
| KernelLens — TensorRT | 2.04 ms | 0.74 ms | 0.096 s | 442.4 MB |

The TensorRT lane has the lowest time-to-first-token of any configuration
(3.4x better than eager, 3.3x better than `torch.compile`). The ONNX Runtime
lane keeps a good TTFT but its per-token overhead is much higher, so it loses
on sustained decode.

## Defect found and fixed on this run

The first conformance run failed 2/14 (`fp16`, `bf16`) with

```
IBuilder::buildSerializedNetwork: Error Code 9: Internal Error
  (/scale_kernel_conf_fp16__0: could not find any supported formats
   consistent with input/output data types)
```

Cause: the TensorRT builder configuration never enabled the FP16/BF16
precision flags. TensorRT therefore kept the plugin layer in FP32, inserted a
cast on the plugin's inputs, and queried `supportsFormatCombination` once with
`kFLOAT`. The generated plugin's dtype guard mirrors the kernel signature and
correctly requires `kHALF`, so it rejected the only combination offered and the
build failed.

Fix (`kernel_lens/runtime/engine.py`): read the ONNX graph's input element
types and set `BuilderFlag.FP16` / `BuilderFlag.BF16` accordingly before
building. With the fix, all 14 conformance cases pass on Ampere.

## Notes

- FP8 is not exercised here: the A100 (Ampere) has no FP8 tensor cores. The FP8
  quantized path remains Blackwell/Hopper-only.
- Timings are from a dedicated cloud instance (per-minute billing, no noisy
  neighbours), but still a virtualised VM; absolute numbers should be compared
  within this table only.
