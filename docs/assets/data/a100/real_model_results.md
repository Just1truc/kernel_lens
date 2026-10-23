# Real pretrained models through KernelLens

Real OPT checkpoints with every decoder `LayerNorm` replaced by the project's
Triton RMSNorm kernel, so the model calls a real `@triton.jit` kernel exactly as
a production model would. Harness: `tests/benchmark_real_model.py`.

All numbers below are from an **A100-SXM4-80GB** (Ampere, sm_80), driver
595.91, CUDA 13.3, PyTorch 2.8.0, Triton 3.4.0, ONNX Runtime 1.31.0,
TensorRT 10.16.1.11.

## Prefill latency, S=512

`torch.compile` is the baseline that matters: it captures the same Triton
kernels rather than reimplementing them, so the two differ only in how the
surrounding operators are scheduled.

| Model | dtype | Eager | `torch.compile` | ORT | TensorRT |
|---|---|---|---|---|---|
| OPT-125M | fp32 | 12.1 ms | 10.9 ms | 4.8 ms | **3.4 ms** |
| OPT-125M | fp16 | 5.7 ms | 2.4 ms | 3.6 ms | **2.0 ms** |
| OPT-350M | fp32 | 33.2 ms | 27.7 ms | 10.2 ms | **8.6 ms** |
| OPT-350M | fp16 | 10.3 ms | 5.1 ms | 7.2 ms | **4.8 ms** |
| OPT-1.3B | fp32 | 98.6 ms | 88.0 ms | 23.4 ms | **19.0 ms** |
| OPT-1.3B | fp16 | 14.7 ms | 10.8 ms | 15.6 ms | **9.9 ms** |
| OPT-2.7B | fp32 | 175.9 ms | 172.7 ms | 37.4 ms | **31.7 ms** |
| OPT-2.7B | fp16 | 22.2 ms | 17.5 ms | 22.6 ms | **16.5 ms** |
| OPT-6.7B | fp16 | 37.0 ms | 33.8 ms | 40.2 ms | **31.1 ms** |

Speedup over `torch.compile`:

| Model | dtype | ORT | TensorRT |
|---|---|---|---|
| OPT-125M | fp32 | 2.27x | **3.18x** |
| OPT-350M | fp32 | 2.71x | **3.24x** |
| OPT-1.3B | fp32 | 3.76x | **4.62x** |
| OPT-2.7B | fp32 | 4.62x | **5.45x** |
| OPT-125M | fp16 | 0.67x | **1.22x** |
| OPT-350M | fp16 | 0.71x | **1.06x** |
| OPT-1.3B | fp16 | 0.69x | **1.09x** |
| OPT-2.7B | fp16 | 0.78x | **1.06x** |
| OPT-6.7B | fp16 | 0.84x | **1.09x** |

**The result splits by precision, and we do not claim a uniform speedup.**

* **fp32** — `torch.compile` barely improves on eager (1.02-1.12x) because it
  leaves the surrounding operators unfused. KernelLens fuses them, so TensorRT
  is **3.2-5.5x faster**, and the margin grows with model size. This is
  graph-level fusion at work, not the normalisation kernels.
* **fp16** — `torch.compile` optimises well there (2.4x over eager at 125M),
  and KernelLens only matches it: TensorRT edges it by **1.06-1.22x**, while
  ONNX Runtime is consistently **slower (0.67-0.84x)**. The compiled fp16 graph
  for OPT-125M carries 1,583 nodes (76 `Cast`, 345 `Unsqueeze`, 549
  `Constant`) against 25 custom operators, so per-node dispatch dominates.

Keeping kernels pays when the baseline leaves the graph unfused; when it
already fuses, we break even.

## Numerical agreement

| Model | dtype | ORT top-1 | ORT mean abs | TRT top-1 | TRT mean abs |
|---|---|---|---|---|---|
| OPT-125M | fp32 | 99.80% | 6.3e-04 | 100.00% | 6.1e-04 |
| OPT-125M | fp16 | 99.80% | 1.0e-03 | 96.09% | 1.2e-02 |
| OPT-350M | fp32 | 100.00% | 3.7e-03 | 99.61% | 3.7e-03 |
| OPT-350M | fp16 | **100.00%** | **0.0e+00** | 95.90% | 4.1e-02 |
| OPT-1.3B | fp32 | 99.80% | 1.4e-03 | 99.80% | 1.4e-03 |
| OPT-1.3B | fp16 | 99.80% | 2.7e-03 | 97.85% | 7.4e-03 |
| OPT-2.7B | fp32 | 98.83% | 4.8e-03 | 98.83% | 6.3e-03 |
| OPT-2.7B | fp16 | 97.66% | 1.7e-02 | 93.55% | 4.0e-02 |

Top-1 is the fraction of positions whose argmax (greedy token) is unchanged.
**Agreement tracks precision rather than speed**: fp32 holds at least 98.8%
everywhere, fp16 under TensorRT falls to 93.6–97.9%. OPT-350M in fp16 under
ONNX Runtime is bit-exact.

Max absolute difference alone is misleading on 50k-way logits — OPT-2.7B fp16
under TensorRT peaks at 1.88 against a logit scale of 16.3 while the mean is
4.0e-02 — so both are reported.

## Why OPT, and not a Llama

The compiler exports the non-kernel part of the model with the TorchScript ONNX
exporter, so exportability depends on the architecture:

| Model | Positions | TorchScript export |
|---|---|---|
| `gpt2`, `distilgpt2` | learned | OK |
| `facebook/opt-*` | learned | OK |
| `bigscience/bloom-560m` | ALiBi | OK |
| `TinyLlama`, `Qwen2.5` | RoPE | fails |

The RoPE families register a rotary cache buffer that stays on CPU under
tracing. This is a property of the exporter, not of the recovered contract:
plain `torch.onnx.export` fails on them identically, with no KernelLens
involved. Liger-Kernel–patched models were also attempted; they fail one step
earlier for the same reason, and (separately) their `in_place` RMSNorm emits
`copy_`/`contiguous` that the exporter rejects.

## Environment notes (learned the hard way)

Three issues had to be fixed on the RunPod image before anything would build:

1. **CUDA 12.8's `nvcc` is broken against a modern libstdc++.** Its
   `math_functions.h` declares `rsqrt`/`sinpi`/... without `noexcept`, which
   GCC >= 11 rejects with *"has a different exception specifier"*. g++-11, -12
   and -13 all failed. Fix: install **CUDA 13.3's nvcc** (driver 595.91
   supports it) and put it first on `PATH`.
2. **The host compiler was hardcoded to `g++-13`** in
   `kernel_lens/backends/builder.py`. Now configurable via
   `KERNEL_LENS_CCBIN`; the scripts select `g++-12` when present.
3. **onnxruntime-gpu's CUDA provider needs CUDA 13 cuBLAS.** The pod ships
   CUDA 12.8 libraries. Fix: install `libcublas-13-3` and friends and point
   `LD_LIBRARY_PATH` at `/usr/local/cuda-13.3/lib64`.

```bash
PATH=/usr/local/cuda-13.3/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64 \
KERNEL_LENS_CCBIN=g++-12 \
  bash scripts/remote_setup.sh && \
PATH=/usr/local/cuda-13.3/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64 \
KERNEL_LENS_CCBIN=g++-12 \
  bash scripts/remote_bench_all.sh
```

## A limitation that turned out to be host-specific

On an earlier A100 host (Thunder Compute) TensorRT deserialization failed for
every engine above roughly 1 GB, with a CUDA runtime error and a host panic in
3 of 4 attempts. TensorRT read the whole blob first (`Loaded engine size: 2710
MiB`), so it was not the 2 GB protobuf limit it resembled. Engines of 315 MB
and 704 MB worked there.

On RunPod the same package builds and deserializes the 2.7 GB (OPT-1.3B) and
5.3 GB (OPT-2.7B) engines without issue. **The limitation was the host, not
TensorRT** — which is why the table above has a TensorRT column at every size.
