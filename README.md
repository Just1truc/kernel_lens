# KernelLens

<div align="center">
  <img src="https://raw.githubusercontent.com/Just1truc/kernel_lens/main/logo.png" style="border-radius: 50%; margin-bottom: 12px;" alt="KernelLens Logo" width="170"/>

  # Compiling PyTorch Triton Kernels into C++ Inference Plugins

  [![PyPI Version](https://img.shields.io/pypi/v/kernel-lens.svg?color=blue)](https://pypi.org/project/kernel-lens/)
  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23194523.svg)](https://doi.org/10.5281/zenodo.23194523)
  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
  [![CUDA 12.0+](https://img.shields.io/badge/CUDA-12.0+-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

  **📖 Documentation & research results: <https://just1truc.github.io/kernel_lens/>**

  *Dual-pass tracing • ONNX custom-node injection • automated OrtCustomOp / TensorRT plugin synthesis • zero-copy VRAM binding • no C++ boilerplate*
</div>

<br/>

## What it is

**KernelLens** compiles a PyTorch model that contains custom `@triton.jit` kernels
into native C++ artifacts for **ONNX Runtime** (`OrtCustomOp`) and **NVIDIA
TensorRT 10.x** (`nvinfer1::IPluginV2DynamicExt`). You write Triton in Python; it
emits the C++ plugin, builds the `.so`, and runs it with **zero-copy VRAM
binding** — no manual C++ and no Python in the serving path.

```
kl.compile(model, sample_input, backends=["onnx", "tensorrt"])
   │
   ├─ trace          eager PTX + symbolic grid + AST argument taxonomy
   ├─ rewrite        triton_custom node injection  ─▶  <name>.onnx
   ├─ synthesize     OrtCustomOp / IPluginV2DynamicExt sources
   └─ build          nvcc / g++  ─▶  libtriton_*.so  (+ TRT engine)

compiled_model.run(inputs, backend="onnx"|"tensorrt")
   └─ zero-copy binding (IOBinding / set_tensor_address)  ─▶  torch tensors
```

## Key features

- **Dual-pass tracing** — a concrete eager pass captures the real compiled PTX,
  shared memory and strides; a symbolic `make_fx` pass captures dynamic launch
  grids; a static `tl.load`/`tl.store` AST scan classifies each argument as
  input, output or in-place.
- **Automated plugin synthesis** — generates `OrtCustomOp` and
  `IPluginV2DynamicExt` C++ sources, reconstructs the PTX entry-slot ABI, and
  compiles them with `nvcc`/`g++`.
- **Dynamic grids in C++** — launch-grid expressions are transpiled to C++
  integer math over runtime shapes, so one plugin serves dynamic batch and
  sequence lengths.
- **One package, several TensorRT versions** — the plugin interface is chosen
  from the environment. TensorRT 10 deprecated the whole `IPluginV2*` hierarchy
  and TensorRT 11 removed it, so `plugin_api="auto"` (the default) emits
  `IPluginV3` plugins where the runtime supports them and `IPluginV2DynamicExt`
  plugins otherwise. This spans TensorRT 8.6 through 10.x today; TensorRT 11+
  works when you point `TENSORRT_INCLUDE_DIR` at matching headers.
- **Shape-generic engines** — `kl.compile(..., dynamic_shapes=True)` exports the
  leading dimension as a symbolic axis, so a single engine serves any size in a
  declared range via a TensorRT optimization profile. Verified on Blackwell
  (`sm_120a`) and Ampere (`sm_80`): one compile at batch 8 serving batches
  1-16, bit-exact on both engines. Off by default, because a shape-specialised
  engine is what the published benchmarks measure.
- **Zero-copy VRAM binding** — PyTorch `data_ptr()` values are registered
  directly with each engine; no host staging copies.
- **Blackwell PTX ISA clamping** — a Triton-generated `.version 9.3` header is
  rewritten to `.version 9.0` so a CUDA driver JIT can load `sm_120a` PTX.
- **FP8 & sub-byte INT4** — 1-byte pointer mapping (`GetTensorData<uint8_t>()`)
  and dtype-accurate in-place copies.
- **Cold-start disk caching** — `.so` and `.engine` artifacts are cached for
  instant reload via `kl.load(name)`.
- **Broad module coverage** — a conformance suite (`tests/test_conformance.py`)
  covers two kernels per graph, the same kernel launched twice, multiple outputs,
  dynamic scalars, strided and dynamic inputs, loops, conditionals, nested
  helpers, in-place mutation, 3-D grids, autotune, and fp16/bf16.

> **Honest scope.** KernelLens is research-grade, and the results are reported
> per configuration rather than blended. The composition is bit-exact against
> native Triton (`0.00e+00`) across the research kernels, the fourteen
> conformance patterns and the quantized GEMMs, on both backends; two
> multi-operator fusions land just above zero and are reported as such. Keeping
> the kernel pays when the baseline leaves the graph unfused and breaks even when
> it already fuses. The TensorRT-LLM backend is design-only (see
> [`docs/assets/data/a100/trtllm_backend_design.md`](docs/assets/data/a100/trtllm_backend_design.md)),
> and the quantized GEMMs are validated on ONNX Runtime only. Full methodology
> and every raw JSON are under [`docs/assets/data/`](docs/assets/data/).

## Installation

```bash
pip install kernel-lens          # core compiler
pip install kernel-lens[ort]     # + ONNX Runtime backend
pip install kernel-lens[trt]     # + TensorRT backend
pip install kernel-lens[all]     # both
```

**Requirements:** Linux x86-64, Python ≥ 3.10, CUDA ≥ 12.0, PyTorch ≥ 2.0,
Triton ≥ 2.1, and `nvcc` + `g++` on `PATH`. TensorRT must be `< 11` (the
generator emits `IPluginV2DynamicExt`).

**TensorRT versions.** Handled automatically. The pip TensorRT wheels ship **no
headers**, so KernelLens fetches the set matching the installed runtime from the
TensorRT OSS repository and caches it under `~/.kernel_lens_headers/<major>.<minor>`;
when the headers it already has match the runtime major, nothing is fetched. If
the download is unavailable the fallback is the bundled set, and a
header/runtime major mismatch is then reported instead of crashing inside
TensorRT. `plugin_api="auto"` picks the interface from the runtime:

| TensorRT | `IPluginV2*` | `IPluginV3` | emitted |
| :--- | :--- | :--- | :--- |
| < 10 (8.6, 9.x) | present | absent | `v2` |
| 10.x | deprecated | present | `v3` |
| 11+ | **removed** | present | `v3` |

Verified on TensorRT 10.16 and 11.3, both 16/16 on the conformance suite. Force
a choice with `plugin_api="v2"` / `"v3"`, or the `KERNEL_LENS_PLUGIN_API`
environment variable. To supply headers yourself — offline installs, or a TensorRT
build the OSS repository does not track — set `TENSORRT_INCLUDE_DIR`, or
`KERNEL_LENS_FETCH_HEADERS=0` to disable fetching entirely.

**Environment:** `KERNEL_LENS_CACHE_DIR` (put it on a persistent volume so the
`.so` / `.engine` artifacts survive `kl.load`), `KERNEL_LENS_CCBIN` (host
compiler for `nvcc` — CUDA toolkits before 12.9 reject GCC 13's exception
specifiers, so set it to `g++-12` there), `KERNEL_LENS_DEBUG` (or `KL_VERBOSE`),
`TENSORRT_INCLUDE_DIR` (override the TensorRT headers),
`KERNEL_LENS_HEADER_CACHE` (where fetched TensorRT headers are kept),
`KERNEL_LENS_FETCH_HEADERS=0` (disable fetching),
`KERNEL_LENS_PLUGIN_API` (`auto`, `v2` or `v3`), and `KERNEL_LENS_STREAM_READER=1` to deserialize engines
above 2 GB from disk rather than from a memory buffer (opt-in; see the
[changelog](CHANGELOG.md)).

## Quickstart

```python
import torch, triton, triton.language as tl
import kernel_lens as kl

@triton.jit
def rmsnorm_kernel(x_ptr, weight_ptr, out_ptr,
                   stride_x_b, stride_x_m, stride_out_b, stride_out_m,
                   N: tl.constexpr, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid_b, pid_m = tl.program_id(0), tl.program_id(1)
    row_x = x_ptr + pid_b * stride_x_b + pid_m * stride_x_m
    row_out = out_ptr + pid_b * stride_out_b + pid_m * stride_out_m
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(row_x + cols, mask=mask, other=0.0)
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    var = tl.sum(x * x, axis=0) / N
    tl.store(row_out + cols, x * tl.rsqrt(var + eps) * w, mask=mask)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
    def forward(self, x):
        B, M, N = x.shape
        out = torch.empty_like(x)
        rmsnorm_kernel[(B, M)](x, self.weight, out, x.stride(0), x.stride(1),
                               out.stride(0), out.stride(1),
                               N=N, eps=1e-6, BLOCK_SIZE=triton.next_power_of_2(N))
        return out

x = torch.randn(4, 512, 4096, device="cuda")
compiled = kl.compile(RMSNorm(4096).cuda(), (x,), name="llama3_rmsnorm",
                      backends=["onnx", "tensorrt"])
out_ort = compiled.run((x,), backend="onnx")
out_trt = compiled.run((x,), backend="tensorrt")

# One engine for any batch: mark the leading dim dynamic. dynamic_max bounds
# the optimization profile (default 4x the traced size).
serving = kl.compile(RMSNorm(4096).cuda(), (x,), name="llama3_rmsnorm_dyn",
                     backends=["onnx", "tensorrt"], dynamic_shapes=True)
for b in (1, 2, 4, 8, 16):                 # all through the engine built at b=4
    serving.run((torch.randn(b, 512, 4096, device="cuda"),), backend="tensorrt")
```

## Results at a glance

Two architectures: an **RTX 5060 Ti** (Blackwell, `sm_120a`) for the folding,
single-operator, end-to-end and quantized results, and an **A100-SXM4-80GB**
(Ampere, `sm_80`) for the conformance, parity and real-model results. Results are
reported per configuration and never blended. Full methodology, complete tables
and every raw JSON are on the
[results page](https://just1truc.github.io/kernel_lens/#results) and under
[`docs/assets/data/`](docs/assets/data/).

**The code you don't write.** The deployment is **nine lines of Python**; the C++
it replaces is **1,137 lines across six files** for one fused RMSNorm kernel on
both backends, measured with `tests/measure_codegen.py`.

**One engine, many shapes.** With `dynamic_shapes=True` a single compiled engine
serves any leading-dimension size in range. Measured with
`tests/test_conformance.py`, which compiles once and then runs every shape through
that engine — `0.00e+00` on both backends, on Blackwell and on Ampere:

| Case | Compiled at | Served |
| :--- | :--- | :--- |
| `dynamic_batch_sweep` | batch 8 | batches 1, 2, 4, 8, 16 |
| `dynamic_grid_sweep` | 524,288 elements | 65,536 / 131,072 / 524,288 / 1,048,576 |

**Real pretrained models.** Every decoder `LayerNorm` of a real OPT checkpoint
(125M → 6.7B) replaced by the Triton RMSNorm kernel, graded against
`torch.compile` — the baseline that *captures the same Triton kernels* rather
than reimplementing them. The result splits by precision:

- **fp32** — `torch.compile` barely improves on eager (1.02–1.12×) because it
  leaves the operators around the kernel unfused. KernelLens fuses them:
  **3.18–5.45× faster** under TensorRT and **2.27–4.62×** under ONNX Runtime,
  and the margin grows with model size.
- **fp16** — `torch.compile` optimises well there, so we only match it: TensorRT
  edges it by **1.06–1.22×**, while ONNX Runtime is consistently **slower
  (0.67–0.84×)** — per-node dispatch dominates when the graph already fuses.
- **Accuracy tracks precision, not speed.** fp32 holds the same greedy token on
  **≥ 98.8%** of positions; fp16 under TensorRT falls to 93.4–97.9%. OPT-350M in
  fp16 under ONNX Runtime is bit-exact.

**Fold + keep, and it pays.** On a two-layer block whose attention is a custom
squared-ReLU flash kernel, KernelLens runs it in **7.63 ms**, versus **15.15 ms**
for the folded-only `torch.compile` formulation and **28.04 ms** for eager
folding — **1.99×** and **3.68×** faster. Against the unfused kernel path it is
**1.08×** faster than eager Triton (8.24 ms) and level with `torch.compile`
(7.55 ms). The engine folds the standard layers; the kernel keeps the operations
folding cannot express.

**Exact parity with native Triton** (`MaxDiff = 0.00e+00`) for LLaMA 3 RMSNorm,
PaLM SwiGLU, Qwen 2.5 RoPE, Liger fused cross-entropy, in-place mutation and the
quantized GEMMs, on both backends — and across all **fourteen** module patterns
of `tests/test_conformance.py`. Two multi-operator fusions are near-exact rather
than exact (`≤ 9.5e-07` residual+RMSNorm, `4.8e-07` Mamba's scan) and are
reported as such.

**Lowest VRAM** in a 4-layer LLaMA decoder: **464 MB** peak vs 478 MB (eager)
and 473 MB (`torch.compile`), with the fastest time-to-first-token (4.14 ms vs
7.85/7.92 ms).

## Documentation

| Section | Contents |
| :--- | :--- |
| [Problem](https://just1truc.github.io/kernel_lens/#problem) | The five gaps a naive export hits |
| [Pipeline](https://just1truc.github.io/kernel_lens/#pipeline) | What you build by hand vs. the compile/run pipeline |
| [Results](https://just1truc.github.io/kernel_lens/#results) | Folding + kernels, parity, conformance, single-operator and end-to-end latency, **real pretrained models** |
| [Why an engine](https://just1truc.github.io/kernel_lens/#why) | `torch.compile` vs TensorRT, both directions |
| [Install & API](https://just1truc.github.io/kernel_lens/#install) | Installation and API reference |
| [Status](https://just1truc.github.io/kernel_lens/#status) | Roadmap |
| [FAQ](https://just1truc.github.io/kernel_lens/#faq) | Fair objections |
| [Paper](https://just1truc.github.io/kernel_lens/#paper) | Abstract and citation |

## Reproduce the results

```bash
git clone https://github.com/Just1truc/kernel_lens.git && cd kernel_lens
pip install -e .[all]

python tests/benchmark_folding_vs_triton.py    # folding + kernels vs the alternatives
python tests/test_conformance.py               # module-pattern conformance (14 patterns)
python tests/test_research_kernels.py          # numerical parity
python tests/benchmark_research_kernels.py     # single-operator latency
python tests/benchmark_e2e_llama.py            # end-to-end decoder
python tests/test_fp8_gemm.py                  # FP8 Tensor Core path
python tests/measure_codegen.py                # generated C++ volume, per file
python tests/benchmark_real_model.py --model facebook/opt-1.3b   # real OPT, both precisions
```

The real-model matrix (OPT-125M → 6.7B, fp32 and fp16) runs unattended on a fresh
CUDA host: `scripts/remote_setup.sh` provisions nvcc, TensorRT, `onnxruntime-gpu`
and `transformers` and installs the package; `scripts/remote_bench_all.sh` then
sweeps the matrix into `results/` and writes `summary.json`, recording failures
instead of aborting. See
[`docs/assets/data/a100/runpod_runbook.md`](docs/assets/data/a100/runpod_runbook.md) for the
exact invocation and the environment traps (CUDA 12.8's `nvcc` does not build
against a modern libstdc++; CUDA 13.3's does).

Raw measurement JSON is committed under [`docs/assets/data/`](docs/assets/data/).

## Technical report

The paper source is [`kernel_lens_fresh.tex`](kernel_lens_fresh.tex). Two derived
variants are generated from it and must not be edited by hand:
`scripts/make_mlsys.py` produces the 2-column MLSys form
(`kernel_lens_mlsys.tex`) and `scripts/make_tmlr_fresh.py` produces the TMLR
forms — `kernel_lens_tmlr_fresh.tex` (named preprint) and
`kernel_lens_tmlr_submission.tex` (anonymized; **do not publish it alongside the
named preprint**).

```bibtex
@article{duc2026kernellens,
  title       = {KernelLens: Compiling PyTorch Triton Kernels into C++ Inference Plugins},
  author      = {Duc, Justin},
  institution = {Galiad Research},
  year        = {2026},
  version     = {1.2.1},
  doi         = {10.5281/zenodo.23194523},
  url         = {https://doi.org/10.5281/zenodo.23194523}
}
```

## License

MIT — see [LICENSE](LICENSE).
