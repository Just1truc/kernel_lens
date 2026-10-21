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

> **Honest scope.** KernelLens is research-grade. The documentation clearly
> separates what is **implemented and measured** from what is **planned**. Two
> TensorRT paths (in-place aliasing, INT8 dequantisation) are known to be
> incorrect and are reported rather than hidden. See the
> [verification page](https://just1truc.github.io/kernel_lens/verification/).

## Installation

```bash
pip install kernel-lens          # core compiler
pip install kernel-lens[ort]     # + ONNX Runtime backend
pip install kernel-lens[trt]     # + TensorRT backend
pip install kernel-lens[all]     # both
```

**Requirements:** Linux x86-64, Python ≥ 3.10, CUDA ≥ 12.0, PyTorch ≥ 2.0,
Triton ≥ 2.1, and `nvcc` + `g++` on `PATH`.

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
```

## Results at a glance

Measured on an **RTX 5060 Ti (Blackwell, `sm_120a`)**, CUDA 12.8, PyTorch 2.14,
Triton 3.8, ONNX Runtime 1.30, TensorRT 10.16. Full tables and methodology are on
the [results page](https://just1truc.github.io/kernel_lens/#results).

- **Fold + keep, and it pays**: on a two-layer block whose attention is a custom
  squared-ReLU flash kernel, KernelLens runs it in **7.63 ms**, versus **15.15 ms**
  for the folded-only `torch.compile` formulation and **28.04 ms** for eager
  folding — **1.99×** and **3.68×** faster — and **1.08×** faster than the
  unfused Triton-kernel path. The engine folds the standard layers; the kernel
  keeps the operations folding cannot express.
- **Exact parity with native Triton** (`MaxDiff = 0.00e+00`) for LLaMA 3 RMSNorm,
  PaLM SwiGLU, Qwen 2.5 RoPE, Liger fused cross-entropy, in-place mutation and
  the quantized GEMMs, on both backends.
- **Lowest VRAM** in a 4-layer LLaMA decoder: **464 MB** peak vs 478 MB (eager)
  and 473 MB (`torch.compile`), with the fastest time-to-first-token (4.14 ms vs
  7.85/7.92 ms).

## Documentation

| Section | Contents |
| :--- | :--- |
| [Problem](https://just1truc.github.io/kernel_lens/#problem) | The five gaps a naive export hits |
| [Pipeline](https://just1truc.github.io/kernel_lens/#pipeline) | What you build by hand vs. the compile/run pipeline |
| [Results](https://just1truc.github.io/kernel_lens/#results) | Folding + kernels, parity, latency, end-to-end |
| [Install & API](https://just1truc.github.io/kernel_lens/#install) | Installation and API reference |
| [Status](https://just1truc.github.io/kernel_lens/#status) | Roadmap |
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
```

Raw measurement JSON is committed under [`docs/assets/data/`](docs/assets/data/).

## Technical report

The LaTeX source is [`architecture_report.tex`](architecture_report.tex); the TMLR
variant is generated from it with `python3 scripts/make_tmlr.py`.

```bibtex
@article{duc2026kernellens,
  title       = {KernelLens: Compiling PyTorch Triton Kernels into C++ Inference Plugins},
  author      = {Duc, Justin},
  institution = {Galiad Research},
  year        = {2026},
  version     = {1.1.9},
  doi         = {10.5281/zenodo.23194523},
  url         = {https://doi.org/10.5281/zenodo.23194523}
}
```

## License

MIT — see [LICENSE](LICENSE).
