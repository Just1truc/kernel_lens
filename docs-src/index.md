---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

<img src="assets/logo.png" width="130" alt="KernelLens logo"/>

# KernelLens

### Compile PyTorch **Triton** kernels into native **ONNX Runtime** and **TensorRT** C++ plugins.

Dual-pass tracing &middot; ONNX graph injection &middot; C++/CUDA plugin synthesis &middot; zero-copy VRAM binding &middot; **no C++ boilerplate.**

</div>

<div class="grid cards" markdown>

-   :material-check-decagram:{ .lg .middle } **Exact parity, in C++**

    ---

    Four published LLM research kernels (LLaMA 3 RMSNorm, PaLM SwiGLU, Qwen 2.5 RoPE, Liger fused cross-entropy) reproduce native Triton execution with a maximum absolute difference of `0.00e+00` inside ONNX Runtime.

-   :material-rocket-launch:{ .lg .middle } **Triton speed inside C++ engines**

    ---

    Squared-ReLU FlashAttention runs **3.3x faster than `torch.compile`** and fused residual + RMSNorm is **2.4x faster than eager PyTorch** — as TensorRT plugins, with no Python in the serving path.

-   :material-memory:{ .lg .middle } **Lowest VRAM footprint**

    ---

    In a 4-layer LLaMA decoder the TensorRT plugin path used **464 MB** peak VRAM versus **478 MB** (eager) and **473 MB** (`torch.compile`), because custom kernels share the engine's memory planner.

-   :material-flask-outline:{ .lg .middle } **Independently verified**

    ---

    Every number on this site was produced by re-running the suite on an RTX 5060 Ti (`sm_120a`). See the [verification page](verification.md) for the full pass/fail matrix, the bugs we found, and the fixes.

</div>

## Why this exists

Writing GPU kernels in OpenAI Triton is now routine. **Shipping** them is not. Production servers increasingly run pure C++ inference engines — Microsoft ONNX Runtime (ORT) and NVIDIA TensorRT (TRT) — for latency, concurrency and hardware-specific graph optimisation. But `torch.onnx.export` cannot trace an imperative `@triton.jit` call, so a model that uses custom Triton kernels either stays in Python (forfeiting the C++ engine) or forces a hand-written C++ plugin (thousands of lines per kernel).

KernelLens closes that gap. It compiles a PyTorch module containing custom Triton kernels into (i) a standard ONNX graph with injected custom nodes and (ii) native shared libraries implementing `OrtCustomOp` and `nvinfer1::IPluginV2DynamicExt`, then runs them with **zero-copy VRAM binding**.

![The deployment gap](assets/figures/deployment_gap.png)

## How it works

KernelLens is a four-phase compiler. Every arrow below is a concrete on-disk or in-memory artifact.

![Pipeline](assets/figures/pipeline_end_to_end.png)

| Phase | Input | What KernelLens does | Artifact |
| :--- | :--- | :--- | :--- |
| **1 — Analysis** | PyTorch module + sample inputs | Dual-pass tracing (eager PTX capture + symbolic `make_fx` grid capture) and a static `tl.load`/`tl.store` AST scan for input/output/in-place classification | `KernelManifest` |
| **2 — Graph rewrite** | `KernelManifest` | `torch.onnx.export` with Triton calls replaced by `triton_custom::<kernel>` nodes; host scalars bound as CPU inputs | `<name>.onnx` |
| **3 — Code synthesis** | `KernelManifest` | Emits `OrtCustomOp` / `IPluginV2DynamicExt` C++ sources, rebuilds the PTX entry-slot ABI, clamps PTX ISA versions, and compiles `.so` libraries with `nvcc`/`g++` | `libtriton_*_plugins.so` |
| **4 — Runtime** | `.onnx` + `.so` | Loads the library with `ctypes`, binds PyTorch `data_ptr()` addresses directly into the engine (ORT `IOBinding` / TRT `set_tensor_address`) | `torch` output tensors |

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
model = RMSNorm(4096).cuda()
compiled = kl.compile(model, (x,), name="llama3_rmsnorm", backends=["onnx", "tensorrt"])
out = compiled.run((x,), backend="tensorrt")   # pure C++ execution
```

Ready for more? Start with the [architecture deep-dive](architecture.md), inspect the [research results](results.md), or read how we [verified everything](verification.md).

!!! warning "What this project is — and is not"
    KernelLens is a research-grade compiler. It is honest about its current
    boundaries: the [verification page](verification.md) lists the features that
    are **implemented and measured** separately from those that are **planned**,
    and it reports the two TensorRT paths that still fail numerically. The
    [roadmap](roadmap.md) tracks the rest.