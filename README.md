# KernelLens

<div align="center">
  <img src="https://raw.githubusercontent.com/Just1truc/kernel_lens/main/logo.png" style="border-radius: 50%; margin-bottom: 12px;" alt="KernelLens Logo" width="180"/>
  
  # Automated Triton-to-C++ Compiler for Enterprise Inference Runtimes

  [![PyPI Version](https://img.shields.io/pypi/v/kernel-lens.svg?color=blue)](https://pypi.org/project/kernel-lens/)
  [![Python 3.9+](https://img.shields.io/badge/python-3.9+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
  [![CUDA 12.0+](https://img.shields.io/badge/CUDA-12.0+-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
  [![GitHub Main](https://img.shields.io/github/actions/workflow/status/Just1truc/kernel_lens/ci.yml?branch=main&label=build)](https://github.com/Just1truc/kernel_lens)

  **Bridge the gap between PyTorch Triton research and high-throughput C++ inference engines.**
  
  *Automated Dual-Pass Tracing • SymPy AST Grid Transpilation • Zero-Copy VRAM IO-Binding • Pure C++ Plugin Synthesis*
</div>

<br/>

---

## Overview

**KernelLens** is an open-source compiler framework designed to transform PyTorch models containing custom `@triton.jit` kernels into standalone, high-performance C++ shared libraries (`.so`). 

It generates native **ONNX Runtime (`OrtCustomOp`)** and **NVIDIA TensorRT 10.x (`nvinfer1::IPluginV2DynamicExt`)** plugins automatically—eliminating thousands of lines of C++ boilerplate, manual `nvcc` build scripts, and host-side memory copies.

```
+-----------------------------------------------------------------------------------+
|                            KernelLens Pipeline Flow                              |
+-----------------------------------------------------------------------------------+
|  PyTorch Module (@triton.jit)                                                    |
|        │                                                                          |
|        ▼                                                                          |
|  Phase 1: Dual-Pass Tracing & AST Inspection (Intercept PTX, Smem, Grids)         |
|        │                                                                          |
|        ▼                                                                          |
|  Phase 2: Base ONNX Graph Transformation (Inject Custom Domain Nodes)             |
|        │                                                                          |
|        ▼                                                                          |
|  Phase 3: Automated C++/CUDA Code Synthesis (Generate OrtCustomOp & TRT Plugins)  |
|        │                                                                          |
|        ▼                                                                          |
|  Phase 4: Zero-Copy C++ Runtime Execution (Direct VRAM Pointer Binding)           |
+-----------------------------------------------------------------------------------+
```

---

## Key Features

* **Zero C++ Boilerplate**: Automatically inspects Triton kernel signatures and synthesizes production-ready C++ plugin code (`.cu`, `.cpp`, `.h`) and compiled shared libraries (`.so`).
* **SymPy Dynamic Grid AST Transpilation**: Parses Triton grid launchers (e.g., `triton.cdiv(M, BLOCK_SIZE)`) into pure C++ integer arithmetic evaluated dynamically at inference time ($< 0.5 \ \mu\text{s}$ CPU overhead).
* **Zero-Copy VRAM Pointer Mapping**: Directly binds PyTorch / C++ VRAM memory addresses (`data_ptr()`) into ONNX Runtime and TensorRT execution contexts, achieving $O(1)$ allocation overhead.
* **16-Byte Memory Alignment Guards**: Automatically injects dynamic 16-byte address check guards in C++ to prevent illegal unaligned vector load traps (`LDG.128`) on sub-aligned tensor slices.
* **Toolchain & Hardware Hardening**: Features automatic PTX ISA version clamping (`.version 9.3` $\rightarrow$ `.version 9.0`) for NVIDIA Blackwell (`sm_120a`) GPU driver compatibility.
* **FP8 & Sub-Byte (INT4) Support**: Native 1-byte pointer mapping (`GetTensorData<uint8_t>()`) for quantized Tensor Core acceleration.
* **Cold-Start Disk Caching**: Caches compiled C++ dynamic libraries (`.so`) and TensorRT engines (`.engine`) for instant cold-start loading in production.

---

## Installation

KernelLens can be installed directly from PyPI:

```bash
# Core compiler (FX Graph tracing & C++ code synthesis)
pip install kernel-lens

# Install with target inference backends
pip install kernel-lens[ort]   # ONNX Runtime support
pip install kernel-lens[trt]   # NVIDIA TensorRT support
pip install kernel-lens[all]   # Full backend suite
```

### System Requirements
* **OS**: Linux (x86_64)
* **Python**: 3.9+
* **CUDA Toolkit**: 12.0+
* **PyTorch**: 2.0+
* **Triton**: 2.1+

---

## Quickstart

Take any standard PyTorch `nn.Module` with a custom `@triton.jit` kernel and compile it for production in **3 lines of Python**:

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl
import kernel_lens as kl

# 1. Define a custom Triton kernel and Module
@triton.jit
def rmsnorm_kernel(x_ptr, weight_ptr, out_ptr, stride_x_b, stride_x_m, stride_out_b, stride_out_m, N: tl.constexpr, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    row_x = x_ptr + pid_b * stride_x_b + pid_m * stride_x_m
    row_out = out_ptr + pid_b * stride_out_b + pid_m * stride_out_m
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(row_x + cols, mask=mask, other=0.0)
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    var = tl.sum(x * x, axis=0) / N
    rsqrt = tl.rsqrt(var + eps)
    tl.store(row_out + cols, x * rsqrt * w, mask=mask)

class RMSNormModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        B, M, N = x.shape
        out = torch.empty_like(x)
        rmsnorm_kernel[(B, M)](x, self.weight, out, x.stride(0), x.stride(1), out.stride(0), out.stride(1), N=N, eps=1e-6, BLOCK_SIZE=triton.next_power_of_2(N))
        return out

# 2. Compile to native C++ backends
x = torch.randn(4, 512, 4096, device="cuda", dtype=torch.float32)
model = RMSNormModule(4096).cuda()

compiled_model = kl.compile(
    model, 
    (x,), 
    name="llama3_rmsnorm", 
    backends=["onnx", "tensorrt"]
)

# 3. Execute zero-copy inference in pure C++ engines
ort_output = compiled_model.run((x,), backend="onnx")
trt_output = compiled_model.run((x,), backend="tensorrt")
```

---

## Empirical GPU Latency Benchmark

Evaluated on NVIDIA GPU across state-of-the-art LLM operators:

| Operator / Workload | PyTorch Eager | `torch.compile` | Native Triton | KernelLens TRT | KernelLens ORT | Speedup vs Eager |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mamba SSM Recurrence ($T=256$)** | $4.99\text{ ms}$ | $0.72\text{ ms}$ | $0.70\text{ ms}$ | **$0.68\text{ ms}$** | $0.71\text{ ms}$ | **$7.0\times$** |
| **Fused Cross-Entropy ($V=32\text{K}$)** | $1.85\text{ ms}$ | $0.42\text{ ms}$ | $0.35\text{ ms}$ | **$0.31\text{ ms}$** | $0.33\text{ ms}$ | **$5.6\times$** |
| **LLaMA 3 RMSNorm ($D=4096$)** | $1.58\text{ ms}$ | $0.44\text{ ms}$ | $0.45\text{ ms}$ | **$0.43\text{ ms}$** | $0.46\text{ ms}$ | **$3.4\times$** |
| **PaLM SwiGLU Activation ($N=4096$)** | $1.09\text{ ms}$ | $0.65\text{ ms}$ | $0.64\text{ ms}$ | **$0.62\text{ ms}$** | $0.64\text{ ms}$ | **$1.7\times$** |
| **Qwen 2.5 RoPE Embedding ($D=128$)** | $2.03\text{ ms}$ | $0.45\text{ ms}$ | $0.51\text{ ms}$ | **$0.44\text{ ms}$** | $0.48\text{ ms}$ | **$4.2\times$** |

*All runs achieve exact numerical parity against native Triton execution.*

---

## System Architecture

```
                                  KernelLens Compiler Pipeline
  
  [ PyTorch Model ] ──► Dual-Pass Tracing ──► AST Grid Inspection ──► Manifest Builder
                                                                           │
  [ C++ Exec Engine ] ◄── Dynamic Plugin Load ◄── nvcc / g++ Build ◄───────┴── Synthesized C++/CUDA Code
```

1. **Phase 1: Dual-Pass Tracing**: Intercepts eager Triton launches to record PTX strings, shared memory bytes, and scalar arguments. A secondary PyTorch FX proxy pass captures dynamic grid expressions.
2. **Phase 2: Base ONNX Graph Transformation**: Injects custom domain nodes (`triton_custom::<kernel_name>`) and binds CPU scalar parameters (`OrtMemTypeCPUInput`).
3. **Phase 3: Code Synthesis & Toolchain Hardening**: Emits complete C++ source files implementing `Ort::CustomOp` and `nvinfer1::IPluginV2DynamicExt`. Injects 16-byte alignment guards and applies PTX version clamping.
4. **Phase 4: Zero-Copy Runtime Execution**: Dynamically loads shared libraries via `ctypes.CDLL` and binds GPU VRAM pointers directly to execution contexts.

---

## Advanced Usage & Native Debugging

### Disk Caching & Production Reloading
Once compiled, KernelLens persists plugins to disk. Reload existing engines instantly without recompilation:

```python
# Load previously compiled C++ plugins directly from disk
engine = kl.load("llama3_rmsnorm")
output = engine.run((x,), backend="tensorrt")
```

### Native C++ Debugging Suite
Enable verbose tripwire logging to inspect generated C++ grid calculations and GPU memory addresses before `cuLaunchKernel`:

```bash
KERNEL_LENS_DEBUG=1 python my_inference_script.py
```

---

## Step-by-Step Reproduction & Verification Guide

Follow these steps to reproduce the empirical benchmarks and verify system correctness:

### Step 1: Environment Setup
Clone the repository and install dependencies in editable mode:

```bash
git clone https://github.com/Just1truc/kernel_lens.git
cd kernel_lens
pip install -e .[all]
```

### Step 2: Verify Research Kernel Numerical Parity
Run the research model operator test suite evaluating LLaMA 3 RMSNorm, SwiGLU, RoPE, and Liger-Kernel Fused Cross-Entropy:

```bash
python3 tests/test_research_kernels.py
```

*Expected Output:*
```text
[Passed] Llama 3 RMSNorm Kernel (MaxDiff: 0.000000e+00)
[Passed] SwiGLU Fused Activation Kernel (MaxDiff: 0.000000e+00)
[Passed] RoPE Positional Embedding Kernel (MaxDiff: 0.000000e+00)
[Passed] Fused Softmax & Cross Entropy Loss Kernel (MaxDiff: 0.000000e+00)
ALL RECENT RESEARCH TRITON KERNELS PASSED WITH KERNEL LENS!
```

### Step 3: Run End-to-End Multi-Layer Transformer Decoder Benchmarks
Run the 4-layer LLaMA Transformer decoder pipeline benchmark comparing PyTorch Eager, `torch.compile`, ONNX Runtime, and TensorRT 10.x C++ plugins:

```bash
python3 tests/benchmark_e2e_llama.py
```

*Expected Output:*
```text
--- MEASURED REAL EMPIRICAL END-TO-END RESULTS ---
Configuration                       | TTFT (S=512) | ITL (S=1)    | Total (128 tok) | Peak VRAM 
-----------------------------------------------------------------------------------------------
PyTorch Eager + Triton              |    42.77 ms |     2.68 ms |        0.384 s |   453.2 MB
torch.compile (Inductor)            |    41.81 ms |     2.31 ms |        0.335 s |   451.2 MB
KernelLens C++ Plugins (ORT)        |    45.38 ms |     2.97 ms |        0.423 s |   440.3 MB
KernelLens TensorRT 10.x Plugin     |    43.90 ms |     2.23 ms |        0.328 s |   440.3 MB

Saved real end-to-end benchmark results to measured_e2e_llama.json
```

### Step 4: Verify Edge-Case Stress Tests & Architectural Fixes
Verify dynamic scalar parameters, 4D high-rank launch grids, and nested stores:

```bash
python3 tests/test_architectural_fixes.py
python3 tests/test_edge_cases.py
```

*Expected Output:*
```text
ALL ARCHITECTURAL FIX VERIFICATION TESTS PASSED SUCCESSFULLY!
```

### Step 5: Recompile Technical Report LaTeX
Verify clean PDF generation of the 19-page research paper:

```bash
pdflatex -interaction=nonstopmode architecture_report.tex
```

*Expected Output:* `Output written on architecture_report.pdf (19 pages)`

---

## Citation & Research Paper

If you use **KernelLens** in your research, please cite our technical report:

```bibtex
@article{duc2026kernellens,
  title={KernelLens: Automated Compilation and Zero-Copy C++ Plugin Synthesis for PyTorch Triton Kernels in Enterprise Inference Runtimes},
  author={Duc, Justin},
  institution={Galiad Research},
  year={2026},
  url={https://github.com/Just1truc/kernel_lens}
}
```

---

## License

KernelLens is open-sourced under the [MIT License](LICENSE).
