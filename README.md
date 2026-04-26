# Kernel Lens

<div align="center">
  <img src="https://raw.githubusercontent.com/Just1truc/kernel_lens/main/logo.png" style="border-radius: 50%;" alt="KL Logo"  width="250"/>
</div>

-------------------


**A production-grade, multi-backend compiler for Triton kernels.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Kernel Lens bridges the gap between PyTorch research and high-performance C++ production. It automatically traces PyTorch modules, intercepts custom Triton kernels, generates optimized C++ bindings, and compiles them into native ONNX Runtime and TensorRT plugins—with zero C++ boilerplate required.

## 📦 Installation

Install the core compiler (PyTorch & ONNX graph tracing):
```bash
pip install kernel-lens
```

Install with inference backends:
```bash
pip install kernel-lens[ort]   # For ONNX Runtime support
pip install kernel-lens[trt]   # For TensorRT support
pip install kernel-lens[all]   # For everything
```

## 🚀 Quickstart

Take any standard PyTorch `nn.Module` containing a `@triton.jit` kernel, and compile it for production in one line:

```python
import torch
import kernel_lens as kl
from my_models import TritonMatmul  # Your custom PyTorch/Triton model

model = TritonMatmul().cuda()
A = torch.randn((128, 128), device='cuda')
B = torch.randn((128, 128), device='cuda')

# 1. Compile the model to native C++ backends
compiled_model = kl.compile(
    model, 
    (A, B), 
    name="my_fast_matmul", 
    backends=["onnx", "tensorrt"]
)

# 2. Execute native zero-copy inference!
trt_output = compiled_model.run((A, B), backend="tensorrt")
ort_output = compiled_model.run((A, B), backend="onnx")
```

## ✨ Comprehensive Features

### 1. Zero C++ Boilerplate
Kernel Lens entirely automates the generation of native C++ bindings. It reads your Triton kernel signatures and seamlessly generates robust `Ort::CustomOp` and `nvinfer1::IPluginV2` plugins. No bash scripts, no manual `nvcc` flags—just Python.

### 2. Dynamic Grid AST Parsing
Triton utilizes dynamic grid calculations (e.g., `triton.cdiv(M, BLOCK_SIZE)`). Kernel Lens intercepts PyTorch's symbolic tracing, sanitizes the AST (Abstract Syntax Tree), and dynamically translates it into raw, high-performance C++ integer math for the GPU block scheduler.

### 3. Deep Multi-Kernel Tracing
Your custom kernels don't need to be at the top level. Kernel Lens uses PyTorch `make_fx` to flatten complex, nested `nn.Module` hierarchies. You can string together multiple different Triton kernels across various submodules, and Kernel Lens will trace the entire computational graph flawlessly.

### 4. Dynamic Multi-Output Support
Unlike primitive compilers that assume a single output tensor, Kernel Lens dynamically dry-runs your network to count outputs and infer exact datatypes. It easily supports kernels that return multiple tensors of varying types (e.g., a `float32` matrix and an `int64` indexing array).

### 5. Zero-Copy TensorRT VRAM Mapping
Kernel Lens bypasses CPU bottlenecks. It hooks directly into PyTorch's CUDA memory allocator, formats the memory layouts safely (enforcing `.contiguous()` checks), and maps the VRAM pointers directly into TensorRT's execution context for instant, zero-overhead execution.

### 6. Cold-Start Persistence
Don't waste time recompiling. Kernel Lens caches your compiled `.so` plugins and `.engine` files. You can load a highly optimized model directly from the cold cache in production:
```python
# Instantly loads previously compiled C++ plugins
production_model = kl.load("my_fast_matmul") 
output = production_model.run((A, B), backend="tensorrt")
```

### 7. Fail-Fast Environment Diagnostics
Kernel Lens respects your time. Before initiating complex graph tracing, the internal diagnostic tool verifies your system environment (`nvcc`, `g++`, TensorRT headers, ONNX Runtime execution providers). If a dependency is missing, it fails instantly with actionable installation advice.

## 🛠️ Advanced Usage & Debugging

If you encounter silent failures or want to see exactly what C++ math is being generated and executed on the GPU, Kernel Lens includes an aggressive native C++ debugging suite. 

Enable it via environment variables before running your script:
```bash
KERNEL_LENS_DEBUG=1 python my_script.py
```
This injects `printf` tripwires directly into the compiled C++ shared libraries, outputting the calculated execution grids and exact VRAM memory addresses right before `cuLaunchKernel` fires.