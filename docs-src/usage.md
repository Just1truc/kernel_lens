# Usage & API

## Installation

```bash
pip install kernel-lens            # core compiler
pip install kernel-lens[ort]       # + ONNX Runtime
pip install kernel-lens[trt]       # + TensorRT
pip install kernel-lens[all]       # both backends
```

**Requirements:** Linux x86-64, Python ≥ 3.10, CUDA toolkit ≥ 12.0, PyTorch ≥ 2.0,
Triton ≥ 2.1, and `nvcc` + `g++` on `PATH` (both backends compile C++ at
`compile()` time).

## Quickstart

```python
import torch, triton, triton.language as tl
import kernel_lens as kl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

class Add(torch.nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        n = x.numel()
        add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
        return out

x = torch.randn(4096, device="cuda"); y = torch.randn(4096, device="cuda")
compiled = kl.compile(Add(), (x, y), name="vector_add", backends=["onnx", "tensorrt"])

out_ort = compiled.run((x, y), backend="onnx")        # ONNX Runtime C++ plugin
out_trt = compiled.run((x, y), backend="tensorrt")    # TensorRT C++ plugin
```

## API reference

### `kl.compile`

```python
kl.compile(
    model: torch.nn.Module,
    inputs: tuple,                      # representative sample inputs (run on GPU)
    name: str = "custom_model",         # cache key; also names artifacts
    backends: list[str] = ["onnx", "tensorrt"],
    interaction_handler=None,           # custom I/O classifier (optional)
    verbose: bool = False,
) -> CompiledModel
```

Traces the model, exports ONNX with injected custom nodes, synthesizes and
compiles the requested C++ plugins, and returns a `CompiledModel`. Artifacts are
written to the cache directory (see below).

### `kl.load`

```python
kl.load(name: str) -> CompiledModel
```

Reconstructs a `CompiledModel` from the cache without recompiling. Use this for
production cold starts. Raises `FileNotFoundError` if no cached plugins exist.

### `CompiledModel.run`

```python
compiled.run(inputs: tuple, backend: str = "onnx") -> torch.Tensor | tuple
```

Executes with zero-copy VRAM binding. `backend` must be one of the backends
requested at compile time. Returns a single tensor or a tuple of tensors,
matching the module's eager output structure.

### `kl.extract_libs`

```python
kl.extract_libs(compiled, dest_dir, libs=None)
```

Copies deployable artifacts (`<name>.onnx`, `libtriton_ort_plugins.so`,
`libtriton_trt_plugins.so`, and `<name>.engine` if built) from the hidden cache
into a normal directory.

```python
kl.extract_libs(compiled, "./deploy", libs=["onnx", "tensorrt"])
```

### Verbosity

```python
import kernel_lens as kl
kl.set_verbose(True)         # or set KERNEL_LENS_DEBUG=1 / KL_VERBOSE=1
```

Verbose mode prints the manifest contents, AST classification decisions, grid
expressions and generated launch geometry.

## Cache layout

```text
$KERNEL_LENS_CACHE_DIR/            (default: ~/.kernel_lens_cache)
└── <name>/
    ├── <name>.onnx                ONNX graph with triton_custom nodes
    ├── saved_args.pt              captured scalar/argument metadata
    ├── ort_plugins/
    │   ├── <kernel>Op.{cu,h}
    │   ├── register_ops.cpp
    │   └── libtriton_ort_plugins.so
    └── trt_plugins/
        ├── <kernel>Plugin.{cu,h}
        ├── register_plugins.cpp
        └── libtriton_trt_plugins.so
```

!!! tip "Cold start"
    Backends are re-detected from the presence of the `.so` files when you call
    `kl.load(name)`. A TensorRT `.engine` is built lazily on the first
    `run(..., backend="tensorrt")` call and then cached.

## Supported operators

KernelLens supports the patterns Triton itself supports, with these conveniences:

- Pointers that are only read, only written, or **read and written in place**.
- `int` / `float` **host scalars** that may change every call.
- Arbitrary **strides** (as long as the contiguous inner dimension is a multiple
  of 8, which the compiler validates).
- **Dynamic launch grids** computed from tensor shapes.
- `tl.constexpr` values baked into PTX.
- dtypes: `fp32`, `fp16`, `bf16`, `fp8_e4m3fn`, `fp8_e5m2`, `int8`, `int32`,
  `int64`, and packed `int4`.
- Kernels that delegate to nested Triton helper functions.

!!! warning "Backend caveats"
    In-place mutation and INT8 dequantisation are currently correct **only** on
    the ONNX Runtime backend (see [Verification](verification.md)). Prefer ORT for
    those kernels.

## Environment variables

| Variable | Effect |
| :--- | :--- |
| `KERNEL_LENS_CACHE_DIR` | Overrides the cache root directory |
| `KERNEL_LENS_DEBUG` / `KL_VERBOSE` | Enables verbose tracing output |
| `TENSORRT_INCLUDE_DIR` | Explicit TensorRT C++ header path for plugin builds |

## Troubleshooting

- **`nvcc` not found** — install the CUDA toolkit and put `nvcc` on `PATH`.
- **TensorRT headers missing** — `pip install tensorrt-cu12-libs`, or set
  `TENSORRT_INCLUDE_DIR`, or copy `NvInfer.h`/`NvInferPlugin.h` to
  `~/tensorrt_headers/`.
- **`cuda_runtime_api.h` not found on conda** — resolved by KernelLens 1.1.9+
  via `targets/<triple>/include` resolution; upgrade if you hit this.
- **Alignment error** — the physical contiguous inner dimension must be a
  multiple of 8. Pad your tensor or choose a different tile.