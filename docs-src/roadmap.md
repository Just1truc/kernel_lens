# Roadmap

KernelLens today is a working compiler for **single-GPU** Triton kernels on
**ONNX Runtime** and **TensorRT**, verified on Blackwell `sm_120a`. The items
below are ordered by how much they would expand that envelope. Items marked
**fix** are known defects from [Verification](verification.md); the rest are new
capabilities.

## Correctness first

| Priority | Item | Notes |
| :--- | :--- | :--- |
| **fix** | TensorRT in-place aliasing | The TRT plugin must agree with the engine on the input/output buffer contract. |
| **fix** | TensorRT INT8 dequant GEMM | Align TRT tensor-desc handling with the verified ORT path. |
| **fix** | True 16-byte alignment guard | Implement the documented staging-buffer fallback in generated C++, or drop the claim. |
| **fix** | Persist output metadata for `load()` | `output_shapes` / `strides` / `dtypes` are not round-tripped through the cache today. |
| **fix** | Multi-kernel `saved_args` | `saved_args.pt` currently captures only the last kernel's node inputs. |

## Compiler coverage

1. **Auto-tuning dispatch.** Fold multiple `@triton.autotune` variants into one
   plugin with a C++ dispatch table keyed on live shapes, instead of compiling a
   single concrete PTX artifact.
2. **CUDA Graphs.** Capture the zero-copy launch sequence into a graph to remove
   residual per-launch host overhead.
3. **Symbolic-grid formalisation.** Replace the regex over `SymInt` strings with a
   real expression visitor, and add out-of-bounds proofs (see below).
4. **Multi-variant PTX caching.** Profile candidate shapes in Pass 1 and emit a
   module handle per variant.

## Hardware and platforms

5. **Blackwell cluster launches.** Synthesize `CUlaunchAttribute` cluster
   descriptors and `cp.async.bulk` via a dedicated `blackwell_gen.py`.
6. **Micro-scaling formats.** Native NVFP4 / MXFP6 pack/unpack in generated C++.
7. **Non-CUDA targets.** Apple Metal (MSL) and WebGPU (WGSL) backends.
8. **ROCm / HIP.** A second code generator for AMD C++ engines.

## Distributed and verification

9. **Embedded NCCL.** Intercept `torch.distributed` collectives and emit plugins
   with communicator handles for tensor/pipeline parallelism.
10. **SMT-based safety proving.** Feed PTX and grid expressions to a solver
    (e.g. Z3) to prove `0 ≤ offset(x,y,z) < buffer_size` before compilation.

## Documentation and process

11. **CI on GPU runners.** Move from CPU-only lint to a real CUDA CI lane so the
    verification matrix is maintained automatically.
12. **Pin the toolchain contract.** Declare the supported CUDA/TensorRT/PyTorch
    ranges and test the build shims against each.

Contributions welcome — see the repository's issue tracker.