# Design notes: a TensorRT-LLM backend

Reconnaissance for extending KernelLens to TensorRT-LLM. Written before any
implementation, to pin down what is actually involved.

## 1. The plugin ABI is already the right one

TensorRT-LLM plugins *are* TensorRT plugins:

| Piece | TensorRT-LLM | KernelLens today |
|---|---|---|
| Base class | `nvinfer1::IPluginV2DynamicExt` (its `PluginBase` derives from it) | `nvinfer1::IPluginV2DynamicExt` |
| Creator | `nvinfer1::IPluginCreator` | `IPluginCreator` subclass |
| Registration | `REGISTER_TENSORRT_PLUGIN` / bulk `initTrtLlmPlugins(logger, "tensorrt_llm")` | explicit `register_triton_plugins_explicit()` |
| Packaging | `.so` linked into the process | `libtriton_trt_plugins.so` |

So the generated plugin needs only cosmetic changes (optionally deriving from
`PluginBase`, registering under a TRT-LLM namespace). **Plugin generation is not
the hard part.**

## 2. TRT-LLM does not currently ship an automatic plugin generator

`tensorrt_llm/tools/plugin_gen/` exists on `main` but contains only an empty
`__init__.py` — a stub. NVIDIA's automatic-plugin tooling lives in
**Torch-TensorRT** ("Automatically Generate a Plugin for a Custom Kernel"),
which requires the user to *declare* the kernel interface.

KernelLens's differentiator stands: it *recovers* the interface (PTX entry
signature, symbolic grid, argument taxonomy) rather than requiring a
declaration.

## 3. The real work is graph injection, not the plugin

TRT-LLM does not consume arbitrary ONNX. It builds its own engine from a
checkpoint (`convert_checkpoint.py` → `trtllm-build`) or from its Python
`Module`/`functional` API. KernelLens's ONNX-injection pass therefore has no
direct equivalent, and this is where all the effort goes.

Candidate injection points:

- **A. `network.add_plugin_v2(...)`** inside TRT-LLM's Python build path —
  substitute a `Module` that calls the generated plugin.
- **B. TRT-LLM's Triton-plugin path** — emit a TRT-LLM-shaped plugin and
  register it beside `initTrtLlmPlugins`.
- **C. Keep ONNX** — the current backend; not TRT-LLM.

## 4. Hard constraints, stated up front

1. **Attention is out of scope.** TRT-LLM's `gpt_attention` owns paged KV
   cache, MQA/GQA, RoPE and in-flight batching. A custom Triton attention
   kernel cannot slot into that without reimplementing the paged-cache
   contract. The MVP covers **non-attention kernels** only: RMSNorm, SwiGLU,
   activations, fused elementwise. That is exactly the Liger-Kernel use case.
2. **Weight conventions.** TRT-LLM has its own `Weights` loading and weight-only
   groupwise quantization. KernelLens plugins take weights as tensor inputs, so
   this is a mapping problem, not a blocker.
3. **Version churn.** TRT-LLM 1.0+ moved toward a PyTorch-native runtime with
   TensorRT used selectively, so "TRT-LLM support" differs by version. Pin one
   and state it.

## 5. Environment conflict (blocks work on the current instance)

Installing `tensorrt-llm` 1.2.1 resolves to a dependency set that **downgrades
the working environment**:

| Package | Now (working) | TRT-LLM wants |
|---|---|---|
| torch | 2.12.1 | 2.9.1 |
| transformers | 5.20.2 | 4.57.3 |
| triton | 3.7.1 | 3.5.1 |
| tensorrt | 10.16.1 | 10.14.1 |
| numpy | 2.x | 1.26.4 |
| onnx | latest | 1.19.1 |

Installing it in place would invalidate every result gathered so far. It must go
in an **isolated virtualenv**, and the KernelLens test suite has to be re-run
inside that venv (different torch/triton, so the earlier results do not carry
over).

The PyPI `tensorrt-llm` package is a ~10 KB stub; the real source is the
GitHub repository.

## 6. Proposed MVP

Reuse ~90% of the existing pipeline:

1. **Manifest recovery** — unchanged (PTX entry signature, symbolic grid,
   argument taxonomy).
2. **Plugin C++** — unchanged, plus an optional `PluginBase`-derived variant and
   a TRT-LLM namespace registration entry point.
3. **New `trtllm` backend** that emits:
   - the plugin `.so` and its registration function, and
   - a TRT-LLM `Module` (route A) or a graph-rewrite pass that substitutes the
     recovered op for the original subgraph.
4. **Scope guard** that refuses attention-shaped kernels with a clear message.

## 7. Sequencing

1. Create an isolated venv on the instance; install TRT-LLM; confirm a stock
   TRT-LLM build runs a small model. *(Enabling step; ~10 GB, 20-40 min.)*
2. Re-run the KernelLens suite inside that venv to establish a baseline.
3. Inspect TRT-LLM's `Module`/plugin registration API at the pinned version.
4. Implement the `trtllm` backend and the injection shim.
5. Test on a non-attention model (OPT + Triton RMSNorm is a good candidate).
6. Re-run the full suite in the original environment to confirm no regression.

This is a multi-session build, not a same-day change. Steps 1-3 are
prerequisites that cannot be skipped, and step 1 alone invalidates the current
environment.
