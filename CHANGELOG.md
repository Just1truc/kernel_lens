# Changelog

Notable changes to `kernel-lens`. This project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `kl.compile(..., dynamic_shapes=True)` marks the leading (batch) dimension of
  every tensor input dynamic, so one engine can serve more than the traced batch
  via a TensorRT optimization profile. Off by default: the published benchmarks
  measure a shape-specialised engine, and turning it on would invalidate them.

### Fixed

- **Plugins baked the traced output dims.** Both generators resolved output
  dimensions through `int(dim)`, so a dynamic graph got a plugin that reported
  the traced batch. Only dims in `manifest.dynamic_axes` are now read from the
  tensor at run time.
- **The ORT runtime re-used the previous call's output buffers.** A cached
  `IOBinding` keeps its OrtValues, and ORT rejects them once the output shape
  moves ("pre-allocated output OrtValue ... has shape {1,512} but the computed
  output shape for this run is {8,512}"). A dynamic graph now binds from
  scratch per call, and resolves output shapes from the live input rather than
  the traced output meta.
- **TensorRT output dims were never inferred.** With a dynamic input the
  execution context returns `-1` for the outputs until asked, and the fallback
  handed back the *traced* shape. `infer_shapes()` now runs when the graph is
  dynamic.
- **The optimization profile took its `opt` from whichever call built the
  engine first**, so a first run at batch 1 produced `max = 4` and the engine
  then refused every batch above it. `opt` now comes from the traced shapes,
  which are persisted alongside the rest of the compile metadata.
- The profile's `max` was a flat `max(8192, opt*4)`, which for a *batch* dim
  means building for a thousand-fold range. It is now `4x` the traced dim by
  default, overridable with `dynamic_max`.

### Known limitation

- `dynamic_shapes=True` is **refused** when the kernel's launch geometry was
  frozen before it reached the plugin. The symbolic capture itself works: the
  tracer runs a real pass and then a `make_fx(tracing_mode="symbolic")` pass, and
  Pass 2 genuinely produces `Grid ASTs: (((s75*s96 + 255)//256),)` with
  `x_ptr: shape=(s75, s96)`. `extract_manifests` merges it
  (`tracer.py:241`). Two things then lose it:

  1. **The SymInts are mutated in place by the launch that produced them.**
     `patched_run` captures the grid and then lets the launch proceed; Triton
     needs an `int`, PyTorch resolves the symbol via its hint and specialises
     the node to a constant -- the same object the manifest holds. Reading the
     same manifest gives `grid_expr_strs = ('((s75*s96 + 255)//256)',)` when
     snapshotted at capture time, and `_sym_grid_asts = (16,)` afterwards.
     Every consumer therefore sees `16`: the generators emit
     `unsigned int grid_x = std::max(1u, (unsigned int)(16));`, the scalar
     becomes `m_n(4096)`, and `grid_cxx_exprs` collapses to `['16','1','1']`.
  2. **The substitution uses symbol ids as dimension indices.** Both generators
     do `re.sub(r'\bs(\d+)\b', ... dims.d[\1] ...)`, but `s75` is a shape-env
     symbol id, not a dim index; it would read dimension 75 of a 2-D tensor and
     the bounds guard silently falls back to 1. The manifest already computes
     the correct mapping in `_transpile_symints_to_cxx()` (keyed by expression
     string, with the real dim index) and stores it in `grid_cxx_exprs` -- but
     neither generator consumes that, and `cxx_expr` is never read at all.

  The fix is correspondingly small: snapshot the expression string and the
  symbol->dim map at capture time, substitute by name in both generators, and
  consume `cxx_expr` for scalars. Until then `_check_dynamic_supported()` raises
  rather than returning wrong numbers or reading out of bounds, and
  `tests/test_conformance.py` pins the refusal (`dynamic_batch_guard`,
  `dynamic_grid_guard`).

## [1.2.1] — 2026-10-23

Everything below comes out of the real-checkpoint evaluation on an
A100-SXM4-80GB. Each fix was reproduced on a real OPT model before it landed.

### Fixed

- **TensorRT built FP16/BF16 engines with the wrong precision, and then failed
  to build them at all.** TensorRT keeps plugin layers in FP32 unless the
  matching `BuilderFlag` is set. Without it the builder inserts a cast on the
  plugin's inputs, and the plugin's dtype guard — which mirrors the kernel's
  signature — rejects them, so the build dies with *"could not find any
  supported formats consistent with input/output data types"*. The precision is
  now derived from the ONNX graph, and read from the **initializers and
  `value_info` as well as the inputs and outputs**: a language model's graph
  inputs are token ids (`int64`), so the input list alone says nothing about the
  precision.

- **ONNX graphs above 2 GB could not be parsed.** Above the protobuf limit the
  weights spill to external files beside the model, and TensorRT's parser can
  only resolve those relative locations when it is told which directory they
  live in. The ONNX path now uses `parser.parse_from_file()` when the binding
  provides it, falling back to the in-memory path otherwise.

- **The `nvcc` host compiler was hardcoded.** CUDA toolkits before 12.9 declare
  `rsqrt`/`sinpi`/… without `noexcept`, which GCC ≥ 13 rejects with *"has a
  different exception specifier"*; g++-11, -12 and -13 all fail against CUDA
  12.8. The choice is now environment-specific via the `KERNEL_LENS_CCBIN`
  override, defaulting to the previous `g++-13`/`g++` probe.

### Added

- **`KERNEL_LENS_STREAM_READER`** — opt-in file-backed `IStreamReaderV2` so that
  engines above 2 GB can be deserialized straight from disk instead of being
  read into a memory buffer first. Off by default: TensorRT's Python binding
  dispatches `read(num_bytes, destination)` in the *reverse* of its documented
  order, and writing through that pointer is not yet reliable. Engines below
  2 GB take the memory path regardless.
- `tests/benchmark_real_model.py` — serves a real checkpoint with every decoder
  `LayerNorm` swapped for the Triton RMSNorm kernel, and grades eager,
  `torch.compile`, ONNX Runtime and TensorRT per precision, reporting top-1
  (greedy-token) agreement alongside the mean and max logit deviation.
- `tests/measure_codegen.py` — per-file line counts of the generated C++.
- `scripts/remote_setup.sh` and `scripts/remote_bench_all.sh` — unattended
  provisioning and full-matrix sweep on a fresh CUDA host.

### Corrected claims

- The generated C++ for one fused RMSNorm kernel on both backends is **1,137
  lines across six files**, measured, not the ≈1,329 previously stated.
- Accuracy is reported per precision rather than blended: fp32 keeps the same
  greedy token on ≥ 98.8 % of positions, fp16 under TensorRT on 93.4–97.9 %.
- Max absolute logit deviation alone is misleading on a 50 k-way output
  (`OPT-2.7B` fp16 under TensorRT peaks at 1.88 against a logit scale of 16.3
  while the mean is 4.0e-02), so it is always reported next to the mean and the
  top-1 agreement.

## [1.2.0] — 2026-10-20

### Added

- `tests/test_conformance.py` — fourteen module patterns (two kernels per graph,
  the same kernel launched twice, multiple outputs, dynamic scalars, strided and
  dynamic inputs, loops, conditionals, nested helpers, in-place mutation, 3-D
  grids, autotune, fp16/bf16), all bit-exact against native Triton on both
  backends.
- `tests/test_alignment_guard.py`, `tests/test_inplace_dtypes.py` and
  `tests/test_unit_cpu.py`.
- Single-page documentation site with the folding, parity, latency and
  end-to-end results.

### Fixed

- TensorRT plugin builder registration; end-to-end decoder benchmark corrected.
- Removed top-level `onnxruntime` imports so the core package imports without an
  engine installed, and added CPU skip guards so the suite runs on CI runners
  without a GPU.

[1.2.1]: https://github.com/Just1truc/kernel_lens/releases/tag/v1.2.1
[1.2.0]: https://github.com/Just1truc/kernel_lens/releases/tag/v1.2.0
