# Changelog

Notable changes to `kernel-lens`. This project follows
[Semantic Versioning](https://semver.org/).

## [2.0.0] — 2026-10-24

TensorRT 8.6 through 11, chosen automatically.

### Added

- **`IPluginV3` emitter** (`backends/trt_v3_gen.py`). TensorRT 10 deprecated the
  entire `IPluginV2*` hierarchy and TensorRT 11 removed it, so a plugin built
  against the legacy interfaces cannot be loaded there at all. The new emitter
  produces the same semantics against `IPluginV3` plus its three capability
  interfaces, and serialises through `PluginFieldCollection` instead of a
  hand-written byte buffer — `getFieldsToSerialize()` writes the scalars and the
  namespace, `createPlugin()` reconstructs from them on engine load.

- **Automatic interface selection.** `plugin_api` defaults to `"auto"` and picks
  from the environment:

  | TensorRT | `IPluginV2*` | `IPluginV3` | emitted |
  |---|---|---|---|
  | < 10 (8.6, 9.x) | present | absent | `v2` |
  | 10.x | deprecated | present | `v3` |
  | 11+ | **removed** | present | `v3` |

  An explicit `plugin_api="v2"` / `"v3"`, or `KERNEL_LENS_PLUGIN_API`, overrides
  it; an impossible request fails with the reason.

- **Header provisioning** (`utils/trt_headers.py`). The pip TensorRT wheels ship
  no headers — verified: the package contains only `__init__.py` and `plugin/`.
  The set matching the installed runtime is now fetched from the TensorRT OSS
  repository (`release/<major>.<minor>`) and cached under
  `~/.kernel_lens_headers/`, atomically and validated. `TENSORRT_INCLUDE_DIR`
  overrides; `KERNEL_LENS_FETCH_HEADERS=0` disables. This is what makes future
  TensorRT releases work without a package update.

### Fixed

- **The V3 plugin's `initialize()` was incomplete.** It loaded the PTX with a
  bare `cuModuleLoadData` and nothing else, so every kernel failed with
  `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` (222) and the engine silently produced
  uninitialised output. It now retains the primary CUDA context, retries with a
  clamped `.version 9.0` and then an older `.target`, and sets
  `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` — all of which the
  `IPluginV2DynamicExt` path already did.
- **Header/runtime major mismatch is now reported, not crashed.** A plugin built
  against 10.x headers and loaded into an 11.x runtime segfaulted inside
  TensorRT with no diagnostic. It now raises with the mismatch, the paths
  involved, and the two ways out.

### Verified

TensorRT 10.16.1.11 with bundled headers, TensorRT 11.3.0.99 with fetched
headers, and `plugin_api="v2"` forced: **16/16 conformance on each**, plus every
parity suite (`0.00e+00` on the research kernels, in-place, quantized GEMMs and
edge cases; `9.536743e-07` residual+RMSNorm; `4.768372e-07` Mamba) with values
identical across the two emitters.

## [1.3.0] — 2026-10-23

Shape-generic engines. `kl.compile(..., dynamic_shapes=True)` produces a single
engine that serves any leading-dimension size within a declared range, instead of
one engine per shape.

Verified on two architectures — RTX 5060 Ti (Blackwell, `sm_120a`) and
A100-SXM4-80GB (Ampere, `sm_80`) — with one engine compiled at batch 8 serving
batches 1, 2, 4, 8 and 16, and one compiled at 524288 elements serving
256, 512, 2048 and 4096, bit-exact (`0.00e+00`) on both ONNX Runtime and
TensorRT. The full conformance suite is 16/16 on both.

### Added

- `dynamic_shapes=True` and `dynamic_max` on `kl.compile()`. Off by default: the
  published benchmarks measure a shape-specialised engine. `dynamic_max` bounds
  the optimization profile (default `4x` the traced size); TensorRT builds
  kernels spanning `min..max`, so a needlessly wide range costs build time and
  can cost per-shape performance.
- Two conformance cases that exercise the capability rather than assuming it:
  `dynamic_batch_sweep` and `dynamic_grid_sweep` compile once and run every shape
  through that single engine.

### Fixed

Two independent defects, both of which had to be fixed for the feature to work.

**The symbolic tracing pass was working and its output was discarded.**
`extract_manifests()` runs a real pass and then a
`make_fx(tracing_mode="symbolic")` pass, and merges them — Pass 2 genuinely
captures `((s75*s96 + 255)//256)` and `shape=(s75, s96)`. But `SymNode` exposes
two accessors and they are not the same:

```
node._expr = ((s75*s96 + 255)//256)   # raw symbol, intact
node.expr  = 16                       # shape_env.replace(), cached
```

`expr` substitutes the values the shape env has recorded, so it folds the
expression to the constant. Every consumer read `.expr`, so the launch grid and
the shape-derived scalars were frozen at the traced shape. Note this is
*correct* for a shape-specialised engine — which is why nothing ever broke
before. Fixed by reading `node._expr` (via `raw_sym_expr`), carrying Pass 2's
symbolic shapes through the merge, and giving the manifest one translation with a
per-backend dimension accessor.

**The tracer picked the PTX for the wrong dtype.** `tracer.py` extracts the PTX
by walking Triton's compile cache and taking the first hit:

```python
for key_or_dev, value in reversed(list(caches.items())):
    for key_tuple, compiled in reversed(items_to_check):
        if hasattr(compiled, 'asm') and 'ptx' in compiled.asm:
            ptx = compiled.asm['ptx']
```

The cache holds one variant per argument specialisation, keyed
`"[('*fp32','D'), ..., ('constexpr',128)]"`. A re-used entry keeps its original
dict slot, so an fp32 variant compiled earlier sits *before* a bf16 one compiled
later and `reversed()` returns the **bf16 PTX for an fp32 launch** — which reads
fp32 memory 16 bits at a time and silently covers half the tensor. Fixed by
matching the variant against the launch's own argument dtypes
(`_TRITON_DTYPE_MARK`), with a positional fallback.

Also fixed while wiring the feature end to end:

- `torch.onnx.export` had no `dynamic_axes`, so no `-1` ever reached the network
  and the optimization-profile code never ran.
- Plugins emitted `exprBuilder.constant(N)` / literal output dims from the traced
  shape; they now read the dynamic dims from the tensor at run time.
- TensorRT output dims were never inferred (`infer_shapes()`), so a dynamic graph
  returned `-1` and the fallback handed back the *traced* shape.
- The optimization profile took `opt` from whichever call happened to build the
  engine first, giving `max = 4` when the first run was batch 1; `opt` now comes
  from the traced shapes.
- The ORT runtime re-used the previous call's `IOBinding`, whose OrtValues ORT
  rejects once the output shape moves. A dynamic graph now binds per call and
  resolves output shapes from the live input.
- `translate_symint_to_cxx` emitted `**` as `^` — bitwise XOR in C++ — and used a
  `re.sub` replacement string that corrupted the substitution.
- The ORT op addressed its inputs by a running counter over `manifest.arguments`;
  indices now come from the node's own input list, which is what
  `GetInputTypeCount()` reports and what ORT actually presents.
- Shape-derived scalars (`n = x.numel()`) are computed from `dim_values` in the
  plugin and recomputed from the live shapes in the runtime, instead of being
  baked at trace time.

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
