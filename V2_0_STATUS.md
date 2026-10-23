# KernelLens 2.0 — IPluginV3 and TensorRT-LLM

Status: **V3 emitter implemented and passing.** Full conformance is 16/16 with
`plugin_api="v3"`, every parity suite passes with values identical to V2, and V2
itself is unregressed at 16/16. Nothing pushed.

---

## 1. What was built

### IPluginV3 emitter — `kernel_lens/backends/trt_v3_gen.py`

Emits the same plugin semantics as `trt_gen` (which targets
`IPluginV2DynamicExt`) against the interfaces TensorRT 10 introduced and
TensorRT 11 requires:

| Interface | Methods implemented |
|---|---|
| `IPluginV3` | `getCapabilityInterface`, `clone` |
| `IPluginV3OneCore` | `getPluginName`, `getPluginVersion`, `getPluginNamespace` |
| `IPluginV3OneBuild` | `getNbOutputs`, `getOutputDataTypes`, `getOutputShapes`, `supportsFormatCombination`, `configurePlugin`, `getWorkspaceSize` |
| `IPluginV3OneRuntime` | `onShapeChange`, `enqueue`, `attachToContext`, `getFieldsToSerialize` |
| `IPluginCreatorV3One` | `createPlugin`, `getFieldNames`, `getPluginName/Version/Namespace` |

Serialisation moves from the V2 raw byte buffer to `PluginFieldCollection`:
`getFieldsToSerialize()` returns the scalars plus the namespace as
`PluginField`s, and `createPlugin()` reconstructs the object from them on engine
load. That deletes the hand-written `serialize`/`deserialize` pair.

Registration uses the non-deprecated
`IPluginRegistry::registerCreator(IPluginCreatorInterface&, ns)` overload —
`IPluginCreatorV3One` derives from `IPluginCreatorInterface`, so the V3 path
emits no deprecation warning (the V2 path does, on `registerCreator` and on the
legacy `IPluginV2` bases).

### Shared execution path (refactor, verified byte-identical)

`trt_gen.py` now exposes three static helpers that both emitters call, so a V3
plugin and a V2 plugin run the same code:

- `build_shared_body(manifest, block_size)` — PTX entry name, `kernelParams`
  assembly, output-type dispatch
- `supports_format_body(manifest, tensor_prefix="")` — the
  `supportsFormatCombination` body. `IPluginV3` hands it a
  `DynamicPluginTensorDesc` whose fields live under `.desc`, so V3 passes
  `".desc"`; the default reproduces the V2 text exactly.
- `trt_datatype(dtype_obj)`

The refactor was proved behaviour-preserving by generating the V2 plugin sources
before and after and comparing them byte for byte:
`SHA256 735dd7d8e5b9de4219017d83094e662c12a9cf2b0b7355a0d086554dacd4edb4`,
unchanged across all three extractions.

### API and wiring

```python
kl.compile(model, sample, backends=["onnx", "tensorrt"], plugin_api="v3")
# or KERNEL_LENS_PLUGIN_API=v3 for a whole session
```

The V3 sources are written into the same `trt_plugins/` directory, so the
existing build and load path is reused unchanged; `engine.py` now accepts either
registration symbol (`register_triton_plugins_explicit`,
`register_triton_v3_plugins_explicit`).

### TensorRT-LLM

Not started. [`docs/assets/data/a100/trtllm_backend_design.md`](docs/assets/data/a100/trtllm_backend_design.md)
is the reconnaissance, and its conclusions still hold: TRT-LLM plugins *are*
TensorRT plugins, so generation is not the hard part — graph injection is, since
TRT-LLM builds its engine from a checkpoint rather than consuming arbitrary ONNX.
Attention is out of scope (TRT-LLM's `gpt_attention` owns the paged KV cache), so
the MVP is non-attention kernels. Installing `tensorrt-llm` also downgrades
torch/transformers/triton and so needs an isolated venv, invalidating the current
measurements inside it.

---

## 2. What is verified

| Claim | Evidence |
|---|---|
| V3 plugins compile | all 16 conformance cases: `nvcc` compiles every `.cu`, `g++` compiles `register_plugins.cpp` |
| The engine builds and loads them | no build or load errors; inference returns shaped tensors |
| The V2 path is untouched | full suite **16/16**, and generation byte-identical to the pre-refactor baseline |
| Shared code is genuinely shared | V2 and V3 enqueue bodies come from the same helper |

## 3. What is not

**V3 runtime correctness: 2/16 pass, 14/16 wrong on TensorRT.** The ONNX path is
unaffected (all `0.00e+00`).

```
[PASS] inplace            onnx=0.00e+00 tensorrt=0.00e+00
[PASS] bf16 (TRT only)    tensorrt=0.00e+00
[FAIL] two_kernels        onnx=0.00e+00 tensorrt=1.47e+01
[FAIL] same_kernel_twice  onnx=0.00e+00 tensorrt=2.20e+01
... 12 more, all on tensorrt
```

Failure mode on a single kernel (`x * 2` then `x * 3`, fp32, 4096 elements):

```
ref[:5]  = [-5.548, -2.5521, -15.8631, 0.8711, -0.7252]
out[:5]  = [-6.8708, -3.4287, -3.906, 12.4647, -8.4195]
out == 2x: False   out == 6x: False   zeros: False
```

So the kernel runs and returns shaped output, but the values are uncorrelated
with the input — it reads or writes the wrong buffer.

### Ruled out

- **Output shapes.** `getOutputShapes` is byte-identical for a passing
  (`bf16`) and a failing (`same_kernel_twice`) case, and the fix for the
  unconditional `outputs[0] = inputs[0]` fallback did not change any result.
- **Registration.** `nvinfer1::getPluginRegistry` (should be global
  `::getPluginRegistry`) and namespace-scoped static creators — both were real
  bugs, both fixed, neither changed the outcome.
- **Compilation.** Everything compiles; there is no partial-build artefact.
- **The shared enqueue body.** Unchanged from the V2 path, which is 16/16.
- **`PluginFieldType::kFLOAT`** (should be `kFLOAT32`) and the float literal
  `2f` (should be `2.0f`) — both fixed, neither changed the outcome.

### Next

The remaining suspect is the boundary between TensorRT and the plugin rather than
the generated code: what `inputs[]`/`outputs[]` V3 hands `enqueue`, and whether
`attachToContext`'s clone is the object actually enqueued (the clone resets the
CUDA handles, so a stale `mKernel`/`mModule` on the original could bind the wrong
function). Instrumenting `enqueue` to log the pointer values and the launched
grid, and comparing against the V2 plugin on the same graph, should separate
those two in one run.

---

## 4. Files touched

```
kernel_lens/backends/trt_v3_gen.py          (new — the V3 emitter)
kernel_lens/backends/trt_gen.py             (shared-body extraction, static helpers)
kernel_lens/compiler/core.py                (plugin_api, env override)
kernel_lens/runtime/engine.py               (either registration symbol)
```

Nothing pushed. The 1.3.0 release on PyPI is unaffected by any of this.
---

## 5. Session 2 — instrumentation results

Added a device-to-host dump inside the generated `enqueue` (gated by `KL_DBG`)
and a one-line entry print. What it showed:

```
[V3-ENTER] scale_kernel_v3_one__0 enqueue in=0x7567e360c000 out=0x7567e3614400 n=4096
single kernel: maxdiff=6.631e+00  out==2x:False  |out|max=2.0000
```

Three things follow.

1. **`enqueue` is called.** The plugin is registered, instantiated, and invoked
   with a real input pointer, a real output pointer and the correct scalar
   (`n=4096`). So registration, `createPlugin`, `attachToContext` and the
   capability dispatch are all working.
2. **The enqueue body is not the problem.** Diffing the generated V2 and V3
   `enqueue` for the same manifest shows them functionally identical — same
   `kernelParams` assembly, same grid, same scalar handling. Only formatting and
   the debug line differ.
3. **The sharpest clue:** for a kernel that computes `x * 2`, `|out|max` is
   **2.0** — exactly the value of the scalar `alpha` — while `x * 2` would peak
   near 8. The output is not a scaled copy of the input; its largest magnitude is
   the scalar itself.

That points at the argument-slot mapping: something in `kernelParams` is landing
the scalar where a tensor belongs (or vice versa). Since the assembly code is
byte-identical to the working V2 path, the difference has to be in **what
TensorRT hands `enqueue`** — the order, count or contents of `inputs[]` and
`outputs[]` under `IPluginV3OneRuntime` versus `IPluginV2DynamicExt`.

### Next step (one run)

Log every slot of `kernelParams` at launch time in both emitters — the PTX slot
index, the argument it was mapped to, and the pointer value — and diff the two
for the same manifest. That will show directly whether V3 is being handed a
different tensor ordering. A secondary check: print `nbInputs`/`nbOutputs` as
`enqueue` receives them and compare against what the manifest expects.


---

## 6. Session 3 — root cause and fix

The output-buffer investigation resolved to something much simpler than the
binding theory. Instrumenting both sides of the boundary showed the addresses
**agree**:

```
[PY-OUT] output_0 -> 0x798645614400
[V3-OUT] scale_kernel_v3_one__0 outputs[0]=0x798645614400 inputs[0]=0x79864560c000 n=4096
```

So the plugin's output pointer *is* the bound tensor. The earlier sentinel test
had been invalid: the sentinel was written with a synchronous `cuMemcpyHtoD`
after an already-queued async kernel, so the kernel simply overwrote it.

Logging the launch geometry never printed, which meant the function was returning
before it. The error path said why:

```
TRT_V3_INIT_ERROR: cuModuleLoadData failed with code 222
TRT_V3_ENQUEUE_ERROR: initialize failed or mKernel is null
```

**222 is `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.** The kernel never loaded, so it
never ran, and the output was simply whatever the tensor already held.

The cause: my V3 `initialize()` was a naive two-call version. The V2 one does
three things it omitted, and two of them are load-bearing:

1. **Retain and set the primary CUDA context** (`cuDevicePrimaryCtxRetain` +
   `cuCtxSetCurrent`) — TensorRT can hand a plugin over before any context is
   current.
2. **Retry with a clamped PTX ISA version** (`.version 9.0`), then with an older
   `.target`, when the driver JIT rejects the newest one. Triton emits
   `.version 9.3`, which this driver will not load.
3. Set `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` when the kernel needs
   dynamic shared memory.

Porting all three fixed it outright. Verified:

```
[V3-GRID] scale_kernel_v3_one__0 grid=(32,1,1) block=128 scalars=[n=4096, alpha=2]
single kernel: maxdiff=0.000e+00  out==2x:True
```

### Verification after the fix

| Suite | V3 | V2 |
|---|---|---|
| `test_conformance.py` | **16/16** | **16/16** |
| research kernels | `0.00e+00` | `0.00e+00` |
| inplace / inplace dtypes | `0.00e+00` | `0.00e+00` |
| alignment guard | `0.00e+00` | `0.00e+00` |
| int4 dequant | `0.00e+00` | `0.00e+00` |
| edge cases | `0.00e+00` | `0.00e+00` |
| fused residual+RMSNorm | `0.00e+00` / `9.536743e-07` | same |
| Mamba SSM | `4.768372e-07` | same |

### Lesson worth recording

Every symptom pointed at the plugin's generated code, and none of it was wrong.
The bug was in the *setup* the generated code performs — a two-call
`initialize()` that silently failed. The diagnostic that mattered was printing
the error path, not the success path: the plugin had been reporting
`cuModuleLoadData failed with code 222` the whole time, and I only looked for it
after exhausting the alternatives. Log the failure branches first.

### Remaining before shipping

- The V3 emitter's instrumentation is removed and the templates are clean.
- `plugin_api` defaults to `"v2"`; V3 is opt-in. Decide whether 2.0 should flip
  the default — V3 is the interface with a future, so probably yes, but it
  changes the emitted library for every existing user.
- No CHANGELOG entry, version bump, paper or docs update yet.
- TensorRT-LLM remains parked.


---

## 7. Automatic plugin-API selection

`plugin_api` now defaults to `"auto"`. The choice is made in
`utils/env_check.resolve_plugin_api()`, which returns `(api, reason)` and is
logged under `KERNEL_LENS_DEBUG`.

### The matrix

| TensorRT | `IPluginV2*` | `IPluginV3` | emitted |
|---|---|---|---|
| < 10 (8.6, 9.x) | present | absent | `v2` |
| 10.x | deprecated | present | `v3` |
| 11+ | **removed** | present | `v3` |

An explicit `plugin_api="v2"` / `"v3"` (or `KERNEL_LENS_PLUGIN_API`) is honoured
unless it cannot work, in which case the error says which way out exists.

### It is a two-axis problem, not a one-axis one

The plugins are compiled against a **header set**, and the pip TensorRT wheels
ship **no headers** — verified: zero `.h` files in both the 10.16 and the 11.3
site-packages. So KernelLens always compiles against its bundled headers, which
are currently **TensorRT 10.14.1** (and do declare `IPluginV3`).

That makes the runtime version only half the story. A plugin built against 10.x
headers and loaded into an 11.x runtime does not have to line up, and when it
does not it **segfaults inside TensorRT with no message** — which is exactly what
happened when the suite was first run against the 11.3 environment on this host
(`EXIT=139`, no test output at all).

So `check_header_runtime_match()` compares the header major against the runtime
major and refuses the build when they differ, with actionable advice:

```
TensorRT version mismatch: the plugins would be compiled against TensorRT 10.x
headers (.../include/tensorrt) but loaded into a TensorRT 11.x runtime. The
plugin ABI is not guaranteed across majors, and the mismatch crashes inside
TensorRT rather than failing cleanly.
Fix: point TENSORRT_INCLUDE_DIR at the headers matching the runtime ...
```

`TENSORRT_INCLUDE_DIR` was already honoured by the builder's include search, so
TensorRT 11+ is reachable — it just needs matching headers, and now says so
instead of crashing.

### Verified

| Environment | `resolve(auto)` | Suite |
|---|---|---|
| TensorRT 10.16.1.11 | `v3` | **16/16** |
| TensorRT 10.16.1.11, `plugin_api="v2"` forced | `v2` | **16/16** |
| TensorRT 11.3.0.99 | `v3` (correct), then refused: headers are 10.x | clean errors, no crash |

The 11.3 case went from `EXIT=139` (segfault, zero test output) to every case
reporting the mismatch as a `RuntimeError`.

### Package metadata

`pyproject.toml` keeps `tensorrt>=8.6.0,<11.0.0` as the default extra, now with a
comment explaining that the pin exists because the bundled headers are 10.x — not
because TensorRT 11 is unsupported. That keeps `pip install kernel-lens[trt]`
working out of the box while the mismatch check covers the rest.

### Still to do

- The site's FAQ says "Everything reported here was measured on ..." and the
  install note lists the env vars; the site does not yet mention `plugin_api` or
  the version matrix. Worth a line in the install/API section.
- No version bump or CHANGELOG entry for 2.0 yet.


---

## 8. Header provisioning — supporting any TensorRT version

The refusal in §7 was a workaround, and a bad one: it made every new TensorRT
release unusable until the bundled headers were refreshed by hand. The better
answer is to get the headers that match the runtime.

### Why they have to be fetched

The pip TensorRT wheels ship **no headers** — verified on both environments, the
package contains only `__init__.py` and `plugin/`. So on a normal install the
only headers present are the ones KernelLens bundles.

### Where they come from

`utils/trt_headers.py` resolves a header set in order of preference:

1. `TENSORRT_INCLUDE_DIR` — explicit, always wins
2. the installed distribution's own include directory, if it has one
3. a local cache, `$KERNEL_LENS_HEADER_CACHE` or `~/.kernel_lens_headers/<major>.<minor>`
4. **download** from the TensorRT OSS repository, branch `release/<major>.<minor>`
5. the bundled set — only correct when the major versions agree

Step 4 is the one that generalises. All eleven headers a build needs are present
on those branches; checked for `release/11.3`:

```
NvInfer.h 200            NvInferPlugin.h 200         NvInferRuntimeCommon.h 200
NvInferImpl.h 200        NvInferPluginBase.h 200     NvInferRuntimePlugin.h 200
NvInferLegacyDims.h 200  NvInferPluginUtils.h 200    NvInferVersion.h 200
NvInferRuntime.h 200     NvInferRuntimeBase.h 200
```

The fetch is atomic (into `<dir>.partial`, moved into place only when every file
arrived) and validated (`NvInferVersion.h` must declare the version requested),
so an interrupted download is never mistaken for a complete set. It is
best-effort: if the network is unavailable we fall back to the bundled headers
and the mismatch check reports the problem in words rather than crashing.
`KERNEL_LENS_FETCH_HEADERS=0` disables it for air-gapped installs.

The builder and the mismatch check both call the same resolver, so what is
compiled against and what is checked can't drift apart.

### Result

| Runtime | Header source | Suite |
|---|---|---|
| TensorRT 10.16.1.11 | bundled (major matches, nothing fetched) | **16/16** |
| TensorRT 11.3.0.99 | **fetched** `release/11.3`, cached | **16/16** |

TensorRT 11 support is new: the package previously refused it outright, because
TensorRT 11 removed the `IPluginV2` interfaces that the only emitter produced.
The V3 emitter plus version-matched headers is what unlocked it.

### What this buys

Any future TensorRT release is supported as soon as NVIDIA publishes its release
branch — no package update needed, and no vendored headers to refresh. The
`tensorrt` extra is no longer pinned below 11; the practical ceiling is now
whatever the OSS repository tracks.

### Residual risks, stated

- A TensorRT release with no matching branch in the OSS repository (a nightly, or
  an internal build) falls back to the bundled headers and the mismatch check.
  `TENSORRT_INCLUDE_DIR` covers that case.
- Fetching depends on `raw.githubusercontent.com` reachability. Cached after the
  first success per version.
- Header availability does not prove ABI compatibility for a *future* major; it
  is evidence that the interfaces exist and compile. A major bump should still be
  run through the conformance suite.
