# Dynamic shapes in KernelLens — investigation notes

Status: **fixed.** 10 files modified, uncommitted. Conformance suite **16/16**,
every case bit-exact (`0.00e+00`) on both ONNX Runtime and TensorRT, including the
two new sweeps that serve several shapes through a single engine.

Two independent root causes, both found by measurement rather than reasoning:

1. **The symbolic tracing pipeline was intact but its output was discarded.**
   `SymNode.expr` substitutes the shape env's known values and returns the
   *collapsed constant*; `_expr` keeps the symbol. Every consumer read `.expr`,
   so the launch grid was frozen at the traced shape. (§2)
2. **The tracer picked the PTX by cache position, not by dtype.** Triton's cache
   holds one compiled variant per argument specialisation; `reversed()` returns
   the most recently *inserted*, and a re-used entry keeps its original slot — so
   an fp32 variant compiled early sits before a bf16 one compiled later, and the
   fp32 launch was handed the **bf16 PTX**. Reading fp32 memory 16 bits at a time
   covers exactly half the tensor. (§4)

The second one only manifests when a kernel is exercised at several dtypes in one
process, which is why it looked like a cross-test interaction.

Environment for every measurement below: RTX 5060 Ti, driver 595.91.07,
torch 2.14.0+cu130, triton 3.8.0, TensorRT 10.16.1.11, onnxruntime-gpu 1.30.0,
nvcc 13.3 (`envs/kernel_lens_fp8`), Python from `envs/kernel_lens_trt`,
package imported from source via `PYTHONPATH=~/kl_121/src`.

---

## 1. What "dynamic shapes" had to achieve

`kl.compile(model, sample)` traced a model at one shape and produced an engine
that only accepted that shape. The goal: one compile, one engine, correct at any
batch (or leading-dim) size within a declared range.

TensorRT supports this natively — dynamic inputs (`-1` dims), an optimization
profile (`min`/`opt`/`max`), and `IPluginV2DynamicExt`, which is the interface
KernelLens already emits. The failure was never TensorRT's; it was in the
pipeline between the tracer and the plugin.

---

## 2. The symbolic pipeline, as it actually works

`extract_manifests()` (`kernel_lens/compiler/tracer.py:199`) runs **two passes**:

| Pass | What it does | What it yields |
|---|---|---|
| 1 | real execution with concrete tensors | real PTX, concrete shapes, strides |
| 2 | `make_fx(module, tracing_mode="symbolic")` | symbolic shapes, symbolic grid |

then **merges** them (`tracer.py:231-251`):

```python
m1._sym_grid_asts = m2._sym_grid_asts
for a1, a2 in zip(m1.arguments, m2.arguments):
    a1._sym_ast = a2._sym_ast
```

Both passes capture launches by patching `triton.JITFunction.run`. Pass 2
genuinely produces symbolic values — measured:

```
Pass 2 Grid ASTs: (((s75*s96 + 255)//256),)
       x_ptr   : shape=(s75, s96), strides=(s96, 1)
```

So the mechanism the design intended does exist and does work.

### The critical detail

`SymNode` has two accessors, and they are not the same:

```python
node._expr = ((s75*s96 + 255)//256)   # raw symbol, intact
node.expr  = 16                       # substituted via shape_env.replace(), cached
node.hint  = 16
```

Source (`torch/fx/experimental/sym_node.py`, `expr` property): it returns `_expr`
while `shape_env._replacements_version_counter == 0`, otherwise
`shape_env.replace(_expr)` — which substitutes `var_to_val` (`s75 -> 8`,
`s96 -> 512`) and folds the expression to the constant.

**Every consumer in the codebase read `.expr` / `str(SymInt)`, i.e. the collapsed
constant.** That is the single root cause of the frozen launch geometry.

It is also why nothing was ever broken before: for a shape-specialised engine the
substituted value *is* the correct answer (`cdiv(8*512,256) = 16`).

---

## 3. Fixes landed

| # | File | Change |
|---|---|---|
| 1 | `compiler/ast_analyzer.py` | `raw_sym_expr()` reads `node._expr`; `translate_symint_to_cxx()` uses it and falls back to the resolved constant if any symbol survives |
| 2 | `compiler/ast_analyzer.py` | powers: `**` was emitted as `^` (bitwise XOR in C++). Now expanded to repeated multiplication **before** symbol substitution, when the base is still a bare symbol |
| 3 | `compiler/ast_analyzer.py` | `re.sub` with a string replacement corrupted the substitution (`GetInput`-style text). Now `re.escape` + lambda |
| 4 | `compiler/tracer.py` | merge also carries Pass 2's symbolic shape: `a1.sym_shape = tuple(a2.shape)` |
| 5 | `compiler/manifest.py` | `sym_map_for(accessor)` / `grid_exprs(accessor)` / `scalar_exprs(accessor)` — one translation, two backends (`dim_values[i]` for ORT, `inputDesc[0].dims.d[i]` for TRT) |
| 6 | `compiler/manifest.py` | `shape_scalar_specs()` — expression + symbol→dim map + traced value for shape-derived scalars, persisted in `output_meta.pt` |
| 7 | `compiler/core.py` | `dynamic_shapes` / `dynamic_max` on `compile()`; `_dynamic_axes()` marks the leading dim; `_check_dynamic_supported()` guard |
| 8 | `runtime/engine.py` | profile `opt` from the **traced** shape (was from whichever call built the engine first, giving `max=4` when the first run was batch 1); `max` = 4x traced, overridable; `infer_shapes()` on dynamic graphs; fresh `IOBinding` per call; output shape resolved from the live input |
| 9 | `backends/*_gen.py` | grid from `manifest.grid_exprs(accessor)`; TRT scalars from `scalar_exprs(accessor)`; ORT `GetInput` indexed by the node's own input list rather than a running counter; ORT scalars prefer the `dim_values` expression when the graph folded them into constants |
| 10 | `compiler/tracer.py` | PTX variant selected by the launch's argument dtypes rather than by cache position (root cause 2) |

### Verified working

One engine, compiled at batch 8, serving batches 1..16 — and compiled at 524288
elements, serving 256..4096 — bit-exactly, on **both** backends:

```
dynamic_batch_sweep  compiled@8     onnx=0.00e+00 tensorrt=0.00e+00
dynamic_grid_sweep   compiled@2048  onnx=0.00e+00 tensorrt=0.00e+00     <- in isolation
```

Plugin logs confirming the grid is computed per call:

```
[ORT-DBG] add_kernel dim_values=(1 512)   grid=(2,1,1)
[ORT-DBG] add_kernel dim_values=(16 512)  grid=(32,1,1)
[ENQ]     add_kernel in dims=(16 512)     grid=(32,1,1)
```

---

## 4. Root cause 2: the PTX was selected by cache position, not by dtype

`dynamic_grid_sweep` (`SameKernelTwice`, two `scale_kernel` launches, BLOCK=128)
failed at **every** batch — but only when the 14 static conformance cases ran
first. Bisected:

| Configuration | Result |
|---|---|
| cases 0-6 + grid sweep | pass |
| cases 7-13 + grid sweep | pass |
| all 14 statics + grid sweep | **fail** |
| no statics + both sweeps | pass |
| any single static case + grid sweep | pass |

Needs the full set; not one case, not either half; the batch sweep is not required.

### Signature

```
OneScale  b=256   n=65536    last_nonzero=32768   (0.5000 of n)
OneScale  b=512   n=131072   last_nonzero=65536   (0.5000 of n)
OneScale  b=2048  n=524288   last_nonzero=262144  (0.5000 of n)
OneScale  b=4096  n=1048576  last_nonzero=524288  (0.5000 of n)
```

- exactly **half** the output written, at *every* batch, **including the traced one**
- the head is bit-correct (`out/x = 2.00002` for `k < n/2`), the tail is `0.0`
- identical on ORT and TRT

### Ruled out (all measured in the failing run)

| Suspect | Evidence |
|---|---|
| launch grid | `grid=(4096,1,1)` for n=524288 — correct |
| block size / reqntid | `block_x = 128`, `.reqntid 128` — consistent |
| `dim_values` | `(524288)` — correct |
| reference | `|ref − 6x|max = 0.000e+00` — exact |
| runtime scalar map | `{524288.0: 524288}` — correct |
| `alpha` | `m_alpha(2.0)` in ctor, `(float)(m_alpha)` at launch |
| plugin input indices | `GetInput(1)/(2)` now match `GetInputTypeCount()=3` |
| **stale element count** | **impossible** — a baked `n` would be *correct* at the traced shape, and it fails there |

A constant 0.5 fraction invariant across shapes is the key clue: it is a
per-program coverage mismatch, not a stale value.

### The cause

Diffing the generated ORT plugin between a passing cache and a failing one showed
the PTX itself was compiled for the **wrong dtype**:

```
failing  > mul.wide.s32 %rd5, %r7, 2;      <-- 2 bytes per element
         > @%p1 ld.global.b16 { %rs1 }, [ %rd1 + 0 ];
         > cvt.f32.bf16 %r8, %rs1;

passing  < mul.wide.s32 %rd5, %r9, 4;      <-- 4 bytes per element
         < @%p1 ld.global.b32 { %r1 }, [ %rd1 + 0 ];
         < mul.f32 %r2, %r6, %r1;
```

`tracer.py` extracts the PTX by walking Triton's compile cache and taking the
first hit:

```python
caches = getattr(jit_self, 'cache', getattr(jit_self, 'device_caches', {}))
for key_or_dev, value in reversed(list(caches.items())):
    for key_tuple, compiled in reversed(items_to_check):
        if hasattr(compiled, 'asm') and 'ptx' in compiled.asm:
            ptx = compiled.asm['ptx']
```

The cache holds one variant per argument specialisation, keyed like

```
"[('*fp32', 'D'), ('*fp32', 'D'), ('i32', 'D'), ('fp32', None), ('constexpr', 128)]"
"[('*bf16', 'D'), ('*bf16', 'D'), ('i32', 'D'), ('fp32', None), ('constexpr', 128)]"
```

with each key's PTX matching its dtype. But `reversed()` returns the most recently
*inserted* entry, and a re-used entry keeps its original dict slot — so an fp32
variant compiled early sits **before** a bf16 one compiled later. The grid sweep's
fp32 launch was therefore handed the bf16 PTX.

This explains the bisect exactly:

| Configuration | Why |
|---|---|
| cases 0-6 + sweep | no bf16 case ran, so the last `scale_kernel` variant is fp32 |
| cases 7-13 + sweep | fp16 and bf16 compiled, but fp32 is a cache **miss**, so it is inserted last |
| all 14 + sweep | fp32 compiled at index 1, bf16 at index 12 → fp32 launch is a cache **hit**, bf16 still last → fail |

### The fix

Build the expected dtype markers from the launch's own arguments and accept only
the variant whose key contains them, falling back to a positional pick if nothing
matches (`_TRITON_DTYPE_MARK` and `expected_markers` in `tracer.py`).

---

## 5. Verification

Full conformance suite, fresh cache, both backends:

```
[PASS] two_kernels            onnx=0.00e+00 tensorrt=0.00e+00
[PASS] same_kernel_twice      onnx=0.00e+00 tensorrt=0.00e+00
[PASS] two_outputs            onnx=0.00e+00 tensorrt=0.00e+00
[PASS] dynamic_scalar         onnx=0.00e+00 tensorrt=0.00e+00
[PASS] strided                onnx=0.00e+00 tensorrt=0.00e+00
[PASS] loop                   onnx=0.00e+00 tensorrt=0.00e+00
[PASS] conditional            onnx=0.00e+00 tensorrt=0.00e+00
[PASS] nested_helper          onnx=0.00e+00 tensorrt=0.00e+00
[PASS] inplace                onnx=0.00e+00 tensorrt=0.00e+00
[PASS] grid3d                 onnx=0.00e+00 tensorrt=0.00e+00
[PASS] autotuned              onnx=0.00e+00 tensorrt=0.00e+00
[PASS] fp16                   onnx=0.00e+00 tensorrt=0.00e+00
[PASS] bf16 (TRT only)        tensorrt=0.00e+00
[PASS] dynamic_batch          onnx=0.00e+00 tensorrt=0.00e+00
[PASS] dynamic_batch_sweep    compiled@8    served=[1,2,4,8,16]        onnx=0.00e+00 tensorrt=0.00e+00
[PASS] dynamic_grid_sweep     compiled@2048 served=[256,512,2048,4096] onnx=0.00e+00 tensorrt=0.00e+00

=== 16/16 conformance cases passed ===
```

Regression check across every other suite, fresh cache (the tracer change affects
*every* compile, so this was run in full rather than assumed):

| Suite | Result |
|---|---|
| `test_research_kernels` | all kernels pass, `0.00e+00` vs native Triton |
| `test_fp8_gemm` | `0.000000e+00` |
| `test_int4_dequant_gemm` | `0.000000e+00` |
| `test_inplace` | `0.000000e+00` |
| `test_inplace_dtypes` | fp32 and fp16 both `0.000000e+00` |
| `test_alignment_guard` | `0.000000e+00` |
| `test_edge_cases` | `0.000000e+00` both backends |
| `test_fused_residual_rms` | residual `0.00e+00`, norm `9.536743e-07` |
| `test_mamba_selective_scan` | `4.768372e-07` |
| `test_fused_cross_entropy` | `1.907349e-06` |
| `test_unit_cpu` | 5 passed |

Every value is identical to the pre-change measurements, so nothing regressed.

Still outstanding before shipping: the `_check_dynamic_supported()` guard is
currently **not called** from `compile()` (removed so the dynamic path could be
exercised). It should either be deleted or rewritten to check the *translated*
expressions — with the fixes in place a correct kernel now produces runtime
expressions, so the guard's original premise no longer holds.

Also: debug logging is env-gated but still present in three files
(`KL_DBG`, `KL_SYM_DEBUG`, `KL_SWEEP_DBG`) and should be removed or folded into
`debug_print` before release.

---

## 6. Tree state

Nothing committed. Modified:

```
DYNAMIC_SHAPES_FINDINGS.md   (this file)
kernel_lens/backends/ort_gen.py
kernel_lens/backends/trt_gen.py
kernel_lens/compiler/ast_analyzer.py
kernel_lens/compiler/core.py
kernel_lens/compiler/manifest.py
kernel_lens/compiler/tracer.py
kernel_lens/runtime/engine.py
tests/test_conformance.py
```

Debug logging is env-gated: `KL_DBG` (plugin dims/grid), `KL_SYM_DEBUG`
(translation), `KL_SWEEP_DBG` (per-batch harness detail).

Earlier commits `fdc3d77`, `c0a1de1`, `a28b9b1` are pushed and still need review;
the two CHANGELOG commits among them contain diagnoses now known to be wrong.

## 7. Corrections to earlier claims

Recorded so they are not repeated:

- "the tracer bakes the grid and there is no code path for a symbolic one" — **wrong**, Pass 2 produces it
- "the SymInts are mutated in place by the launch" — **wrong**, same `id` and expression before/after
- "a ShapeEnv lifetime problem" — **wrong**, the env is alive, `_expr` intact
- "`node.expr` is the hint accessor" — imprecise; it is `shape_env.replace(_expr)`, cached on a version counter
- "`grid_cxx_exprs` is never read" — true, and still true; the generators use `grid_exprs(accessor)` instead
- "`cxx_expr` is never read by the generators" — **wrong**, `trt_gen.py:365` reads it (an earlier `grep` used a bad working directory)

---

## 8. Second architecture: A100-SXM4-80GB (sm_80, Ampere)

Host: `ubuntu@198.145.126.228:32398`, driver 610.43.02, Ubuntu 22.04.
Not a CUDA image — had to be provisioned:

```bash
curl -fsSL -o cuda-keyring.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring.deb && sudo apt-get update
sudo apt-get install -y cuda-nvcc-13-3 cuda-cudart-dev-13-3 cuda-crt-13-3
pip install onnx onnxruntime-gpu pytest transformers "tensorrt>=10,<11" "tensorrt-cu13-libs<11"
```

torch 2.12.1+cu130, triton 3.7.1, onnx 1.23.2, onnxruntime 1.31.0,
tensorrt 10.16.1.11, g++ 11.4.

### Conformance: 16/16

```
[PASS] two_kernels / same_kernel_twice / two_outputs / dynamic_scalar
[PASS] strided / loop / conditional / nested_helper / inplace
[PASS] grid3d / autotuned / fp16 / bf16 (TRT only) / dynamic_batch
[PASS] dynamic_batch_sweep    compiled@8    served=[1,2,4,8,16]         onnx=0.00e+00 tensorrt=0.00e+00
[PASS] dynamic_grid_sweep     compiled@2048 served=[256,512,2048,4096]  onnx=0.00e+00 tensorrt=0.00e+00

=== 16/16 conformance cases passed ===
```

Both dynamic sweeps bit-exact on Ampere, matching the Blackwell result.

### Parity suites

| Suite | Result |
|---|---|
| `test_research_kernels` | all pass, `0.00e+00` vs native Triton |
| `test_int4_dequant_gemm` | `0.000000e+00` |
| `test_inplace` / `test_inplace_dtypes` | `0.000000e+00` (fp32 and fp16) |
| `test_alignment_guard` | `0.000000e+00` |
| `test_edge_cases` | `0.000000e+00` both backends |
| `test_fused_residual_rms` | residual `0.00e+00`, norm `9.536743e-07` |
| `test_mamba_selective_scan` | `7.152557e-07` (was `4.768372e-07` on the 5060 Ti — different accumulation order) |
| `test_fused_cross_entropy` | `1.907349e-06` (identical to the 5060 Ti) |
| `test_unit_cpu` | 5 passed |
| `test_fp8_gemm` | **cannot run on this GPU** — `fp8e4nv not supported in this architecture`, Triton only offers `fp8e4b15`/`fp8e5` on sm_80. E4M3 needs sm_89+. Not a KernelLens issue; it is why the FP8 results were measured on the 5060 Ti. |

### Real model: accuracy reproduces exactly, latency does not

`facebook/opt-125m`, S=512, prefill:

| dtype | eager | torch.compile | ORT | top1 | mean\|d\| | TRT | top1 | mean\|d\| |
|---|---|---|---|---|---|---|---|---|
| fp32 | 12.88 ms | 12.09 ms | 12.11 ms | 99.80% | 6.26e-04 | 5.49 ms | 100.00% | 6.14e-04 |
| fp16 | 5.18 ms | 2.76 ms | 10.81 ms | 99.80% | 1.04e-03 | 3.85 ms | 96.09% | 1.17e-02 |

**Every `top1` and `mean|d|` matches the published A100 table exactly**
(fp32 99.80/100.00 and 6.3e-04/6.1e-04; fp16 99.80/96.09 and 1.0e-03/1.2e-02).
So the numerics are unchanged by the dynamic-shape work.

Latency does **not** match the published table (published fp32 ORT 4.8 / TRT 3.4;
fp16 ORT 3.6 / TRT 2.0). This host is slower across the board, but ORT more so —
on a different host with a different driver (610 vs 595), torch (2.12.1 vs
2.14.0) and triton (3.7.1 vs 3.8.0). **Not diagnosed.** ORT fp32 measuring at
eager speed suggests its graph-level fusion is not happening at all here.

### A provisioning trap worth recording

The first ORT measurement on this host was **invalid**: the CUDA provider silently
failed to load and ORT fell back to CPU, which both slowed it to eager speed and
made the custom op unimplemented.

```
Failed to load libonnxruntime_providers_cuda.so with error:
libcublasLt.so.13: cannot open shared object file: No such file or directory
```

The libraries ship as wheels under `dist-packages/nvidia/*/lib`. Setting
`LD_LIBRARY_PATH` to include them fixed it:

```bash
NV=/usr/local/lib/python3.12/dist-packages/nvidia
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64:$NV/cu13/lib:$NV/cudnn/lib:$NV/cusparselt/lib:$NV/nccl/lib:$LD_LIBRARY_PATH
```

after which `get_providers()` returned `['CUDAExecutionProvider', 'CPUExecutionProvider']`.
`scripts/remote_setup.sh` should detect and export this — it currently does not.
