# Full-model attempt: real LLM with Liger-Kernel Triton kernels

Goal: evaluate KernelLens on a real pretrained model whose RMSNorm and SwiGLU
are real `@triton.jit` kernels, rather than on the controlled models used in
the main evaluation.

## What was set up

- Hardware: NVIDIA A100-SXM4-80GB (Ampere, sm_80), 80 GB VRAM
- Model: `Qwen/Qwen2.5-0.5B` (0.49B params, 24 layers, bf16), loaded and
  running eagerly in 19-20 ms per 512-token prefill
- Kernels: `liger-kernel` 0.8.5, applied with
  `apply_liger_kernel_to_qwen2()` so that RMSNorm and SwiGLU become Triton
  kernels (49 modules patched)
- Harness: `tests/benchmark_real_model.py`

## Result: blocked before KernelLens is reached

The model cannot be exported to ONNX on this software stack, with **or without**
Liger, and with **either** torch ONNX exporter:

| Configuration | `dynamo=False` (TorchScript) |
|---|---|
| Qwen2.5-0.5B, no Liger | FAILED — `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!` |
| Qwen2.5-0.5B + Liger | FAILED — `CompilationError at 27:18` |

The plain `torch.onnx.export` calls above do **not** involve KernelLens at all,
so this is not a KernelLens defect: it is a limitation of the ONNX export path
on transformers 5.20 + torch 2.12. The base model already fails to trace
cleanly (the rotary-embedding cache buffer stays on CPU under tracing).

Three further incompatibilities had to be worked around before reaching that
point, each in third-party code:

1. transformers 5.x returns a `CausalLMOutput` holding a `DynamicCache`, which
   TorchScript cannot trace — worked around with a logits-only wrapper.
2. `transformers/masking_utils.py` computes the packed-sequence mask with
   `torch.diff`, which is not expressible in ONNX opset 17. It only short-
   circuits when *not* tracing, so the export always hits it — worked around by
   rewriting it as slice subtraction.
3. Liger defaults RMSNorm to `in_place=True`, emitting `copy_`/`contiguous`,
   which the exporter rejects — worked around by clearing `in_place` on the
   patched modules.

## Why this matters for the paper

KernelLens sits downstream of the ONNX exporter: it requires the *non-kernel*
part of the model to be exportable. Modern HuggingFace models
(transformers >= 5) are not, out of the box, on the current torch release. That
is a genuine boundary of the approach and should be stated as a limitation
rather than glossed over.

It does not affect the reported evaluation, whose models are controlled
PyTorch modules that are exportable by construction.

## If this is worth pursuing

The likely fix is to pin the stack to the era the codebase targets
(transformers ~4.44, torch ~2.4/2.5, liger ~0.5) and retry. That is an
uncertain, multi-hour experiment; it should be attempted only if the
full-model result is judged to be worth more than the risk.

The cross-architecture evidence already gathered (14/14 conformance at exact
parity, 4/4 parity, folding and end-to-end on Ampere) does not depend on this.
