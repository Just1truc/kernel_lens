# Real pretrained models through KernelLens

Follow-up to the earlier attempt that was blocked. The blocker was the model
family, not KernelLens.

## Why OPT works and the RoPE families do not

KernelLens exports the non-kernel part of the model with the TorchScript ONNX
exporter (`dynamo=False`). Exportability therefore depends on the architecture:

| Model | Positional scheme | TorchScript export |
|---|---|---|
| `gpt2`, `distilgpt2` | learned | OK (single file) |
| `facebook/opt-125m` | learned | OK (single file) |
| `bigscience/bloom-560m` | ALiBi | OK (single file) |
| `TinyLlama-1.1B` | RoPE | FAILS (only exports via the dynamo path, which uses external data) |
| `Qwen/Qwen2.5-0.5B` | RoPE | FAILS for the same reason |

The RoPE families register a rotary cache buffer that stays on CPU under
tracing, so the TorchScript path rejects them. This is a property of
transformers >= 5 + torch 2.12, not of KernelLens: plain `torch.onnx.export`
fails identically with no KernelLens involved.

## Setup

OPT's architecture is used because it is exportable. Every `nn.LayerNorm` in the
decoder is replaced by the project's own Triton RMSNorm kernel, carrying the
LayerNorm weight across, so the model calls a real `@triton.jit` kernel exactly
as a production model would. Harness: `tests/benchmark_real_model.py`.

## Results

### OPT-125M — 12 layers, 25 Triton RMSNorm modules, fp32, S=512

| Configuration | Latency | vs eager |
|---|---|---|
| Eager + Triton | 11.97 ms | 1.00x |
| KernelLens — ONNX Runtime | 9.15 ms | **1.31x** |
| KernelLens — TensorRT | 3.65 ms | **3.28x** |

`max|diff|` against the eager reference: 1.59e-01 (ORT), 1.65e-01 (TensorRT) on
logits of shape (1, 512, 50272). This is fp32 and the engine is free to
reassociate the surrounding matmuls, so it is not the bit-exact agreement seen
on the controlled models; it is the expected graph-level difference.

### OPT-1.3B — 24 layers, 49 Triton RMSNorm modules, fp16, S=512

| Configuration | Latency | max abs diff |
|---|---|---|
| Eager + Triton | 14.20 ms | - |
| KernelLens — ONNX Runtime | 21.80 ms | **0.00e+00** |
| KernelLens — TensorRT | not available (see below) | - |

The ONNX Runtime lane is bit-exact against the eager reference on a real 1.3B
model, but it is slower than eager here. That is a genuine negative result: the
per-node host overhead of the runtime dominates once the kernels are small
relative to the surrounding GEMMs.

## Limitation: TensorRT cannot load models above the 2 GB protobuf limit

`torch.onnx.export` spills weights to external files once a graph exceeds 2 GB.
For OPT-1.3B this produced 2.6 GB of per-tensor spill files — both initializers
*and* `Constant` node attributes — written beside the `.onnx`. TensorRT resolves
those relative to the working directory rather than the model's directory and
fails with:

```
[TRT] [E] WeightsContext.cpp:183: Failed to open file: m.model.decoder.embed_tokens.weight
[TRT] [E] ... UNSUPPORTED_NODE: Failed to import initializer
```

`_consolidate_external_data` in `kernel_lens/compiler/core.py` folds
initializer spill files into a single sibling `.data` file, which is the correct
layout, but it does not cover the externally stored `Constant` attributes, so
TensorRT still rejects the graph. Models below 2 GB (OPT-125M, GPT-2, BLOOM-560M)
are unaffected.

Making large models work with TensorRT would need the export to keep constant
attributes inline (or to place them where the parser can find them). This is an
open item, not a resolved one.
