# Paper

The technical report corresponding to this documentation is
**“KernelLens: Compiling PyTorch Triton Kernels into C++ Inference Plugins”**
by Justin Duc (Galiad Research). This page is the reader's companion; the full
LaTeX source lives at [`architecture_report.tex`](https://github.com/Just1truc/kernel_lens/blob/main/architecture_report.tex).

## Abstract

> Writing custom GPU kernels in OpenAI Triton has become routine, but *deploying*
> them is not. Production inference is increasingly served by pure C++ engines —
> Microsoft ONNX Runtime (ORT) and NVIDIA TensorRT (TRT) — which apply whole-graph
> optimisation, engine tuning and memory planning. Yet `torch.onnx.export` cannot
> trace an imperative `@triton.jit` call, and a hand-written C++ plugin costs
> thousands of lines per kernel. We present **KernelLens**, a compiler and runtime
> that closes this gap. KernelLens runs a four-phase pipeline: (1) dual-pass
> tracing and static AST analysis that recover the compiled PTX, the launch grid
> and a per-argument input/output/in-place taxonomy; (2) ONNX graph rewriting that
> injects `triton_custom` nodes and binds host scalars as CPU inputs; (3) automated
> `OrtCustomOp` and `IPluginV2DynamicExt` synthesis, including PTX entry-slot ABI
> reconstruction and PTX ISA version clamping for Blackwell; and (4) zero-copy
> VRAM binding that registers PyTorch `data_ptr()` values directly with each
> engine. We evaluate KernelLens on research kernels from LLaMA 3, PaLM, Qwen 2.5
> and Liger-Kernel, on FP8 and INT4 quantized GEMMs, and on a 4-layer LLaMA
> decoder. KernelLens reproduces native Triton execution exactly on the four
> research kernels inside ONNX Runtime, and its TensorRT plugins are up to 3.3×
> faster than `torch.compile` on custom attention while using the least peak VRAM.
> We also report, in full, the TensorRT paths that remain incorrect, and correct
> several claims from earlier drafts of this work.

## Where to look

<div class="grid cards" markdown>

-   :material-sitemap: **[Architecture](architecture.md)**

    The four phases, end to end, with the artifacts each one produces.

-   :material-chart-line: **[Research Results](results.md)**

    Parity, latency, end-to-end and quantization measurements.

-   :material-flask-outline: **[Verification](verification.md)**

    The exact protocol, the bugs found and fixed, and the remaining limitations.

</div>

## Contributions, in one table

| # | Contribution | Where it is demonstrated |
| :--- | :--- | :--- |
| 1 | A dual-pass tracer that recovers PTX, dynamic grid and argument taxonomy from an unmodified PyTorch module | [Architecture § Phase 1](architecture.md) |
| 2 | Automated ONNX custom-node injection with dynamic CPU-scalar binding and in-place aliasing | [Architecture § Phase 2](architecture.md) |
| 3 | Automatic synthesis of `OrtCustomOp` and `IPluginV2DynamicExt` plugins, including PTX entry-slot ABI reconstruction and PTX ISA clamping | [Architecture § Phase 3](architecture.md) |
| 4 | Zero-copy runtime binding for both engines | [Architecture § Phase 4](architecture.md) |
| 5 | An empirical study across LLM research kernels, quantized kernels and a full decoder, with an honest verification appendix | [Results](results.md) &middot; [Verification](verification.md) |

## Citation

```bibtex
@article{duc2026kernellens,
  title        = {KernelLens: Compiling PyTorch Triton Kernels into C++ Inference Plugins},
  author       = {Duc, Justin},
  institution  = {Galiad Research},
  year         = {2026},
  version      = {1.1.9},
  doi          = {10.5281/zenodo.23194523},
  url          = {https://doi.org/10.5281/zenodo.23194523}
}
```