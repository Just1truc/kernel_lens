# Runbook: running the full evaluation on a fresh GPU host

One command provisions; one command runs the whole matrix. Both are in
`scripts/` and validated end to end.

## On RunPod

1. **Image**: any recent CUDA 12.x/13.x image with PyTorch (e.g. a
   `runpod/pytorch:*` template). It must ship `nvcc` — the plugin generator
   compiles C++ at run time.
2. **GPU**: A100 80GB (or L40S / H100). The 80GB matters for OPT-2.7B in fp32
   and OPT-6.7B.
3. **Volume**: attach a network volume and point the cache at it. The cache
   holds the built ONNX graphs and TensorRT engines and dominates the runtime —
   on the Thunder run the OPT-2.7B engine alone was 5.3 GB.

## Commands

```bash
# 1. provision (installs TensorRT <11, onnxruntime-gpu, transformers; clones and
#    installs the package; prints the detected GPU)
WORKDIR=/workspace/kl \
KERNEL_LENS_CACHE_DIR=/workspace/kl_cache \
  bash <(curl -sL https://raw.githubusercontent.com/Just1truc/kernel_lens/main/scripts/remote_setup.sh)

# 2. run everything
cd /workspace/kl
KERNEL_LENS_CACHE_DIR=/workspace/kl_cache bash scripts/remote_bench_all.sh
```

Results land in `./results/`: per-run logs, one JSON per model, and
`summary.json`.

Roughly 1–2 hours for the full matrix, dominated by the OPT-2.7B runs.

## What the matrix covers

- **conformance** — 14 module patterns, both backends
- **parity** — 4 research kernels
- **folding** — the headline comparison
- **end-to-end** — the 4-layer decoder
- **real models** — OPT-125M / 350M / 1.3B / 2.7B / 6.7B, each in fp32 and fp16
  where the model fits, both backends

## The one thing to watch

TensorRT deserialization failed on the Thunder A100 for engines above roughly
1 GB, with a CUDA runtime error, and the host panicked in 3 of 4 attempts.
Engines of 315 MB and 704 MB deserialize normally. TensorRT reads the whole
blob first (`Loaded engine size: 2710 MiB`), so this is **not** the 2 GB
protobuf limit it first resembles — it looked host-specific.

**On RunPod, expect the TensorRT column for OPT-1.3B and 2.7B to fill in.** If
it does, that is the missing row in Table `tab:real`. If it fails identically,
the limitation is TensorRT-side and the paper's existing wording already covers
it.

`remote_bench_all.sh` records failures and keeps going, so a partial matrix is
still usable.

## Bringing the results back

```bash
# from your machine
scp -r <user>@<host>:<port>:/workspace/kl/results ./runpod_results
```

Then the numbers for Table `tab:real` are in `summary.json`, and the accuracy
metrics (top-1 agreement, mean/max logit difference) are per-model in each
`real_*.json`.

## Environment notes

The provisioning script was validated on an A100-SXM4-80GB (sm_80), CUDA 13.3:
torch 2.12.1, triton 3.7.1, onnx 1.23.2, onnxruntime-gpu 1.31.0, TensorRT
10.16.1.11, kernel_lens 1.2.1.

Two version constraints worth keeping:

- **TensorRT must be `<11`.** The package refuses 11+ because the generator
  emits the `IPluginV2DynamicExt` interface that 10.x uses.
- **`nvcc` must match the driver's CUDA major version reasonably.** On the
  Thunder host the driver was CUDA 13.3 and `cuda-nvcc-13-3` +
  `cuda-cudart-dev-13-3` were needed.
