"""
Where do TensorRT / ONNX Runtime beat torch.compile?

A conv-heavy model (where TensorRT uses cuDNN/cutlass kernels and autotuning)
plus one custom Triton kernel, run fp32 and fp16.

Run:  KL_DTYPE=fp16 python3 tests/benchmark_trt_vs_compile.py
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import kernel_lens as kl


@triton.jit
def relu_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    z = tl.zeros_like(x)
    tl.store(o_ptr + offs, tl.maximum(x, z), mask=m)


class CNN(nn.Module):
    def __init__(self, w=224, ch=64):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.b1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.b2 = nn.BatchNorm2d(64)
        self.c3 = nn.Conv2d(64, ch, 3, padding=1)
        self.b3 = nn.BatchNorm2d(ch)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        o = torch.empty_like(x)
        n = x.numel()
        relu_kernel[(triton.cdiv(n, 1024),)](x, o, n, BLOCK=1024)
        x = self.b3(self.c3(o))
        return x.mean(dim=(2, 3))


def bench(fn, args=(), warmup=10, iters=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    DT = os.environ.get("KL_DTYPE", "fp32")
    tdt = {"fp32": torch.float32, "fp16": torch.float16}[DT]
    B, W = int(os.environ.get("KL_B", 16)), int(os.environ.get("KL_W", 224))
    x = torch.randn(B, 3, W, W, device="cuda", dtype=tdt)
    model = CNN(W).cuda().eval().to(tdt)
    print(f"config: conv-heavy B={B} W={W} dtype={DT}")
    res = {}
    with torch.no_grad():
        ref = model(x)
        res["eager"] = bench(lambda: model(x))
        try:
            tc = torch.compile(model)
            for _ in range(3):
                tc(x)
            res["torch.compile"] = bench(lambda: tc(x))
        except Exception as ex:
            print("compile failed:", str(ex)[:200])
        cm = kl.compile(model, (x,), name=f"TRTvsCompile_{DT}", backends=["onnx", "tensorrt"])
        res["KernelLens ORT"] = bench(lambda: cm.run((x,), backend="onnx"))
        res["KernelLens TensorRT"] = bench(lambda: cm.run((x,), backend="tensorrt"))
        o_trt = cm.run((x,), backend="tensorrt")
        o_ort = cm.run((x,), backend="onnx")
        print("parity vs eager: TRT max diff =", (o_trt - ref).abs().max().item(),
              "| ORT max diff =", (o_ort - ref).abs().max().item())
    base = res.get("torch.compile", res["eager"])
    print("\n--- Latency (ms) ---")
    for k, v in res.items():
        print(f"  {k:22}: {v:8.3f} ms  ({base / v:5.2f}x vs torch.compile)")
    with open(f"measured_trt_vs_compile_{DT}.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()