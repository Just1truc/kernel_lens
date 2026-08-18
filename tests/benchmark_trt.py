import time
import torch
import torch.nn as nn
import numpy as np
import triton
import kernel_lens as kl
from final import TritonNHWCSequentialDecoder, SequentialDecoder

def benchmark_models():
    device = "cuda"
    channels = 128
    B, C, H, W = 1, channels, 64, 64
    x_nchw = torch.randn(B, C, H, W, device=device)
    x_nhwc = x_nchw.contiguous(memory_format=torch.channels_last)

    # 1. PyTorch Eager SequentialDecoder
    py_model = SequentialDecoder(channels).to(device).eval()
    
    # 2. Triton NHWC SequentialDecoder
    triton_model = TritonNHWCSequentialDecoder(channels).to(device).eval()

    # 3. KernelLens compilation (TensorRT & ONNX)
    kl_conv = kl.compile(triton_model, (x_nchw,), name="NHWC_Conv_SOTA", backends=["onnx", "tensorrt"])

    # Warmup
    with torch.no_grad():
        for _ in range(50):
            _ = py_model(x_nchw)
            _ = triton_model(x_nhwc)
            _ = kl_conv.run((x_nchw,), backend="onnx")
            _ = kl_conv.run((x_nchw,), backend="tensorrt")

    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    iters = 200

    # Timing PyTorch Eager
    start_evt.record()
    with torch.no_grad():
        for _ in range(iters):
            _ = py_model(x_nchw)
    end_evt.record()
    torch.cuda.synchronize()
    py_ms = start_evt.elapsed_time(end_evt) / iters

    # Timing Standalone Triton Kernel
    start_evt.record()
    with torch.no_grad():
        for _ in range(iters):
            _ = triton_model(x_nhwc)
    end_evt.record()
    torch.cuda.synchronize()
    triton_ms = start_evt.elapsed_time(end_evt) / iters

    # Timing ONNX Runtime Custom Op
    start_evt.record()
    with torch.no_grad():
        for _ in range(iters):
            _ = kl_conv.run((x_nchw,), backend="onnx")
    end_evt.record()
    torch.cuda.synchronize()
    ort_ms = start_evt.elapsed_time(end_evt) / iters

    # Timing TensorRT Custom Plugin Engine
    start_evt.record()
    with torch.no_grad():
        for _ in range(iters):
            _ = kl_conv.run((x_nchw,), backend="tensorrt")
    end_evt.record()
    torch.cuda.synchronize()
    trt_ms = start_evt.elapsed_time(end_evt) / iters

    print(f"\n=======================================================")
    print(f"       BENCHMARK RESULTS & SPEEDUP COMPARISON          ")
    print(f"=======================================================")
    print(f" PyTorch Eager Latency      : {py_ms:.4f} ms")
    print(f" Standalone Triton Latency   : {triton_ms:.4f} ms (Speedup vs PyTorch: {py_ms/triton_ms:.2f}x)")
    print(f" ONNX Runtime Kernel Latency : {ort_ms:.4f} ms (Speedup vs PyTorch: {py_ms/ort_ms:.2f}x)")
    print(f" TensorRT Plugin Engine      : {trt_ms:.4f} ms (Speedup vs PyTorch: {py_ms/trt_ms:.2f}x)")
    print(f"=======================================================\n")

if __name__ == "__main__":
    benchmark_models()

