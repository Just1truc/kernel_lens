import time
import torch
import torch.nn as nn
import triton
import triton.language as tl
import kernel_lens as kl
import json
import os

# -----------------------------------------------------------------------------
# 1. FP8 Scaled GEMM Kernel
# -----------------------------------------------------------------------------
@triton.jit
def fp8_scaled_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    scale,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        mask_a = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        mask_b = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    acc = acc * scale

    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_c)


class FP8ScaledGEMMModule(nn.Module):
    def __init__(self, m: int, n: int, k: int, scale: float = 1.0):
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.scale = scale

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        c = torch.empty((self.m, self.n), device=a.device, dtype=torch.float32)
        grid = (triton.cdiv(self.m, 64), triton.cdiv(self.n, 64))
        
        fp8_scaled_gemm_kernel[grid](
            a, b, c,
            self.m, self.n, self.k,
            self.scale,
            self.k, 1,
            self.n, 1,
            self.n, 1,
            BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64
        )
        return c


# -----------------------------------------------------------------------------
# 2. INT4 W4A16 Dequantization GEMM Kernel
# -----------------------------------------------------------------------------
@triton.jit
def int4_dequant_gemm_kernel(
    x_ptr, w_packed_ptr, scales_ptr, zeros_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_packed_ptr + (offs_k[:, None] // 2) * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        mask_x = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        mask_w = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w_byte = tl.load(w_ptrs, mask=mask_w, other=0)

        is_high = (offs_k[:, None] % 2) == 1
        w_u4 = tl.where(is_high, (w_byte >> 4) & 0x0F, w_byte & 0x0F)

        k_curr = k + offs_k[:, None]
        g_idx = k_curr // group_size
        scale = tl.load(scales_ptr + g_idx * N + offs_n[None, :], mask=mask_w, other=1.0)
        zero = tl.load(zeros_ptr + g_idx * N + offs_n[None, :], mask=mask_w, other=0.0)

        w_fp16 = (w_u4.to(tl.float32) - zero) * scale

        acc += tl.dot(x, w_fp16.to(tl.float16))

        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += (BLOCK_SIZE_K // 2) * stride_wk

    mask_y = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(tl.float16), mask=mask_y)


class INT4DequantGEMMModule(nn.Module):
    def __init__(self, m: int, n: int, k: int, group_size: int = 128):
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.group_size = group_size

    def forward(self, x: torch.Tensor, w_packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor):
        y = torch.empty((self.m, self.n), device=x.device, dtype=torch.float16)
        grid = (triton.cdiv(self.m, 64), triton.cdiv(self.n, 64))

        int4_dequant_gemm_kernel[grid](
            x, w_packed, scales, zeros, y,
            self.m, self.n, self.k,
            self.k, 1,
            self.n, 1,
            self.n, 1,
            group_size=self.group_size,
            BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64
        )
        return y


def benchmark_fn(fn, warmups=20, iters=100):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return ((end - start) / iters) * 1000.0  # ms


def main():
    print("=" * 80)
    print("🔥 RTX 5060 Ti FP8 & INT4 QUANTIZATION PERFORMANCE BENCHMARK")
    print("=" * 80)

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU Hardware: {device_name}")

    results = {}

    # -------------------------------------------------------------------------
    # Benchmark 1: FP8 Scaled GEMM (4096 x 4096 x 4096)
    # -------------------------------------------------------------------------
    M, N, K = 4096, 4096, 4096
    print(f"\n[1/2] Benchmarking FP8 GEMM ({M}x{K} @ {K}x{N})...")
    
    a_fp16 = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b_fp16 = torch.randn((K, N), device="cuda", dtype=torch.float16)
    
    a_fp8 = a_fp16.to(torch.float8_e4m3fn)
    b_fp8 = b_fp16.to(torch.float8_e4m3fn)
    
    fp8_model = FP8ScaledGEMMModule(M, N, K, scale=1.0).cuda()
    
    # 1. PyTorch Eager FP16
    t_eager_fp16 = benchmark_fn(lambda: torch.matmul(a_fp16, b_fp16))
    print(f"  -> PyTorch Eager FP16: {t_eager_fp16:.3f} ms")

    # 2. Native Triton FP8
    t_triton_fp8 = benchmark_fn(lambda: fp8_model(a_fp8, b_fp8))
    print(f"  -> Native Triton FP8:  {t_triton_fp8:.3f} ms")

    # 3. KernelLens FP8 C++ ONNX Runtime Plugin
    compiled_fp8 = kl.compile(fp8_model, (a_fp8, b_fp8), name="Bench_FP8_GEMM", backends=["onnx"])
    t_kl_fp8 = benchmark_fn(lambda: compiled_fp8.run((a_fp8, b_fp8), backend="onnx"))
    print(f"  -> KernelLens ORT FP8: {t_kl_fp8:.3f} ms")

    speedup_fp8 = t_eager_fp16 / t_kl_fp8
    print(f"  🚀 KernelLens FP8 Speedup vs Eager FP16: {speedup_fp8:.2f}x")

    results["fp8_gemm"] = {
        "eager_fp16_ms": t_eager_fp16,
        "triton_fp8_ms": t_triton_fp8,
        "kernel_lens_fp8_ms": t_kl_fp8,
        "speedup_vs_fp16": speedup_fp8
    }

    # -------------------------------------------------------------------------
    # Benchmark 2: INT4 W4A16 Dequantization GEMM (2048 x 4096 x 4096)
    # -------------------------------------------------------------------------
    M_i4, N_i4, K_i4 = 2048, 4096, 4096
    group_size = 128
    print(f"\n[2/2] Benchmarking INT4 W4A16 Dequant GEMM ({M_i4}x{K_i4} @ {K_i4}x{N_i4})...")

    x_fp16 = torch.randn((M_i4, K_i4), device="cuda", dtype=torch.float16)
    w_packed = torch.randint(0, 255, (K_i4 // 2, N_i4), device="cuda", dtype=torch.uint8)
    scales = torch.randn((K_i4 // group_size, N_i4), device="cuda", dtype=torch.float16)
    zeros = torch.randn((K_i4 // group_size, N_i4), device="cuda", dtype=torch.float16)

    w_unpacked = torch.randn((K_i4, N_i4), device="cuda", dtype=torch.float16)

    int4_model = INT4DequantGEMMModule(M_i4, N_i4, K_i4, group_size=group_size).cuda()

    # 1. PyTorch Eager Unquantized FP16 Linear
    t_eager_int4 = benchmark_fn(lambda: torch.matmul(x_fp16, w_unpacked))
    print(f"  -> PyTorch Eager FP16 Linear: {t_eager_int4:.3f} ms")

    # 2. Native Triton INT4 W4A16
    t_triton_int4 = benchmark_fn(lambda: int4_model(x_fp16, w_packed, scales, zeros))
    print(f"  -> Native Triton INT4 W4A16:  {t_triton_int4:.3f} ms")

    # 3. KernelLens INT4 C++ ONNX Runtime Plugin
    compiled_int4 = kl.compile(int4_model, (x_fp16, w_packed, scales, zeros), name="Bench_INT4_GEMM", backends=["onnx"])
    t_kl_int4 = benchmark_fn(lambda: compiled_int4.run((x_fp16, w_packed, scales, zeros), backend="onnx"))
    print(f"  -> KernelLens ORT INT4 W4A16: {t_kl_int4:.3f} ms")

    speedup_int4 = t_eager_int4 / t_kl_int4
    print(f"  🚀 KernelLens INT4 Speedup vs Eager FP16: {speedup_int4:.2f}x")

    results["int4_gemm"] = {
        "eager_fp16_ms": t_eager_int4,
        "triton_int4_ms": t_triton_int4,
        "kernel_lens_int4_ms": t_kl_int4,
        "speedup_vs_fp16": speedup_int4
    }

    # Save benchmark metrics to json
    with open("quantization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("📊 SUMMARY OF QUANTIZATION BENCHMARKS ON RTX 5060 Ti:")
    print(f"  FP8 Scaled GEMM (4096):   Eager FP16: {t_eager_fp16:.2f} ms | ORT FP8: {t_kl_fp8:.2f} ms ({speedup_fp8:.2f}x Speedup)")
    print(f"  INT4 W4A16 GEMM (2048):   Eager FP16: {t_eager_int4:.2f} ms | ORT INT4: {t_kl_int4:.2f} ms ({speedup_int4:.2f}x Speedup)")
    print("=" * 80)


if __name__ == "__main__":
    main()
