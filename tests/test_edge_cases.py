import torch
import torch.nn as nn
import triton
import triton.language as tl
import kernel_lens as kl
import time
import os

# ============================================================================
# TRICKY / EDGE CASE TRITON KERNELS
# ============================================================================

# ----------------------------------------------------------------------------
# EDGE CASE 1: Multi-Output with Different Output Shapes & Ranks
# Output 1 (out_ptr): Shape (B, H, N, D)
# Output 2 (lse_ptr): Shape (B, H, N)  <-- Rank 3 vs Rank 4!
# ----------------------------------------------------------------------------
@triton.jit
def fused_softmax_lse_kernel(
    x_ptr, out_ptr, lse_ptr,
    stride_b, stride_h, stride_n, stride_d,
    lse_stride_b, lse_stride_h, lse_stride_n,
    B, H, N, D,
    scale,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    offsets_d = tl.arange(0, BLOCK_D)
    mask_d = offsets_d < D

    in_offset = pid_b * stride_b + pid_h * stride_h + pid_n * stride_n + offsets_d * stride_d
    vals = tl.load(x_ptr + in_offset, mask=mask_d, other=-1e9) * scale

    max_val = tl.max(vals, axis=0)
    shifted = vals - max_val
    exp_vals = tl.exp(shifted)
    sum_exp = tl.sum(exp_vals, axis=0)
    lse_val = max_val + tl.log(sum_exp)

    softmax_vals = exp_vals / sum_exp
    tl.store(out_ptr + in_offset, softmax_vals, mask=mask_d)

    lse_offset = pid_b * lse_stride_b + pid_h * lse_stride_h + pid_n * lse_stride_n
    tl.store(lse_ptr + lse_offset, lse_val)

class SoftmaxLSEModule(nn.Module):
    def __init__(self, scale=0.125):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        B, H, N, D = x.shape
        out = torch.empty_like(x)
        lse = torch.empty((B, H, N), device=x.device, dtype=x.dtype)
        
        grid = (B, H, N)
        fused_softmax_lse_kernel[grid](
            x, out, lse,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            B, H, N, D,
            self.scale,
            BLOCK_D=64
        )
        return out, lse

def native_softmax_lse(x, scale=0.125):
    scaled_x = x * scale
    lse = torch.logsumexp(scaled_x, dim=-1)
    out = torch.softmax(scaled_x, dim=-1)
    return out, lse


# ----------------------------------------------------------------------------
# EDGE CASE 2: Fused Dequantization / Int8 GEMM + FP32 Accumulation + Bias
# Mixed pointer types: Int8 inputs, FP32 output, FP32 bias, Int32/Float scalars
# ----------------------------------------------------------------------------
@triton.jit
def int8_dequant_bias_gemm_kernel(
    a_ptr, b_ptr, bias_ptr, out_ptr,
    M, N, K,
    scale_a, scale_b,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_idx in range(0, K, BLOCK_K):
        mask_a = (offs_m[:, None] < M) & ((k_idx + offs_k[None, :]) < K)
        mask_b = ((k_idx + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        a_vals = tl.load(a_ptrs, mask=mask_a, other=0).to(tl.float32)
        b_vals = tl.load(b_ptrs, mask=mask_b, other=0).to(tl.float32)

        acc += tl.dot(a_vals, b_vals)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = acc * (scale_a * scale_b)

    mask_bias = offs_n < N
    bias_vals = tl.load(bias_ptr + offs_n, mask=mask_bias, other=0.0)
    acc = acc + bias_vals[None, :]

    # Silu activation fused
    acc = acc * tl.sigmoid(acc)

    out_ptrs = out_ptr + (offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn)
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask_out)

class Int8DequantGEMMModule(nn.Module):
    def __init__(self, M, N, K, scale_a=0.02, scale_b=0.03):
        super().__init__()
        self.M, self.N, self.K = M, N, K
        self.scale_a = scale_a
        self.scale_b = scale_b

    def forward(self, a_int8, b_int8, bias):
        out = torch.empty((self.M, self.N), device=a_int8.device, dtype=torch.float32)
        grid = (triton.cdiv(self.M, 32), triton.cdiv(self.N, 32))
        
        int8_dequant_bias_gemm_kernel[grid](
            a_int8, b_int8, bias, out,
            self.M, self.N, self.K,
            self.scale_a, self.scale_b,
            a_int8.stride(0), a_int8.stride(1),
            b_int8.stride(0), b_int8.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=32, BLOCK_N=32, BLOCK_K=32
        )
        return out

def native_int8_dequant_gemm(a_int8, b_int8, bias, scale_a=0.02, scale_b=0.03):
    a_fp = a_int8.to(torch.float32) * scale_a
    b_fp = b_int8.to(torch.float32) * scale_b
    res = a_fp @ b_fp + bias
    return res * torch.sigmoid(res)


# ----------------------------------------------------------------------------
# EDGE CASE 3: Fused SwiGLU Gated MLP with Multiple Scalars
# ----------------------------------------------------------------------------
@triton.jit
def swiglu_fused_kernel(
    x_ptr, w_gate_ptr, w_up_ptr, out_ptr,
    N, D,
    alpha, beta,
    stride_xn, stride_xd,
    stride_outn, stride_outd,
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask = offs_d < D

    x_offset = pid * stride_xn + offs_d * stride_xd
    x = tl.load(x_ptr + x_offset, mask=mask, other=0.0)

    w_g = tl.load(w_gate_ptr + offs_d, mask=mask, other=1.0)
    w_u = tl.load(w_up_ptr + offs_d, mask=mask, other=1.0)

    gate = (x * w_g) * alpha
    up = (x * w_u) * beta

    sig_gate = gate * tl.sigmoid(gate)
    res = sig_gate * up

    out_offset = pid * stride_outn + offs_d * stride_outd
    tl.store(out_ptr + out_offset, res, mask=mask)

class SwiGLUModule(nn.Module):
    def __init__(self, alpha=1.2, beta=0.8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, w_gate, w_up):
        N, D = x.shape
        out = torch.empty_like(x)
        grid = (N,)
        swiglu_fused_kernel[grid](
            x, w_gate, w_up, out,
            N, D,
            self.alpha, self.beta,
            x.stride(0), x.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_D=128
        )
        return out

def native_swiglu(x, w_gate, w_up, alpha=1.2, beta=0.8):
    gate = (x * w_gate) * alpha
    up = (x * w_up) * beta
    return (gate * torch.sigmoid(gate)) * up


# ============================================================================
# BENCHMARK & TESTING HARNESS
# ============================================================================
def benchmark_latency(fn, args, kwargs=None, warmup=10, rep=50):
    if kwargs is None: kwargs = {}
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(rep):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.time() - start) / rep * 1000.0

def run_edge_case_test(model, inputs, name, native_fn, verbose=False):
    print(f"\n======================================================================")
    print(f"🔥 EDGE-CASE BENCHMARK: {name}")
    print(f"======================================================================")

    # 1. Native PyTorch
    print("[1/4] Benchmarking Native PyTorch Baseline...")
    native_out = native_fn(*inputs)
    native_lat = benchmark_latency(native_fn, inputs)

    # 2. Native Triton
    print("[2/4] Benchmarking Native Triton (Python)...")
    triton_out = model(*inputs)
    triton_lat = benchmark_latency(model, inputs)

    # 3. Kernel Lens -> TensorRT
    print("[3/4] Kernel Lens -> TensorRT Plugin...")
    kl_model_trt = kl.compile(model, inputs, name=f"{name}_trt", backends=["tensorrt"], verbose=verbose)
    with torch.no_grad():
        trt_out = kl_model_trt.run(inputs, backend="tensorrt")
    trt_lat = benchmark_latency(kl_model_trt.run, (inputs,), {"backend": "tensorrt"})

    # 4. Kernel Lens -> ONNX Runtime
    print("[4/4] Kernel Lens -> ONNX Runtime Plugin...")
    kl_model_ort = kl.compile(model, inputs, name=f"{name}_ort", backends=["onnx"], verbose=verbose)
    with torch.no_grad():
        ort_out = kl_model_ort.run(inputs, backend="onnx")
    ort_lat = benchmark_latency(kl_model_ort.run, (inputs,), {"backend": "onnx"})

    # Compute Max Diff
    def calc_diff(a, b):
        if isinstance(a, (tuple, list)):
            return max(torch.max(torch.abs(a_i - b_i)).item() for a_i, b_i in zip(a, b))
        return torch.max(torch.abs(a - b)).item()

    err_trt = calc_diff(triton_out, trt_out)
    err_ort = calc_diff(triton_out, ort_out)
    max_err = max(err_trt, err_ort)

    speedup_trt = triton_lat / trt_lat if trt_lat > 0 else 0

    print("\n📊 EDGE CASE RESULTS:")
    print(f"  -> Triton (Python):   {triton_lat:.4f} ms")
    print(f"  -> Kernel Lens (ORT): {ort_lat:.4f} ms")
    print(f"  -> Kernel Lens (TRT): {trt_lat:.4f} ms")
    print(f"  -------------------------------------")
    print(f"  🏆 SPEEDUP TRT vs Triton: {speedup_trt:.2f}x")
    print(f"  -> Max Diff TRT vs Triton: {err_trt:.6e}")
    print(f"  -> Max Diff ORT vs Triton: {err_ort:.6e}")
    if max_err < 1e-4:
        print("  -> Numerical Parity: ✅ PASSED (Bit-wise Exact / Parity OK)")
    else:
        print(f"  -> Numerical Parity: ❌ FAILED (Max Err: {max_err:.6e})")


if __name__ == "__main__":
    torch.manual_seed(42)
    verbose = False  # Set to True to enable debug logging

    # ------------------------------------------------------------------------
    # TEST 1: Softmax + LSE (Multi-Output with Rank 4 and Rank 3 Tensors)
    # ------------------------------------------------------------------------
    lse_model = SoftmaxLSEModule(scale=0.125).cuda()
    x_lse = torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.float32)
    run_edge_case_test(lse_model, (x_lse,), "Multi_Output_Rank4_Rank3_LSE", native_fn=native_softmax_lse, verbose=verbose)

    # ------------------------------------------------------------------------
    # TEST 2: Int8 GEMM + Dequantization + Bias + SiLU Activation
    # ------------------------------------------------------------------------
    M, N, K = 64, 64, 128
    gemm_model = Int8DequantGEMMModule(M, N, K).cuda()
    a_int8 = torch.randint(-128, 127, (M, K), device='cuda', dtype=torch.int8)
    b_int8 = torch.randint(-128, 127, (K, N), device='cuda', dtype=torch.int8)
    bias = torch.randn(N, device='cuda', dtype=torch.float32)
    run_edge_case_test(gemm_model, (a_int8, b_int8, bias), "Int8_Dequant_Bias_SiLU_GEMM", native_fn=native_int8_dequant_gemm, verbose=verbose)

    # ------------------------------------------------------------------------
    # TEST 3: Fused SwiGLU Gated MLP with Multiple Scalars
    # ------------------------------------------------------------------------
    swiglu_model = SwiGLUModule(alpha=1.2, beta=0.8).cuda()
    x_swi = torch.randn(128, 128, device='cuda', dtype=torch.float32)
    w_gate = torch.randn(128, device='cuda', dtype=torch.float32)
    w_up = torch.randn(128, device='cuda', dtype=torch.float32)
    run_edge_case_test(swiglu_model, (x_swi, w_gate, w_up), "SwiGLU_Gated_MLP_MultiScalar", native_fn=native_swiglu, verbose=verbose)

    print("\n🚀 ALL EDGE-CASE STRESS TESTS COMPLETED.")
