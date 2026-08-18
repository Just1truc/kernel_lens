import os
import torch
import torch.nn as nn
import triton
import triton.language as tl
import kernel_lens as kl

# =====================================================================
# KERNEL 1: Fused LayerNorm (Reduction: tl.sum, tl.sqrt, float32)
# =====================================================================
@triton.jit
def _fused_layernorm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, M,
    stride_x_n, stride_x_m,
    stride_out_n, stride_out_m,
    eps: tl.constexpr = 1e-5,
    BLOCK_M: tl.constexpr = 128
):
    pid = tl.program_id(0)
    row_idx = pid
    if row_idx >= N:
        return
    
    col_offs = tl.arange(0, BLOCK_M)
    mask = col_offs < M
    
    x_ptrs = x_ptr + row_idx * stride_x_n + col_offs * stride_x_m
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    mean = tl.sum(x, axis=0) / M
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / M
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    w_ptrs = weight_ptr + col_offs
    b_ptrs = bias_ptr + col_offs
    weight = tl.load(w_ptrs, mask=mask, other=1.0)
    bias = tl.load(b_ptrs, mask=mask, other=0.0)
    
    out = (x - mean) * inv_std * weight + bias
    out_ptrs = out_ptr + row_idx * stride_out_n + col_offs * stride_out_m
    tl.store(out_ptrs, out, mask=mask)

class FusedLayerNormModel(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.weight = nn.Parameter(torch.ones(M))
        self.bias = nn.Parameter(torch.zeros(M))

    def forward(self, x):
        N, M = x.shape
        out = torch.empty_like(x)
        grid = (N,)
        _fused_layernorm_kernel[grid](
            x, self.weight, self.bias, out,
            N, M,
            x.stride(0), x.stride(1),
            out.stride(0), out.stride(1),
            eps=1e-5, BLOCK_M=128
        )
        return out

# =====================================================================
# KERNEL 2: Fused GEMM + Sigmoid Epilogue (tl.dot, constexpr tile sizes)
# =====================================================================
@triton.jit
def _fused_gemm_sigmoid_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 32
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K
        a_mask = (offs_m[:, None] < M) & k_mask[None, :]
        b_mask = k_mask[:, None] & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Epilogue: Sigmoid activation
    acc = 1.0 / (1.0 + tl.exp(-acc))

    out_ptrs = out_ptr + (offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)

class FusedGemmSigmoidModel(nn.Module):
    def __init__(self, K, N):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(K, N))

    def forward(self, x):
        M, K = x.shape
        _, N = self.weight.shape
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        _fused_gemm_sigmoid_kernel[grid](
            x, self.weight, out,
            M, N, K,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )
        return out

# =====================================================================
# TEST RUNNER & PRECISION VALIDATOR
# =====================================================================
def run_precision_tests():
    device = "cuda"
    torch.manual_seed(42)

    print("=====================================================================")
    print("      RUNNING KERNEL-LENS DEEPLY INGRAINED TRITON PRECISION TESTS    ")
    print("=====================================================================")

    # --- Test 1: LayerNorm Kernel ---
    print("\n[TEST 1] Fused LayerNorm Kernel (Reduction, Exp/Sqrt)...")
    N, M = 32, 128
    x1 = torch.randn(N, M, device=device)
    model1 = FusedLayerNormModel(M).to(device).eval()

    with torch.no_grad():
        out1_triton = model1(x1)

    kl_model1 = kl.compile(model1, (x1,), name="LayerNorm_Test", backends=["tensorrt"])
    trt_out1 = kl_model1.run((x1,), backend="tensorrt")[0]

    diff_trt1 = (out1_triton - torch.as_tensor(trt_out1, device=device)).abs().max().item()

    print(f"  -> LayerNorm Max Diff (Triton vs TRT): {diff_trt1:e}")
    assert diff_trt1 < 1e-4, f"LayerNorm TRT Precision test failed with diff {diff_trt1}"

    # --- Test 2: Fused GEMM + Sigmoid Kernel ---
    print("\n[TEST 2] Fused GEMM + Sigmoid Kernel (tl.dot + Epilogue)...")
    M, K, N = 128, 64, 128
    x2 = torch.randn(M, K, device=device)
    model2 = FusedGemmSigmoidModel(K, N).to(device).eval()

    with torch.no_grad():
        out2_triton = model2(x2)

    kl_model2 = kl.compile(model2, (x2,), name="Gemm_Sigmoid_Test", backends=["tensorrt"])
    trt_out2 = kl_model2.run((x2,), backend="tensorrt")[0]

    diff_trt2 = (out2_triton - torch.as_tensor(trt_out2, device=device)).abs().max().item()

    print(f"  -> Fused GEMM Max Diff (Triton vs TRT): {diff_trt2:e}")
    assert diff_trt2 < 1e-4, f"Fused GEMM TRT Precision test failed with diff {diff_trt2}"

    print("\n=====================================================================")
    print("      ALL KERNEL-LENS PRECISION TESTS PASSED WITH 100% SUCCESS!      ")
    print("=====================================================================\n")

if __name__ == "__main__":
    run_precision_tests()
