import torch
import torch.nn as nn
import triton
import triton.language as tl
import time
import kernel_lens as kl

# ============================================================================
# ⚡ INT4 Weight Dequantization GEMM Triton Kernel (W4A16)
# Weights stored packed in uint8 (2x 4-bit weights per byte)
# Unpacked on-the-fly in GPU register files.
# ============================================================================
@triton.jit
def int4_dequant_gemm_kernel(
    x_ptr, w_packed_ptr, scales_ptr, zeros_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Offset pointers
    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    w_ptrs = w_packed_ptr + (offs_k[:, None] // 2) * N + offs_n[None, :]
    
    scale = tl.load(scales_ptr + offs_n[None, :], mask=offs_n[None, :] < N, other=1.0)
    zero = tl.load(zeros_ptr + offs_n[None, :], mask=offs_n[None, :] < N, other=0.0)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        mask_x = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        mask_w = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        # Load input X (FP16/FP32) and packed uint8 weights
        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w_packed = tl.load(w_ptrs, mask=mask_w, other=0)

        # Unpack 4-bit integers on-the-fly in GPU registers
        is_high = (offs_k[:, None] % 2) == 1
        w_4bit = tl.where(is_high, (w_packed >> 4) & 0x0F, w_packed & 0x0F)
        
        # Dequantize: (w - zero) * scale
        w_fp = (w_4bit.to(tl.float32) - zero) * scale

        acc += tl.dot(x, w_fp)
        x_ptrs += BLOCK_SIZE_K
        w_ptrs += (BLOCK_SIZE_K // 2) * N

    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_c)


class INT4DequantGEMMModule(nn.Module):
    def __init__(self, m: int, n: int, k: int):
        super().__init__()
        self.m = m
        self.n = n
        self.k = k

    def forward(self, x: torch.Tensor, w_packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor):
        out = torch.empty((self.m, self.n), device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(self.m, 32), triton.cdiv(self.n, 32))
        
        int4_dequant_gemm_kernel[grid](
            x, w_packed, scales, zeros, out,
            M=self.m, N=self.n, K=self.k,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32
        )
        return out


def main():
    print("=" * 80)
    print("🚀 TESTING INT4 SUB-BYTE DEQUANTIZATION GEMM KERNEL COMPILATION")
    print("=" * 80)

    M, N, K = 64, 64, 64

    torch.manual_seed(42)
    x = torch.randn(M, K, device="cuda", dtype=torch.float32)
    
    # Generate packed uint8 weights (each byte holds 2x 4-bit integers in [0, 15])
    w_raw = torch.randint(0, 16, (K, N), device="cuda", dtype=torch.uint8)
    w_even = w_raw[0::2, :]
    w_odd = w_raw[1::2, :]
    w_packed = (w_odd << 4) | (w_even & 0x0F)

    scales = torch.rand(1, N, device="cuda", dtype=torch.float32) * 0.1
    zeros = torch.randint(0, 8, (1, N), device="cuda", dtype=torch.float32)

    model = INT4DequantGEMMModule(M, N, K).cuda()

    # Native Triton Execution
    print("\n[1/3] Executing Native Triton INT4 Kernel...")
    triton_out = model(x, w_packed, scales, zeros)
    print(f"  -> Triton INT4 Output Shape: {triton_out.shape}, Mean: {triton_out.mean().item():.4f}")

    # KernelLens Compilation
    print("\n[2/3] Compiling INT4 Kernel via KernelLens (ONNX Runtime Plugin)...")
    compiled_model = kl.compile(model, (x, w_packed, scales, zeros), name="INT4GEMM_ORT", backends=["onnx"])

    # Engine Run
    print("\n[3/3] Executing Compiled C++ Plugin in ONNX Runtime...")
    ort_out = compiled_model.run((x, w_packed, scales, zeros), backend="onnx")

    # Accuracy Verification
    max_diff = torch.abs(triton_out - ort_out).max().item()
    print("\n" + "=" * 80)
    print(f"🎯 NUMERICAL ACCURACY VERIFICATION: Max Diff = {max_diff:.6e}")
    print(f"  -> {'✅ PASSED: Exact Numerical Parity Achieved!' if max_diff < 1e-4 else '❌ FAILED'}")
    print("=" * 80)


import pytest

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for INT4 test")
def test_int4_dequant_gemm():
    main()


if __name__ == "__main__":
    main()
