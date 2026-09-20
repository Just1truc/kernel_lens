import torch
import torch.nn as nn
import triton
import triton.language as tl
import time
import kernel_lens as kl

# ============================================================================
# ⚡ FP8 Scaled GEMM Triton Kernel (E4M3FN Format)
# ============================================================================
@triton.jit
def fp8_scaled_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    scale: float,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        mask_a = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        mask_b = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        # Load 8-bit float tiles (fp8e4m3fn)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # Perform Tensor Core Dot Product
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N

    # Apply scaling factor and store output in FP16/FP32
    c = acc * scale
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c.to(tl.float32), mask=mask_c)


class FP8ScaledGEMMModule(nn.Module):
    def __init__(self, m: int, n: int, k: int, scale: float = 1.0):
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.scale = scale

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        c = torch.empty((self.m, self.n), device=a.device, dtype=torch.float32)
        grid = (triton.cdiv(self.m, 32), triton.cdiv(self.n, 32))
        
        fp8_scaled_gemm_kernel[grid](
            a, b, c,
            M=self.m, N=self.n, K=self.k,
            scale=self.scale,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32
        )
        return c


def main():
    print("=" * 80)
    print("🚀 TESTING FP8 (E4M3FN) KERNEL COMPILATION ON RTX 5060 Ti GPU")
    print("=" * 80)

    if not hasattr(torch, "float8_e4m3fn"):
        print("❌ PyTorch does not support FP8 float8_e4m3fn on this setup.")
        return

    M, N, K = 128, 128, 128
    scale = 0.5

    torch.manual_seed(42)
    # Generate random FP8 tensors
    a_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b_fp32 = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    a_fp8 = a_fp32.to(torch.float8_e4m3fn)
    b_fp8 = b_fp32.to(torch.float8_e4m3fn)

    model = FP8ScaledGEMMModule(M, N, K, scale=scale).cuda()

    # Native Triton Execution
    print("\n[1/3] Executing Native Triton FP8 Kernel...")
    triton_out = model(a_fp8, b_fp8)
    print(f"  -> Triton FP8 Output Shape: {triton_out.shape}, Mean: {triton_out.mean().item():.4f}")

    # KernelLens Compilation
    print("\n[2/3] Compiling FP8 Kernel via KernelLens (ONNX Runtime Plugin)...")
    compiled_model = kl.compile(model, (a_fp8, b_fp8), name="FP8GEMM_ORT", backends=["onnx"])

    # Engine Run
    print("\n[3/3] Executing Compiled C++ Plugin in ONNX Runtime...")
    ort_out = compiled_model.run((a_fp8, b_fp8), backend="onnx")

    # Accuracy Verification
    max_diff = torch.abs(triton_out - ort_out).max().item()
    print("\n" + "=" * 80)
    print(f"🎯 NUMERICAL ACCURACY VERIFICATION: Max Diff = {max_diff:.6e}")
    print(f"  -> {'✅ PASSED: Exact Numerical Parity Achieved!' if max_diff < 1e-4 else '❌ FAILED'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
