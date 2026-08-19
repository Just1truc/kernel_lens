import torch
import torch.nn as nn
import triton
import triton.language as tl
import time
import kernel_lens as kl

# ============================================================================
# 1. FUSED TRITON KERNEL: Fused Residual Add + RMSNorm (LLaMA/Mistral Style)
# ============================================================================
@triton.jit
def fused_residual_rms_norm_kernel(
    x_ptr, residual_ptr, weight_ptr,
    out_res_ptr, out_norm_ptr,
    D: tl.constexpr, EPS: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_offset = row_idx * D
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D
    
    # Load input x, residual, and weight ONCE into registers
    x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
    res = tl.load(residual_ptr + row_offset + cols, mask=mask, other=0.0)
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    
    # 1. Fuse Residual Addition
    new_res = x + res
    tl.store(out_res_ptr + row_offset + cols, new_res, mask=mask)
    
    # 2. Compute RMSNorm in SRAM/Registers (No re-reading from VRAM)
    x_sq = new_res * new_res
    mean_sq = tl.sum(x_sq, axis=0) / D
    rsqrt = tl.rsqrt(mean_sq + EPS)
    
    out_norm = new_res * rsqrt * w
    tl.store(out_norm_ptr + row_offset + cols, out_norm, mask=mask)


class FusedResidualRMSNormModule(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor):
        shape = x.shape
        n_rows = shape[0] * shape[1]
        
        out_res = torch.empty_like(residual)
        out_norm = torch.empty_like(x)
        
        grid = (n_rows,)
        BLOCK_SIZE = triton.next_power_of_2(self.hidden_dim)
        
        fused_residual_rms_norm_kernel[grid](
            x, residual, weight,
            out_res, out_norm,
            D=self.hidden_dim, EPS=self.eps, BLOCK_SIZE=BLOCK_SIZE
        )
        return out_res, out_norm


# Native PyTorch reference implementation (unfused operations)
def native_fused_residual_rms(x, residual, weight, eps=1e-5):
    new_res = x + residual
    variance = new_res.pow(2).mean(-1, keepdim=True)
    norm_out = new_res * torch.rsqrt(variance + eps) * weight
    return new_res, norm_out


def benchmark_fn(fn, args, runs=100, warmup=20):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs * 1000.0


def main():
    print("=" * 75)
    print("⚡ DEMO: Custom Triton Fusion vs PyTorch / torch.compile")
    print("Scenario: Fused Residual Addition + RMSNorm (LLaMA/Qwen Layer)")
    print("=" * 75)

    B, T, D = 16, 512, 4096  # Typical LLM Layer Activation Tensor Shape
    print(f"Tensor Shape: Batch={B}, SeqLen={T}, HiddenDim={D} (Total Elements per Tensor: {B*T*D:,})")

    torch.manual_seed(42)
    x = torch.randn(B, T, D, device="cuda", dtype=torch.float32)
    residual = torch.randn(B, T, D, device="cuda", dtype=torch.float32)

    model = FusedResidualRMSNormModule(hidden_dim=D).cuda()

    # 1. Native PyTorch (Eager)
    eager_lat = benchmark_fn(native_fused_residual_rms, (x, residual, model.weight))
    print(f"  [1] PyTorch Eager:             {eager_lat:.4f} ms")

    # 2. PyTorch torch.compile (Inductor)
    print("  [2] Compiling with torch.compile (Inductor)...")
    compiled_native = torch.compile(native_fused_residual_rms)
    for _ in range(5): compiled_native(x, residual, model.weight)
    tc_lat = benchmark_fn(compiled_native, (x, residual, model.weight))
    print(f"  [2] torch.compile (Inductor):  {tc_lat:.4f} ms")

    # 3. Native Triton (Python)
    triton_lat = benchmark_fn(model, (x, residual, model.weight))
    print(f"  [3] Native Triton (Python):    {triton_lat:.4f} ms")

    # 4. Kernel Lens -> TensorRT Plugin
    print("  [4] Compiling with Kernel Lens (TensorRT Plugin)...")
    kl_trt = kl.compile(model, (x, residual, model.weight), backends=["tensorrt"], name="FusedResRMS_TRT")
    trt_lat = benchmark_fn(lambda: kl_trt.run((x, residual, model.weight), backend="tensorrt"), ())
    trt_out_res, trt_out_norm = kl_trt.run((x, residual, model.weight), backend="tensorrt")
    print(f"  [4] Kernel Lens (TensorRT):    {trt_lat:.4f} ms")

    # 5. Kernel Lens -> ONNX Runtime Plugin
    print("  [5] Compiling with Kernel Lens (ONNX Runtime Plugin)...")
    kl_ort = kl.compile(model, (x, residual, model.weight), backends=["onnx"], name="FusedResRMS_ORT")
    ort_lat = benchmark_fn(lambda: kl_ort.run((x, residual, model.weight), backend="onnx"), ())
    ort_out_res, ort_out_norm = kl_ort.run((x, residual, model.weight), backend="onnx")
    print(f"  [5] Kernel Lens (ONNX Runtime): {ort_lat:.4f} ms")

    # Check Numerical Parity
    py_res, py_norm = native_fused_residual_rms(x, residual, model.weight)
    diff_res_trt = torch.abs(py_res - trt_out_res).max().item()
    diff_norm_trt = torch.abs(py_norm - trt_out_norm).max().item()
    diff_res_ort = torch.abs(py_res - ort_out_res).max().item()
    diff_norm_ort = torch.abs(py_norm - ort_out_norm).max().item()

    print("\n" + "=" * 75)
    print("📊 SPEEDUP COMPARISON")
    print("=" * 75)
    print(f"  🚀 Kernel Lens (TRT) vs torch.compile:  {(tc_lat / trt_lat):.2f}x SPEEDUP")
    print(f"  🚀 Kernel Lens (TRT) vs PyTorch Eager:  {(eager_lat / trt_lat):.2f}x SPEEDUP")
    print(f"  🚀 Kernel Lens (ORT) vs torch.compile:  {(tc_lat / ort_lat):.2f}x SPEEDUP")
    print("\n🎯 NUMERICAL ACCURACY VERIFICATION")
    print(f"  -> TRT Residual Output Max Diff:    {diff_res_trt:.6e}  ({'✅ PASSED' if diff_res_trt < 1e-5 else '❌ FAILED'})")
    print(f"  -> TRT Norm Output Max Diff:        {diff_norm_trt:.6e}  ({'✅ PASSED' if diff_norm_trt < 1e-5 else '❌ FAILED'})")
    print(f"  -> ORT Residual Output Max Diff:    {diff_res_ort:.6e}  ({'✅ PASSED' if diff_res_ort < 1e-5 else '❌ FAILED'})")
    print(f"  -> ORT Norm Output Max Diff:        {diff_norm_ort:.6e}  ({'✅ PASSED' if diff_norm_ort < 1e-5 else '❌ FAILED'})")
    print("=" * 75)


if __name__ == "__main__":
    main()
