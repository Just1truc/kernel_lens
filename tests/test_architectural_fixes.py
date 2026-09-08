import torch
import triton
import triton.language as tl
import kernel_lens as kl
import os

# --- Test Case 1: Nested Helper Function Store ---
@triton.jit
def _helper_store(ptr, offsets, values, mask):
    tl.store(ptr + offsets, values, mask=mask)

@triton.jit
def nested_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    val = tl.load(x_ptr + offsets, mask=mask)
    _helper_store(out_ptr, offsets, val * 3.0, mask)

class NestedStoreModule(torch.nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
        nested_kernel[grid](x, out, n, BLOCK=64)
        return out

def test_nested_store():
    print("Testing Test Case 1: Nested Helper Function Store...")
    x = torch.randn(128, device='cuda')
    m = NestedStoreModule().cuda()
    comp = kl.compile(m, (x,), name="test_nested_store", backends=["onnx"])
    res = comp.run((x,), backend="onnx")
    diff = torch.max(torch.abs((x * 3.0) - res)).item()
    print(f"  Nested Store Max Diff: {diff:e}")
    assert diff < 1e-5
    print("  ✅ Passed Test Case 1!")

# --- Test Case 2: Dynamic Scalar Attribute Updates ---
@triton.jit
def scale_kernel(x_ptr, out_ptr, alpha, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * alpha, mask=mask)

class ScaleModule(torch.nn.Module):
    def forward(self, x, alpha: float):
        out = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
        scale_kernel[grid](x, out, alpha, n, BLOCK=64)
        return out

def test_dynamic_scalars():
    print("Testing Test Case 2: Dynamic Runtime Scalar Updates...")
    x = torch.randn(128, device='cuda')
    m = ScaleModule().cuda()
    comp = kl.compile(m, (x, 2.0), name="test_dynamic_scalars", backends=["onnx"])
    
    # Run with initial scalar (alpha = 2.0)
    res1 = comp.run((x, 2.0), backend="onnx")
    diff1 = torch.max(torch.abs((x * 2.0) - res1)).item()
    
    # Run with NEW dynamic scalar (alpha = 5.5) without recompiling!
    res2 = comp.run((x, 5.5), backend="onnx")
    diff2 = torch.max(torch.abs((x * 5.5) - res2)).item()
    
    print(f"  Dynamic Scalar Alpha=2.0 Max Diff: {diff1:e}")
    print(f"  Dynamic Scalar Alpha=5.5 Max Diff: {diff2:e}")
    assert diff1 < 1e-5 and diff2 < 1e-5
    print("  ✅ Passed Test Case 2!")

# --- Test Case 3: 4D Tensor High-Rank Launch Grid ---
@triton.jit
def rank4_kernel(x_ptr, out_ptr, N, C, H, W, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    total = N * C * H * W
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    val = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, val + 10.0, mask=mask)

class Rank4Module(torch.nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        N, C, H, W = x.shape
        total = N * C * H * W
        grid = lambda meta: (triton.cdiv(total, meta['BLOCK']),)
        rank4_kernel[grid](x, out, N, C, H, W, BLOCK=128)
        return out

def test_high_rank_grid():
    print("Testing Test Case 3: 4D High-Rank Tensor Launch Grid...")
    x = torch.randn(2, 16, 8, 8, device='cuda')
    m = Rank4Module().cuda()
    comp = kl.compile(m, (x,), name="test_rank4_grid", backends=["onnx"])
    res = comp.run((x,), backend="onnx")
    diff = torch.max(torch.abs((x + 10.0) - res)).item()
    print(f"  High Rank Grid Max Diff: {diff:e}")
    assert diff < 1e-5
    print("  ✅ Passed Test Case 3!")

if __name__ == "__main__":
    test_nested_store()
    test_dynamic_scalars()
    test_high_rank_grid()
    print("\n🎉 ALL ARCHITECTURAL FIX VERIFICATION TESTS PASSED SUCCESSFULLY!")
