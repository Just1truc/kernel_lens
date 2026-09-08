import torch
import triton
import triton.language as tl
import kernel_lens as kl

@triton.jit
def inplace_add_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # In-place operation: Load from x_ptr, add 5.0, and Store back into x_ptr
    x = tl.load(x_ptr + offsets, mask=mask)
    x_new = x + 5.0
    tl.store(x_ptr + offsets, x_new, mask=mask)

class InplaceAddModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n_elements = x.numel()
        grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
        inplace_add_kernel[grid](x, n_elements, BLOCK_SIZE=64)
        return x

def test_inplace_onnx_parity():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA not available. Skipping inplace test.")
        return

    # Baseline PyTorch input
    x = torch.randn(1024, device=device, dtype=torch.float32)

    # Native Triton execution
    x_triton = x.clone()
    model_native = InplaceAddModule().to(device)
    out_triton = model_native(x_triton)

    # KernelLens ONNX Compilation (pass fresh clone for dummy tracing)
    model_onnx = InplaceAddModule().to(device)
    x_dummy = x.clone()
    compiled_model = kl.compile(model_onnx, (x_dummy,), name="InplaceAddKernel", backends=["onnx"])
    
    x_eval = x.clone()
    out_onnx = compiled_model.run((x_eval,), backend="onnx")
    out_onnx = out_onnx[0] if isinstance(out_onnx, (tuple, list)) else out_onnx

    max_diff = torch.max(torch.abs(out_triton - out_onnx)).item()
    print(f"Max Diff Inplace Triton vs ONNX: {max_diff:e}")
    assert max_diff < 1e-5, f"Inplace test failed! Max diff: {max_diff}"
    print("🎉 INPLACE ONNX PARITY TEST PASSED!")

if __name__ == "__main__":
    test_inplace_onnx_parity()
