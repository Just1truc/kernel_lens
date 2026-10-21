"""Verifies the generated 16-byte alignment guard on sub-aligned input slices."""
import pytest
import torch
import triton
import triton.language as tl
import kernel_lens as kl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


@triton.jit
def scale_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask, other=0.0) * 2.0, mask=mask)


class Scale(torch.nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        scale_kernel[(triton.cdiv(n, 128),)](x, out, n, BLOCK=128)
        return out


@pytest.mark.parametrize("backend", ["onnx", "tensorrt"])
def test_aligned_input(backend):
    x = torch.randn(4096, device="cuda", dtype=torch.float32)
    assert x.data_ptr() % 16 == 0
    cm = kl.compile(Scale().cuda(), (x,), name=f"AlignOK_{backend}", backends=[backend])
    out = cm.run((x,), backend=backend)
    out = out[0] if isinstance(out, (tuple, list)) else out
    assert (out - x * 2).abs().max().item() < 1e-5


@pytest.mark.parametrize("backend", ["onnx", "tensorrt"])
def test_misaligned_input(backend):
    base = torch.randn(4097, device="cuda", dtype=torch.float32)
    x = base[1:]                       # data_ptr offset by 4 bytes -> not 16B aligned
    assert x.data_ptr() % 16 != 0, "expected a misaligned view"
    cm = kl.compile(Scale().cuda(), (x.clone(),), name=f"AlignBad_{backend}", backends=[backend])
    out = cm.run((x,), backend=backend)
    out = out[0] if isinstance(out, (tuple, list)) else out
    diff = (out - x * 2).abs().max().item()
    print(f"[{backend}] misaligned input max diff: {diff:e}")
    assert diff < 1e-4, f"{backend} alignment guard failed: {diff}"


if __name__ == "__main__":
    test_aligned_input("onnx")
    test_misaligned_input("onnx")