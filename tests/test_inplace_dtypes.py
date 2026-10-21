"""Regression test for in-place buffer aliasing across element widths.

The generated in-place D2D copy historically assumed ``float`` (4 bytes/element),
which over-copied for fp16/bf16/fp8/int8 buffers. This test exercises in-place
mutation for several dtypes on both the ONNX Runtime and TensorRT backends.
"""
import pytest
import torch
import triton
import triton.language as tl
import kernel_lens as kl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


@triton.jit
def inplace_add_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(x_ptr + offsets, x + 5.0, mask=mask)


class InplaceAddModule(torch.nn.Module):
    def forward(self, x):
        n = x.numel()
        grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
        inplace_add_kernel[grid](x, n, BLOCK_SIZE=64)
        return x


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("backend", ["onnx", "tensorrt"])
def test_inplace_dtype_parity(dtype, backend):
    name = f"InplaceDtype_{str(dtype).split('.')[-1]}_{backend}"
    x = torch.randn(2048, device="cuda", dtype=dtype)

    native = InplaceAddModule().cuda()
    out_triton = native(x.clone())

    compiled = kl.compile(
        InplaceAddModule().cuda(),
        (x.clone(),),
        name=name,
        backends=[backend],
    )
    out = compiled.run((x.clone(),), backend=backend)
    out = out[0] if isinstance(out, (tuple, list)) else out

    max_diff = (out_triton.float() - out.float()).abs().max().item()
    print(f"[{dtype} / {backend}] max diff vs native Triton: {max_diff:e}")
    assert max_diff < 1e-2, f"{dtype} {backend} in-place parity failed: {max_diff}"


if __name__ == "__main__":
    for dt in (torch.float32, torch.float16):
        test_inplace_dtype_parity(dt, "onnx")