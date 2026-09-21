import pytest
import torch
import kernel_lens as kl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")

def test_tensorrt():
    from tests.final import TritonNHWCSequentialDecoder
    device = torch.device('cuda')
    C, H, W = 128, 64, 64
    torch.manual_seed(42)

    x_nchw = torch.randn(1, C, H, W, device=device)
    x_nhwc = x_nchw.contiguous(memory_format=torch.channels_last)

    model = TritonNHWCSequentialDecoder(C).to(device)

    kl_conv = kl.compile(model, (x_nhwc, model.weight), name="NHWC_Conv_SOTA", backends=["tensorrt"])
    trt_out = kl_conv.run((x_nhwc, model.weight), backend="tensorrt")

    with torch.no_grad():
        out_triton = model(x_nhwc, model.weight)

    trt_raw = torch.as_tensor(trt_out[0] if isinstance(trt_out, (tuple, list)) else trt_out, device='cuda', dtype=out_triton.dtype)
    trt_tensor = trt_raw.reshape(1, H, W, 4 * C).permute(0, 3, 1, 2).reshape(1, 4, C, H, W)

    diff_trt = (out_triton - trt_tensor).abs().max().item()
    assert diff_trt < 1e-4, f"TRT parity test failed with max diff: {diff_trt}"


if __name__ == "__main__":
    test_tensorrt()
