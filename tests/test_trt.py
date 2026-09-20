import os
import sys
import pytest
import torch
import kernel_lens as kl

def is_trt_available():
    if not torch.cuda.is_available():
        return False
    try:
        import tensorrt
        trt_inc_dirs = []
        if os.environ.get("TENSORRT_INCLUDE_DIR"):
            trt_inc_dirs.append(os.environ["TENSORRT_INCLUDE_DIR"])
        user_trt_inc = os.path.expanduser("~/tensorrt_headers")
        if os.path.exists(user_trt_inc):
            trt_inc_dirs.append(user_trt_inc)
        
        trt_pkg_dir = os.path.dirname(tensorrt.__file__)
        parent_dir = os.path.dirname(trt_pkg_dir)
        for c in [
            os.path.join(trt_pkg_dir, "include"),
            os.path.join(parent_dir, "tensorrt_libs", "include"),
            os.path.join(parent_dir, "tensorrt_cu12_libs", "include"),
            os.path.join(parent_dir, "tensorrt_cu13_libs", "include"),
            os.path.join(sys.prefix, "include"),
            "/usr/include",
            "/usr/local/include",
            "/usr/include/x86_64-linux-gnu",
        ]:
            if os.path.exists(c) and c not in trt_inc_dirs:
                trt_inc_dirs.append(c)

        for d in trt_inc_dirs:
            if os.path.exists(os.path.join(d, "NvInferPlugin.h")) or os.path.exists(os.path.join(d, "NvInfer.h")):
                return True
        return False
    except Exception:
        return False


@pytest.mark.skipif(not is_trt_available(), reason="TensorRT package and C++ headers (NvInferPlugin.h) required for TensorRT backend tests")
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
    if is_trt_available():
        test_tensorrt()
    else:
        print("TensorRT not available, skipping.")
