import torch
import kernel_lens as kl
from final import TritonNHWCSequentialDecoder

device = torch.device('cuda')
C, H, W = 128, 64, 64
torch.manual_seed(42)

x_nchw = torch.randn(1, C, H, W, device=device)
x_nhwc = x_nchw.contiguous(memory_format=torch.channels_last)
print(f"[PYTORCH DEBUG] x_nchw sample: {x_nchw.flatten()[:5].tolist()}")
print(f"[PYTORCH DEBUG] x_nhwc sample: {x_nhwc.flatten()[:5].tolist()}")

model = TritonNHWCSequentialDecoder(C).to(device)
print(f"[PYTORCH DEBUG] model.weight raw data_ptr sample: {model.weight.data_ptr()} -> {model.weight.flatten()[:5].tolist()}")

kl_conv = kl.compile(model, (x_nhwc, model.weight), name="NHWC_Conv_SOTA", backends=["onnx", "tensorrt"])

ort_out = kl_conv.run((x_nhwc, model.weight), backend="onnx")
trt_out = kl_conv.run((x_nhwc, model.weight), backend="tensorrt")

with torch.no_grad():
    out_triton = model(x_nhwc, model.weight)

ort_tensor = torch.as_tensor(ort_out[0], device='cuda', dtype=out_triton.dtype)
trt_tensor = torch.as_tensor(trt_out[0], device='cuda', dtype=out_triton.dtype)

diff_ort = (out_triton - ort_tensor).abs().max().item()
diff_trt = (out_triton - trt_tensor).abs().max().item()

print(f"Max Diff Triton vs ORT: {diff_ort:e}")
print(f"Max Diff Triton vs TRT: {diff_trt:e}")

print(f"[DEBUG ELEMENT] Triton out[0]: {out_triton.flatten()[0].item()}")
print(f"[DEBUG ELEMENT] TRT out[0]:    {trt_tensor.flatten()[0].item()}")
print(f"[DEBUG ELEMENT] ORT out[0]:    {ort_tensor.flatten()[0].item()}")
assert diff_trt < 1e-4, f"TRT parity test failed with max diff: {diff_trt}"

