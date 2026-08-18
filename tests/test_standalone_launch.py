import torch
import triton
import triton.language as tl
from final import _fused_seq_conv_nhwc_kernel

device = 'cuda'
channels = 128
B, C, H, W = 1, channels, 64, 64

x = torch.ones(B, C, H, W, device=device).contiguous(memory_format=torch.channels_last)
w = torch.ones(4 * C, C, 3, 3, device=device).contiguous(memory_format=torch.channels_last)
out = torch.zeros(B, 4 * C, H, W, device=device).contiguous(memory_format=torch.channels_last)

grid = (B, triton.cdiv(H * W, 128), triton.cdiv(C, 32))

_fused_seq_conv_nhwc_kernel[grid](
    x, w, out,
    B, C, H, W,
    x.stride(0), x.stride(2), x.stride(3),
    w.stride(0), w.stride(2), w.stride(3),
    out.stride(0), out.stride(2), out.stride(3),
    stride_xc=1, stride_wc=1, stride_oc=1,
    BLOCK_SIZE_IC=32,
    BLOCK_SIZE_OC=32,
    BLOCK_SIZE_HW=128,
)

caches = _fused_seq_conv_nhwc_kernel.device_caches
for k, v in caches.items():
    d = v[0]
    for ck, compiled_kernel in d.items():
        print("ck key:", ck)
        print("compiled_kernel metadata:", compiled_kernel.metadata)
