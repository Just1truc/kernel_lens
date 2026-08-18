import torch
import kernel_lens as kl
import triton.testing
import numpy as np

import torch
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def _fused_seq_conv_nhwc_kernel(
    x_ptr, w_ptr, out_ptr,
    batch, channels, H, W,
    stride_xn, stride_xh, stride_xw,
    stride_wn, stride_wh, stride_ww,
    stride_on, stride_oh, stride_ow,
    stride_xc: tl.constexpr = 1,
    stride_wc: tl.constexpr = 1,
    stride_oc: tl.constexpr = 1,
    BLOCK_SIZE_IC: tl.constexpr = 32,
    BLOCK_SIZE_OC: tl.constexpr = 32,
    BLOCK_SIZE_HW: tl.constexpr = 128
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_oc = tl.program_id(2)

    # 1. Spatial indices
    offs_hw = pid_hw * BLOCK_SIZE_HW + tl.arange(0, BLOCK_SIZE_HW)
    offs_h = offs_hw // W
    offs_w = offs_hw % W
    
    # 2. Output channel indices
    offs_oc = pid_oc * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC)
    safe_oc0 = tl.minimum(tl.maximum(offs_oc, 0), channels - 1)
    oc1 = channels + offs_oc
    safe_oc1 = tl.minimum(tl.maximum(oc1, 0), 4 * channels - 1)
    oc2 = 2 * channels + offs_oc
    safe_oc2 = tl.minimum(tl.maximum(oc2, 0), 4 * channels - 1)
    oc3 = 3 * channels + offs_oc
    safe_oc3 = tl.minimum(tl.maximum(oc3, 0), 4 * channels - 1)
    
    # 3. Accumulators: Shape [BLOCK_SIZE_HW, BLOCK_SIZE_OC]
    # Notice the flipped shape compared to the NCHW version
    acc0 = tl.zeros((BLOCK_SIZE_HW, BLOCK_SIZE_OC), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_SIZE_HW, BLOCK_SIZE_OC), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_HW, BLOCK_SIZE_OC), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_SIZE_HW, BLOCK_SIZE_OC), dtype=tl.float32)

    # 4. Loop over Input Channels (which are now contiguous in memory!)
    for ic in range(0, channels, BLOCK_SIZE_IC):
        offs_ic = ic + tl.arange(0, BLOCK_SIZE_IC)
        
        # 5. Loop over the 3x3 kernel spatial window
        for ky in range(3):
            for kx in range(3):
                in_h = offs_h + ky - 1
                in_w = offs_w + kx - 1
                
                spatial_mask = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)
                
                safe_in_h = tl.minimum(tl.maximum(in_h, 0), H - 1)
                safe_in_w = tl.minimum(tl.maximum(in_w, 0), W - 1)

                # Load X tile: Shape [BLOCK_SIZE_HW, BLOCK_SIZE_IC]
                x_off = (pid_b * stride_xn) + \
                        (safe_in_h[:, None] * stride_xh) + \
                        (safe_in_w[:, None] * stride_xw) + \
                        (offs_ic[None, :] * stride_xc)
                safe_x_off = tl.minimum(x_off, (batch * channels * H * W) - 4)
                x_ptrs = x_ptr + safe_x_off
                
                x_mask = spatial_mask[:, None] & (offs_ic[None, :] < channels)
                x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
                
                # Load Weight tiles: Shape [BLOCK_SIZE_OC, BLOCK_SIZE_IC]
                w_base_ptrs = w_ptr + (ky * stride_wh) + (kx * stride_ww) + \
                              (offs_ic[None, :] * stride_wc)
                
                ic_mask = offs_ic[None, :] < channels

                # Group 0
                w0_mask = (offs_oc[:, None] < channels) & ic_mask
                w0_ptrs = w_base_ptrs + (safe_oc0[:, None] * stride_wn)
                w0_tile = tl.load(w0_ptrs, mask=w0_mask, other=0.0)
                acc0 += tl.dot(x_tile, tl.trans(w0_tile))
                
                # Group 1
                w1_mask = (oc1[:, None] < 4 * channels) & ic_mask
                w1_ptrs = w_base_ptrs + (safe_oc1[:, None] * stride_wn)
                w1_tile = tl.load(w1_ptrs, mask=w1_mask, other=0.0)
                acc1 += tl.dot(x_tile, tl.trans(w1_tile))
                
                # Group 2
                w2_mask = (oc2[:, None] < 4 * channels) & ic_mask
                w2_ptrs = w_base_ptrs + (safe_oc2[:, None] * stride_wn)
                w2_tile = tl.load(w2_ptrs, mask=w2_mask, other=0.0)
                acc2 += tl.dot(x_tile, tl.trans(w2_tile))
                
                # Group 3
                w3_mask = (oc3[:, None] < 4 * channels) & ic_mask
                w3_ptrs = w_base_ptrs + (safe_oc3[:, None] * stride_wn)
                w3_tile = tl.load(w3_ptrs, mask=w3_mask, other=0.0)
                acc3 += tl.dot(x_tile, tl.trans(w3_tile))

    # 6. Store Output tiles in NHWC format
    out_spatial_mask = offs_hw[:, None] < (H * W)
    out_base_off = (pid_b * stride_on) + \
                   (offs_h[:, None] * stride_oh) + \
                   (offs_w[:, None] * stride_ow)
                   
    out_mask0 = out_spatial_mask & (offs_oc[None, :] < channels)
    out_mask1 = out_spatial_mask & ((channels + offs_oc)[None, :] < 4 * channels)
    out_mask2 = out_spatial_mask & ((2 * channels + offs_oc)[None, :] < 4 * channels)
    out_mask3 = out_spatial_mask & ((3 * channels + offs_oc)[None, :] < 4 * channels)

    max_out_off = (batch * 4 * channels * H * W) - 1
    
    off0 = tl.minimum(tl.maximum(out_base_off + (safe_oc0[None, :] * stride_oc), 0), max_out_off)
    off1 = tl.minimum(tl.maximum(out_base_off + (safe_oc1[None, :] * stride_oc), 0), max_out_off)
    off2 = tl.minimum(tl.maximum(out_base_off + (safe_oc2[None, :] * stride_oc), 0), max_out_off)
    off3 = tl.minimum(tl.maximum(out_base_off + (safe_oc3[None, :] * stride_oc), 0), max_out_off)

    tl.store(out_ptr + off0, acc0, mask=out_mask0)
    tl.store(out_ptr + off1, acc1, mask=out_mask1)
    tl.store(out_ptr + off2, acc2, mask=out_mask2)
    tl.store(out_ptr + off3, acc3, mask=out_mask3)


class TritonNHWCSequentialDecoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # Initialize weights and FORCE channels_last memory format
        weight = torch.randn(4 * channels, channels, 3, 3) * (2.0 / (9 * channels))**0.5
        self.weight = nn.Parameter(weight.to(memory_format=torch.channels_last))

    def forward(self, x, weight=None):
        if weight is None:
            weight = self.weight

        # Ensure the input tensor is physically arranged as NHWC
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
            
        B, C, H, W = x.shape
        
        # Pre-allocate output directly in channels_last format
        # Note: We allocate [B, 4*C, H, W] to natively support the NHWC striding
        out = torch.empty((B, 4 * C, H, W), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
        
        BLOCK_SIZE_IC = min(32, C)
        BLOCK_SIZE_OC = min(32, C)
        BLOCK_SIZE_HW = 128
        
        grid = lambda meta: (
            B,
            triton.cdiv(H * W, meta['BLOCK_SIZE_HW']),
            triton.cdiv(C, meta['BLOCK_SIZE_OC'])
        )
        
        _fused_seq_conv_nhwc_kernel[grid](
            x, weight, out,
            B, C, H, W,
            x.stride(0), x.stride(2), x.stride(3),
            weight.stride(0), weight.stride(2), weight.stride(3),
            out.stride(0), out.stride(2), out.stride(3),
            stride_xc=1, stride_wc=1, stride_oc=1,
            BLOCK_SIZE_IC=BLOCK_SIZE_IC,
            BLOCK_SIZE_OC=BLOCK_SIZE_OC,
            BLOCK_SIZE_HW=BLOCK_SIZE_HW,
        )
        
        # Reshape back to your expected [B, 4, C, H, W] view
        return out.view(B, 4, C, H, W)

class SequentialDecoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 4 separate decoders
        self.decoders = nn.ModuleList([nn.Conv2d(channels, channels, 3, padding=1) for _ in range(4)])
    def forward(self, x):
        outs = []
        for dec in self.decoders:
            outs.append(dec(x))
        return torch.stack(outs, dim=1)

class GroupedDecoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 1 massive grouped convolution
        self.conv = nn.Conv2d(channels * 4, channels * 4, 3, padding=1, groups=4)
    def forward(self, x):
        B, C, T, F = x.shape
        # x_ptr = x.data_ptr()
        x_grouped = x.broadcast_to(B, 4, C, T, F).reshape(B, 4*C, T, F)
        # assert x_ptr == x_grouped.data_ptr()
        # no copy happens
        out = self.conv(x_grouped)
        return out.view(B, 4, C, T, F)

def benchmark_configurations():
    configs = [
        (1, 120, 10, 64),
        (1, 96, 20, 128),
        (1, 72, 40, 256),
        (1, 48, 80, 512),
        (1, 64, 128, 1024)
    ]

    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    device = torch.device('cuda')
    
    # Print updated table header
    print(f"\n{'='*105}")
    print(f"{'Shape (B, C, H, W)':<22} | {'Max Diff':<12} | {'PyTorch Seq (ms)':<18} | {'PyTorch Grp (ms)':<18} | {'Triton Fused (ms)':<18}")
    print(f"{'-'*105}")

    for B, C, H, W in configs:
        # 1. Initialize input with channels_last format
        x = torch.randn(B, C, H, W, device=device).contiguous(memory_format=torch.channels_last)
        
        # 2. Initialize models (forcing channels_last where applicable)
        torch_seq = SequentialDecoder(C).to(device).to(memory_format=torch.channels_last)
        torch_grp = GroupedDecoder(C).to(device).to(memory_format=torch.channels_last)
        triton_seq = TritonNHWCSequentialDecoder(C).to(device)
        
        # 3. Synchronize Weights for Correctness Verification
        with torch.no_grad():
            # Stack the 4 PyTorch convolutions into one weight tensor [4*C, C, 3, 3]
            stacked_weights = torch.cat([conv.weight for conv in torch_seq.decoders], dim=0)
            
            # Copy to Triton model, maintaining its memory format
            triton_seq.weight.copy_(stacked_weights)
            
            # Zero out PyTorch biases because our Triton kernel does not implement bias addition
            for conv in torch_seq.decoders:
                if conv.bias is not None:
                    conv.bias.zero_()

        # 4. Correctness Check (Diff)
        torch_seq.eval()
        triton_seq.eval()
        
        with torch.no_grad():
            out_torch = torch_seq(x)
            out_triton = triton_seq(x)
            
        # Compute maximum absolute error between outputs
        max_diff = torch.max(torch.abs(out_torch - out_triton)).item()
        diff_str = f"{max_diff:.2e}"

        # 5. Benchmarking
        torch_grp.eval()
        def run_torch_seq(): return torch_seq(x)
        def run_torch_grp(): return torch_grp(x)
        def run_triton_seq(): return triton_seq(x)
        
        try:
            ms_torch_seq = triton.testing.do_bench(run_torch_seq)
            ms_torch_grp = triton.testing.do_bench(run_torch_grp)
            ms_triton_seq = triton.testing.do_bench(run_triton_seq)
        except Exception as e:
            print(f"Error benchmarking shape {(B, C, H, W)}: {e}")
            continue
            
        # 6. Print formatted row
        shape_str = f"({B}, {C}, {H}, {W})"
        print(f"{shape_str:<22} | {diff_str:<12} | {ms_torch_seq:<18.4f} | {ms_torch_grp:<18.4f} | {ms_triton_seq:<18.4f}")

    print(f"{'='*105}\n")

def run_ultimate_benchmark_suite():
    device = torch.device('cuda')

    # ======================================================================
    # TEST 2: Fused NHWC Sequential Conv (The "VRAM Reuse" King)
    # ======================================================================
    print(f"\n{'='*80}\n🧪 BATTLE: NHWC Sequential Conv (Register Reuse)\n{'='*80}")
    C, H, W = 128, 64, 64
    x_nhwc = torch.randn(1, C, H, W, device=device).contiguous(memory_format=torch.channels_last)
    model = TritonNHWCSequentialDecoder(C).to(device)
    pt_out = model(x_nhwc)
    pt_out_list = list(pt_out) if isinstance(pt_out, (tuple, list)) else [pt_out]

    # 1. Compile ONNX backend
    kl_conv = kl.compile(model, (x_nhwc,), name="NHWC_Conv_SOTA", backends=["onnx"])

    print("Running single ORT execution...")
    ort_outputs = kl_conv.run((x_nhwc,), backend="onnx")
    print("Single ORT execution successful!")

    triton_native_out = model(x_nhwc)
    def check_stability(backend_name, outputs):
        is_stable = True
        max_err = 0.0
        for p, t in zip([triton_native_out], outputs):
            t_tensor = torch.as_tensor(t, device='cuda', dtype=p.dtype).reshape_as(p)
            diff = torch.abs(p - t_tensor)
            curr_max = diff.max().item()
            max_err = max(max_err, curr_max)
            print(f"[DEBUG {backend_name}] p sample [0, 0, 0, 5:8, 5:8]:\n  {p[0, 0, 0, 5:8, 5:8].tolist()}")
            print(f"[DEBUG {backend_name}] t sample [0, 0, 0, 5:8, 5:8]:\n  {t_tensor[0, 0, 0, 5:8, 5:8].tolist()}")
            print(f"[DEBUG {backend_name}] p nonzeros: {(p != 0).sum().item()} / {p.numel()}")
            print(f"[DEBUG {backend_name}] t nonzeros: {(t_tensor != 0).sum().item()} / {t_tensor.numel()}")
            if not torch.allclose(p, t_tensor, atol=1e-3, rtol=1e-3):
                is_stable = False
        
        status = "✅ PASSED" if is_stable else "❌ FAILED"
        print(f"  -> Stability ({backend_name}): {status} (Max Err: {max_err:.6e})")
        return is_stable

    ort_stable = check_stability("ORT", ort_outputs)

if __name__ == "__main__":
    run_ultimate_benchmark_suite()