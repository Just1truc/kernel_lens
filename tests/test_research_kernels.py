import torch
import torch.nn as nn
import triton
import triton.language as tl
import kernel_lens as kl

# ============================================================================
# Research Kernel 1: Llama 3 / Mistral Fused RMSNorm
# ============================================================================
@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, out_ptr,
    stride_x_b, stride_x_m,
    stride_out_b, stride_out_m,
    N: tl.constexpr, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    row_x = x_ptr + pid_b * stride_x_b + pid_m * stride_x_m
    row_out = out_ptr + pid_b * stride_out_b + pid_m * stride_out_m
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    x = tl.load(row_x + cols, mask=mask, other=0.0)
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    
    x_sq = x * x
    var = tl.sum(x_sq, axis=0) / N
    rsqrt = tl.rsqrt(var + eps)
    
    out = x * rsqrt * w
    tl.store(row_out + cols, out, mask=mask)

class Llama3RMSNormModule(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        B, M, N = x.shape
        out = torch.empty_like(x)
        grid = (B, M)
        BLOCK_SIZE = triton.next_power_of_2(N)
        rmsnorm_kernel[grid](
            x, self.weight, out,
            x.stride(0), x.stride(1),
            out.stride(0), out.stride(1),
            N=N, eps=self.eps, BLOCK_SIZE=BLOCK_SIZE
        )
        return out

# Reference PyTorch RMSNorm
def reference_rmsnorm(x, weight, eps=1e-6):
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight

def test_llama3_rmsnorm():
    print("\n--- [Research Kernel 1] Llama 3 / Mistral Fused RMSNorm ---")
    B, M, N = 4, 32, 128
    x = torch.randn(B, M, N, device="cuda", dtype=torch.float32)
    m = Llama3RMSNormModule(N).cuda()
    
    # Native Triton execution
    triton_out = m(x)
    ref_out = reference_rmsnorm(x, m.weight)
    eager_diff = torch.abs(ref_out - triton_out).max().item()
    print(f"  Triton Eager vs Ref PyTorch Max Diff: {eager_diff:e}")
    
    # KernelLens ONNX Runtime compilation & execution
    comp = kl.compile(m, (x,), name="Llama3_RMSNorm", backends=["onnx"])
    kl_ort_out = comp.run((x,), backend="onnx")
    ort_diff = torch.abs(triton_out - kl_ort_out).max().item()
    print(f"  KernelLens ORT vs Triton Eager Max Diff: {ort_diff:e}")
    assert ort_diff < 1e-5, f"Llama 3 RMSNorm parity failed: diff={ort_diff}"
    print("  ✅ [Passed] Llama 3 RMSNorm Kernel")

# ============================================================================
# Research Kernel 2: Liger-Kernel / PaLM SwiGLU Fused Activation
# ============================================================================
@triton.jit
def swiglu_kernel(
    gate_ptr, up_ptr, out_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    g = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    u = tl.load(up_ptr + offsets, mask=mask, other=0.0)
    
    # SiLU(g) * u = (g * sigmoid(g)) * u
    sig_g = tl.sigmoid(g)
    silu_g = g * sig_g
    out = silu_g * u
    
    tl.store(out_ptr + offsets, out, mask=mask)

class SwiGLUModule(nn.Module):
    def forward(self, gate: torch.Tensor, up: torch.Tensor):
        out = torch.empty_like(gate)
        n = gate.numel()
        grid = (triton.cdiv(n, 256),)
        swiglu_kernel[grid](gate, up, out, n, BLOCK_SIZE=256)
        return out

def reference_swiglu(gate, up):
    return torch.nn.functional.silu(gate) * up

def test_swiglu_activation():
    print("\n--- [Research Kernel 2] Liger-Kernel / PaLM SwiGLU Fused MLP Activation ---")
    gate = torch.randn(16, 256, device="cuda", dtype=torch.float32)
    up = torch.randn(16, 256, device="cuda", dtype=torch.float32)
    
    m = SwiGLUModule().cuda()
    triton_out = m(gate, up)
    ref_out = reference_swiglu(gate, up)
    
    comp = kl.compile(m, (gate, up), name="SwiGLU_Activation", backends=["onnx"])
    kl_ort_out = comp.run((gate, up), backend="onnx")
    ort_diff = torch.abs(triton_out - kl_ort_out).max().item()
    print(f"  KernelLens ORT vs Triton Eager Max Diff: {ort_diff:e}")
    assert ort_diff < 1e-5, f"SwiGLU parity failed: diff={ort_diff}"
    print("  ✅ [Passed] SwiGLU Fused Activation Kernel")

# ============================================================================
# Research Kernel 3: Llama 3 / Qwen 2.5 Rotary Position Embedding (RoPE)
# ============================================================================
@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_x_b, stride_x_s, stride_x_h, stride_x_d,
    stride_out_b, stride_out_s, stride_out_h, stride_out_d,
    D: tl.constexpr, HALF_D: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    offset_x = pid_b * stride_x_b + pid_s * stride_x_s + pid_h * stride_x_h
    offset_out = pid_b * stride_out_b + pid_s * stride_out_s + pid_h * stride_out_h
    offset_cs = pid_s * D  # [S, D]
    
    cols_first = tl.arange(0, HALF_D)
    cols_second = cols_first + HALF_D
    
    mask_half = cols_first < HALF_D
    
    # Load first and second half of head_dim
    x1 = tl.load(x_ptr + offset_x + cols_first * stride_x_d, mask=mask_half, other=0.0)
    x2 = tl.load(x_ptr + offset_x + cols_second * stride_x_d, mask=mask_half, other=0.0)
    
    cos1 = tl.load(cos_ptr + offset_cs + cols_first, mask=mask_half, other=1.0)
    sin1 = tl.load(sin_ptr + offset_cs + cols_first, mask=mask_half, other=0.0)
    
    # RoPE formula:
    # out1 = x1 * cos - x2 * sin
    # out2 = x1 * sin + x2 * cos
    out1 = x1 * cos1 - x2 * sin1
    out2 = x1 * sin1 + x2 * cos1
    
    tl.store(out_ptr + offset_out + cols_first * stride_out_d, out1, mask=mask_half)
    tl.store(out_ptr + offset_out + cols_second * stride_out_d, out2, mask=mask_half)

class RoPEModule(nn.Module):
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        B, S, H, D = x.shape
        half_d = D // 2
        out = torch.empty_like(x)
        grid = (B, S, H)
        rope_kernel[grid](
            x, cos, sin, out,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            D=D, HALF_D=half_d, BLOCK_D=triton.next_power_of_2(half_d)
        )
        return out

def reference_rope(x, cos, sin):
    B, S, H, D = x.shape
    half_d = D // 2
    x1 = x[..., :half_d]
    x2 = x[..., half_d:]
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, S, 1, D/2]
    sin = sin.unsqueeze(0).unsqueeze(2)
    c1 = cos[..., :half_d]
    s1 = sin[..., :half_d]
    o1 = x1 * c1 - x2 * s1
    o2 = x1 * s1 + x2 * c1
    return torch.cat([o1, o2], dim=-1)

def test_rope_embedding():
    print("\n--- [Research Kernel 3] Llama 3 / Qwen 2.5 Rotary Position Embedding (RoPE) ---")
    B, S, H, D = 2, 16, 8, 64
    x = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    cos = torch.randn(S, D, device="cuda", dtype=torch.float32)
    sin = torch.randn(S, D, device="cuda", dtype=torch.float32)
    
    m = RoPEModule().cuda()
    triton_out = m(x, cos, sin)
    ref_out = reference_rope(x, cos, sin)
    eager_diff = torch.abs(ref_out - triton_out).max().item()
    print(f"  Triton Eager vs Ref PyTorch Max Diff: {eager_diff:e}")
    
    comp = kl.compile(m, (x, cos, sin), name="RoPE_Embedding", backends=["onnx"])
    kl_ort_out = comp.run((x, cos, sin), backend="onnx")
    ort_diff = torch.abs(triton_out - kl_ort_out).max().item()
    print(f"  KernelLens ORT vs Triton Eager Max Diff: {ort_diff:e}")
    assert ort_diff < 1e-5, f"RoPE parity failed: diff={ort_diff}"
    print("  ✅ [Passed] RoPE Positional Embedding Kernel")

# ============================================================================
# Research Kernel 4: Liger-Kernel Fused Cross-Entropy Loss
# ============================================================================
@triton.jit
def fused_cross_entropy_kernel(
    logits_ptr, targets_ptr, loss_ptr,
    stride_logits_b, stride_logits_c,
    V: tl.constexpr, BLOCK_V: tl.constexpr
):
    pid = tl.program_id(0)
    
    row_logits = logits_ptr + pid * stride_logits_b
    target_idx = tl.load(targets_ptr + pid)
    
    cols = tl.arange(0, BLOCK_V)
    mask = cols < V
    
    logits = tl.load(row_logits + cols * stride_logits_c, mask=mask, other=-1e9)
    
    # 1. Max for numerical stability
    max_logit = tl.max(logits, axis=0)
    
    # 2. Exponent and Sum
    exp_logits = tl.exp(logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    
    # 3. Log-Softmax at target_idx
    target_logit = tl.load(row_logits + target_idx * stride_logits_c)
    loss = tl.log(sum_exp) + max_logit - target_logit
    
    tl.store(loss_ptr + pid, loss)

class FusedCrossEntropyModule(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        B = logits.shape[0]
        loss = torch.empty(B, device=logits.device, dtype=logits.dtype)
        grid = (B,)
        BLOCK_V = triton.next_power_of_2(self.vocab_size)
        fused_cross_entropy_kernel[grid](
            logits, targets, loss,
            logits.stride(0), logits.stride(1),
            V=self.vocab_size, BLOCK_V=BLOCK_V
        )
        return loss

def test_fused_cross_entropy():
    print("\n--- [Research Kernel 4] Liger-Kernel Fused Softmax & Cross Entropy Loss ---")
    B, V = 16, 512
    logits = torch.randn(B, V, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, V, (B,), device="cuda", dtype=torch.int64)
    
    m = FusedCrossEntropyModule(V).cuda()
    triton_out = m(logits, targets)
    
    ref_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    eager_diff = torch.abs(ref_loss - triton_out).max().item()
    print(f"  Triton Eager vs Ref PyTorch Max Diff: {eager_diff:e}")
    
    comp = kl.compile(m, (logits, targets), name="FusedCrossEntropy", backends=["onnx"])
    kl_ort_out = comp.run((logits, targets), backend="onnx")
    ort_diff = torch.abs(triton_out - kl_ort_out).max().item()
    print(f"  KernelLens ORT vs Triton Eager Max Diff: {ort_diff:e}")
    assert ort_diff < 1e-5, f"Fused Cross Entropy parity failed: diff={ort_diff}"
    print("  ✅ [Passed] Fused Softmax & Cross Entropy Loss Kernel")

if __name__ == "__main__":
    print("==========================================================================")
    print("🔥 TESTING TRITON KERNELS FROM RECENT RESEARCH MODELS WITH KERNEL LENS 🔥")
    print("==========================================================================")
    test_llama3_rmsnorm()
    test_swiglu_activation()
    test_rope_embedding()
    test_fused_cross_entropy()
    print("\n🎉 ALL RECENT RESEARCH TRITON KERNELS PASSED WITH KERNEL LENS! 🎉")
