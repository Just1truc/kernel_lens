import torch
import torch.nn as nn
import triton
import triton.language as tl
import kernel_lens as kl
import time

# ============================================================================
# Research Kernels Definitions
# ============================================================================

# 1. RMSNorm
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
    var = tl.sum(x * x, axis=0) / N
    rsqrt = tl.rsqrt(var + eps)
    tl.store(row_out + cols, x * rsqrt * w, mask=mask)

class RMSNormModule(nn.Module):
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

def ref_rmsnorm(x, weight, eps=1e-6):
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight

# 2. SwiGLU
@triton.jit
def swiglu_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    g = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    u = tl.load(up_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, (g * tl.sigmoid(g)) * u, mask=mask)

class SwiGLUModule(nn.Module):
    def forward(self, gate: torch.Tensor, up: torch.Tensor):
        out = torch.empty_like(gate)
        n = gate.numel()
        grid = (triton.cdiv(n, 256),)
        swiglu_kernel[grid](gate, up, out, n, BLOCK_SIZE=256)
        return out

def ref_swiglu(gate, up):
    return torch.nn.functional.silu(gate) * up

# 3. RoPE
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
    offset_cs = pid_s * D
    cols_first = tl.arange(0, HALF_D)
    cols_second = cols_first + HALF_D
    mask_half = cols_first < HALF_D
    x1 = tl.load(x_ptr + offset_x + cols_first * stride_x_d, mask=mask_half, other=0.0)
    x2 = tl.load(x_ptr + offset_x + cols_second * stride_x_d, mask=mask_half, other=0.0)
    cos1 = tl.load(cos_ptr + offset_cs + cols_first, mask=mask_half, other=1.0)
    sin1 = tl.load(sin_ptr + offset_cs + cols_first, mask=mask_half, other=0.0)
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

def ref_rope(x, cos, sin):
    B, S, H, D = x.shape
    half_d = D // 2
    x1 = x[..., :half_d]
    x2 = x[..., half_d:]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    c1 = cos[..., :half_d]
    s1 = sin[..., :half_d]
    o1 = x1 * c1 - x2 * s1
    o2 = x1 * s1 + x2 * c1
    return torch.cat([o1, o2], dim=-1)

# 4. Fused Cross Entropy
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
    max_logit = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
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

def ref_cross_entropy(logits, targets):
    return nn.functional.cross_entropy(logits, targets, reduction='none')


# ============================================================================
# Benchmark Timing Utility (High Precision CUDA Events)
# ============================================================================
def time_fn(fn, args, warmup=50, iters=200):
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        fn(*args)
    end_event.record()
    
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iters  # latency in milliseconds


def benchmark_all():
    print("=======================================================================")
    print("📊 MEASURING REAL EMPIRICAL EXECUTION LATENCIES ON GPU")
    print("=======================================================================")
    
    results = {}
    
    # --- Operator 1: RMSNorm ---
    B, M, N = 64, 512, 4096
    x = torch.randn(B, M, N, device="cuda", dtype=torch.float32)
    mod_rmsnorm = RMSNormModule(N).cuda()
    compiled_rmsnorm = kl.compile(mod_rmsnorm, (x,), name="Bench_RMSNorm", backends=["onnx"])
    
    # Warmup torch.compile
    compiled_ref_rmsnorm = torch.compile(lambda inp: ref_rmsnorm(inp, mod_rmsnorm.weight))
    for _ in range(10): compiled_ref_rmsnorm(x)
    
    lat_eager_rms = time_fn(ref_rmsnorm, (x, mod_rmsnorm.weight))
    lat_tc_rms = time_fn(compiled_ref_rmsnorm, (x,))
    lat_triton_rms = time_fn(mod_rmsnorm, (x,))
    lat_ort_rms = time_fn(lambda: compiled_rmsnorm.run((x,), backend="onnx"), ())
    
    results["RMSNorm"] = {
        "PyTorch Eager": lat_eager_rms,
        "torch.compile": lat_tc_rms,
        "Native Triton": lat_triton_rms,
        "KernelLens ORT": lat_ort_rms
    }
    
    # --- Operator 2: SwiGLU ---
    gate = torch.randn(64, 4096, device="cuda", dtype=torch.float32)
    up = torch.randn(64, 4096, device="cuda", dtype=torch.float32)
    mod_swiglu = SwiGLUModule().cuda()
    compiled_swiglu = kl.compile(mod_swiglu, (gate, up), name="Bench_SwiGLU", backends=["onnx"])
    
    compiled_ref_swiglu = torch.compile(ref_swiglu)
    for _ in range(10): compiled_ref_swiglu(gate, up)
    
    lat_eager_swi = time_fn(ref_swiglu, (gate, up))
    lat_tc_swi = time_fn(compiled_ref_swiglu, (gate, up))
    lat_triton_swi = time_fn(mod_swiglu, (gate, up))
    lat_ort_swi = time_fn(lambda: compiled_swiglu.run((gate, up), backend="onnx"), ())
    
    results["SwiGLU"] = {
        "PyTorch Eager": lat_eager_swi,
        "torch.compile": lat_tc_swi,
        "Native Triton": lat_triton_swi,
        "KernelLens ORT": lat_ort_swi
    }

    # --- Operator 3: RoPE ---
    B, S, H, D = 16, 512, 32, 128
    x_rope = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    cos = torch.randn(S, D, device="cuda", dtype=torch.float32)
    sin = torch.randn(S, D, device="cuda", dtype=torch.float32)
    mod_rope = RoPEModule().cuda()
    compiled_rope = kl.compile(mod_rope, (x_rope, cos, sin), name="Bench_RoPE", backends=["onnx"])
    
    compiled_ref_rope = torch.compile(ref_rope)
    for _ in range(10): compiled_ref_rope(x_rope, cos, sin)
    
    lat_eager_rope = time_fn(ref_rope, (x_rope, cos, sin))
    lat_tc_rope = time_fn(compiled_ref_rope, (x_rope, cos, sin))
    lat_triton_rope = time_fn(mod_rope, (x_rope, cos, sin))
    lat_ort_rope = time_fn(lambda: compiled_rope.run((x_rope, cos, sin), backend="onnx"), ())
    
    results["RoPE"] = {
        "PyTorch Eager": lat_eager_rope,
        "torch.compile": lat_tc_rope,
        "Native Triton": lat_triton_rope,
        "KernelLens ORT": lat_ort_rope
    }

    # --- Operator 4: Fused CrossEntropy ---
    B, V = 256, 32000
    logits = torch.randn(B, V, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, V, (B,), device="cuda", dtype=torch.int64)
    mod_ce = FusedCrossEntropyModule(V).cuda()
    compiled_ce = kl.compile(mod_ce, (logits, targets), name="Bench_CrossEntropy", backends=["onnx"])
    
    compiled_ref_ce = torch.compile(ref_cross_entropy)
    for _ in range(10): compiled_ref_ce(logits, targets)
    
    lat_eager_ce = time_fn(ref_cross_entropy, (logits, targets))
    lat_tc_ce = time_fn(compiled_ref_ce, (logits, targets))
    lat_triton_ce = time_fn(mod_ce, (logits, targets))
    lat_ort_ce = time_fn(lambda: compiled_ce.run((logits, targets), backend="onnx"), ())
    
    results["CrossEntropy"] = {
        "PyTorch Eager": lat_eager_ce,
        "torch.compile": lat_tc_ce,
        "Native Triton": lat_triton_ce,
        "KernelLens ORT": lat_ort_ce
    }
    
    print("\n--- MEASURED REAL GPU LATENCIES (ms) ---")
    for op, data in results.items():
        print(f"\n{op}:")
        for key, val in data.items():
            print(f"  {key:<25}: {val:.4f} ms")
            
    import json
    with open("measured_latencies.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n✅ Saved real latency statistics to measured_latencies.json")

if __name__ == "__main__":
    benchmark_all()
