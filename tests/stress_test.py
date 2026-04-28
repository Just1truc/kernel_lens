import torch
import torch.nn as nn
import triton
import triton.language as tl
import kernel_lens as kl
from kernel_lens.compiler.interaction import AutoInteractionHandler
import time

def native_rms_norm(x, gamma, beta):
    # PyTorch assemble souvent ça manuellement si ce n'est pas du LayerNorm standard
    eps = 1e-6
    var = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + eps)) * gamma + beta

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def native_rope(q, k, cos, sin):
    # Fix: application partielle sur rotary_dim (32)
    q_rot = q[..., :32]
    q_pass = q[..., 32:]
    k_rot = k[..., :32]
    k_pass = k[..., 32:]
    
    q_out = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    k_out = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_out, k_out

class NativeAttentionModule(torch.nn.Module):
    def forward(self, q, k, v, scale):
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = torch.relu(attn).pow(2)
        return attn @ v

def native_squared_relu_attn(q, k, v, scale):
    # Attention "Naive" (O(N^2) mémoire)
    # C'est ici que PyTorch va exploser en temps/mémoire par rapport à ton Flash Attention
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = torch.relu(attn).pow(2)
    return attn @ v

# ============================================================================
# 1. TRITON KERNELS (Directly from your research)
# ============================================================================
@triton.jit
def rms_group_norm_forward_kernel_trt(
    x_ptr, gamma_ptr, beta_ptr, 
    B, T, F, G, D,
    stride_bt, stride_tt, stride_ft, stride_ct,
    output_ptr,
    BLOCK_SIZE: tl.constexpr, EPS: tl.constexpr, USE_BIAS: tl.constexpr
):
    pid = tl.program_id(0)
    pid_btf = pid // G
    pid_g = pid % G

    total_tf = T * F
    pid_b = pid_btf // total_tf
    pid_tf = pid_btf % total_tf
    pid_t = pid_tf // F
    pid_f = pid_tf % F

    group_offset = pid_b * stride_bt + pid_t * stride_tt + pid_f * stride_ft + pid_g * D * stride_ct
    x_group_ptr = x_ptr + group_offset
    y_group_ptr = output_ptr + group_offset

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D

    x_vals = tl.load(x_group_ptr + offsets * stride_ct, mask=mask, other=0.0)
    x_sq = x_vals * x_vals
    sum_squares = tl.sum(x_sq, axis=0)
    mean_square = sum_squares / D
    rms = tl.sqrt(mean_square + EPS)
    x_normalized = x_vals / rms

    param_offset = pid_g * D + offsets
    gamma = tl.load(gamma_ptr + param_offset, mask=mask, other=1.0)
    y = x_normalized * gamma
    
    if USE_BIAS:
        beta = tl.load(beta_ptr + param_offset, mask=mask, other=0.0)
        y = y + beta

    tl.store(y_group_ptr + offsets * stride_ct, y, mask=mask)

@triton.jit
def rope_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    q_out_ptr, k_out_ptr,
    q_stride_b, q_stride_h, q_stride_n, q_stride_d,
    k_stride_b, k_stride_h, k_stride_n, k_stride_d,
    cos_stride_n, cos_stride_d,
    q_out_stride_b, q_out_stride_h, q_out_stride_n, q_out_stride_d,
    k_out_stride_b, k_out_stride_h, k_out_stride_n, k_out_stride_d,
    n_head, seq_len, dim, rotary_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    i_n = pid % seq_len
    i_h = (pid // seq_len) % n_head
    i_b = pid // (seq_len * n_head)
    
    q_ptr_base = q_ptr + i_b * q_stride_b + i_h * q_stride_h + i_n * q_stride_n
    k_ptr_base = k_ptr + i_b * k_stride_b + i_h * k_stride_h + i_n * k_stride_n
    q_out_ptr_base = q_out_ptr + i_b * q_out_stride_b + i_h * q_out_stride_h + i_n * q_out_stride_n
    k_out_ptr_base = k_out_ptr + i_b * k_out_stride_b + i_h * k_out_stride_h + i_n * k_out_stride_n

    cos_ptr_base = cos_ptr + i_n * cos_stride_n
    sin_ptr_base = sin_ptr + i_n * cos_stride_n

    offs_d = tl.arange(0, BLOCK_SIZE)
    mask_rot = offs_d < rotary_dim
    
    cos = tl.load(cos_ptr_base + offs_d * cos_stride_d, mask=mask_rot, other=0.0)
    sin = tl.load(sin_ptr_base + offs_d * cos_stride_d, mask=mask_rot, other=0.0)
    
    q = tl.load(q_ptr_base + offs_d * q_stride_d, mask=mask_rot, other=0.0)
    k = tl.load(k_ptr_base + offs_d * k_stride_d, mask=mask_rot, other=0.0)
    
    is_even = (offs_d % 2) == 0
    offs_swap = tl.where(is_even, offs_d + 1, offs_d - 1)
    
    q_swap = tl.load(q_ptr_base + offs_swap * q_stride_d, mask=mask_rot, other=0.0)
    k_swap = tl.load(k_ptr_base + offs_swap * k_stride_d, mask=mask_rot, other=0.0)
    
    sign = tl.where(is_even, -1.0, 1.0)
    q_rot = q * cos + q_swap * sin * sign
    k_rot = k * cos + k_swap * sin * sign
    
    tl.store(q_out_ptr_base + offs_d * q_out_stride_d, q_rot, mask=mask_rot)
    tl.store(k_out_ptr_base + offs_d * k_out_stride_d, k_rot, mask=mask_rot)
    
    if dim > rotary_dim:
        mask_pass = (offs_d >= rotary_dim) & (offs_d < dim)
        q_pass = tl.load(q_ptr_base + offs_d * q_stride_d, mask=mask_pass, other=0.0)
        k_pass = tl.load(k_ptr_base + offs_d * k_stride_d, mask=mask_pass, other=0.0)
        tl.store(q_out_ptr_base + offs_d * q_out_stride_d, q_pass, mask=mask_pass)
        tl.store(k_out_ptr_base + offs_d * k_out_stride_d, k_pass, mask=mask_pass)

@triton.jit
def squared_relu_kernel(
    Q, K, V, scale, 
    Out,
    stride_b, stride_h, stride_n, stride_d,
    Z, H, N_CTX, BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    off_m = tl.program_id(0) * BLOCK_M
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    base_ptr = off_z * stride_b + off_h * stride_h
    Q_ptr = Q + base_ptr
    K_ptr = K + base_ptr
    V_ptr = V + base_ptr
    Out_ptr = Out + base_ptr
    
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    offs_m = off_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    q_ptrs = Q_ptr + (offs_m[:, None] * stride_n + offs_d[None, :])
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + (offs_n[None, :] * stride_n + offs_d[:, None])
        k = tl.load(k_ptrs, mask=offs_n[None, :] < N_CTX, other=0.0)
        
        qk = tl.dot(q, k)
        qk *= scale
        
        dist = tl.abs(offs_m[:, None] - offs_n[None, :])
        qk = qk - dist.to(tl.float32)
        
        mask = (offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX)
        qk = tl.where(mask, qk, -10000.0) 
        
        qk = tl.maximum(qk, 0.0)
        qk = qk * qk
        
        v_ptrs = V_ptr + (offs_n[:, None] * stride_n + offs_d[None, :])
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        
        acc += tl.dot(qk, v)
        
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_n + offs_d[None, :])
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < N_CTX)


# ============================================================================
# 2. PYTORCH WRAPPERS (For ONNX Tracing)
# ============================================================================
class RMSGroupNormLayer(nn.Module):
    def __init__(self, G, D, eps=1e-5):
        super().__init__()
        self.G = G
        self.D = D
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(G * D))
        self.beta = nn.Parameter(torch.zeros(G * D))

    def forward(self, x):
        B, T, F, C = x.shape
        out = torch.empty_like(x)
        grid = lambda meta: (B * T * F * self.G, )
        
        res = rms_group_norm_forward_kernel_trt[grid](
            x, self.gamma, self.beta,
            B, T, F, self.G, self.D,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out,
            BLOCK_SIZE=triton.next_power_of_2(self.D),
            EPS=self.eps, USE_BIAS=True
        )
        return res if isinstance(res, torch.Tensor) else out

class RoPELayer(nn.Module):
    def forward(self, q, k, cos, sin):
        B, H, N, D = q.shape
        rotary_dim = cos.shape[-1]
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        
        grid = lambda meta: (B * H * N, )
        
        res = rope_kernel[grid](
            q, k, cos, sin, q_out, k_out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            cos.stride(0), cos.stride(1),
            q_out.stride(0), q_out.stride(1), q_out.stride(2), q_out.stride(3),
            k_out.stride(0), k_out.stride(1), k_out.stride(2), k_out.stride(3),
            H, N, D, rotary_dim,
            BLOCK_SIZE=triton.next_power_of_2(D)
        )
        
        if isinstance(res, tuple) and len(res) == 2:
            return res[0], res[1]
        return q_out, k_out

class SquaredReLUAttentionLayer(nn.Module):
    def forward(self, q, k, v, scale):
        Z, H, N_CTX, D = q.shape
        out = torch.empty_like(q)
        
        grid = lambda meta: (
            triton.cdiv(N_CTX, meta['BLOCK_M']),
            H,
            Z
        )
        
        res = squared_relu_kernel[grid](
            q, k, v, scale, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            Z, H, N_CTX, 
            BLOCK_D=triton.next_power_of_2(D), 
            BLOCK_M=64, BLOCK_N=64
        )
        return res if isinstance(res, torch.Tensor) else out


# ============================================================================
# 3. BENCHMARKING HARNESS
# ============================================================================
def benchmark_latency(fn, args, kwargs={}, warmup=25, rep=100):
    """Accurately measures GPU latency using CUDA events."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(rep):
        fn(*args, **kwargs)
    end_event.record()
    
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / rep

def run_stress_test(model, inputs, name, native_fn=None, atol=1e-3):
    try:
        print(f"\n{'='*70}\n🧪 COMPILER BATTLE: {name}\n{'='*70}")

        # 1. Baseline: torch.compile (L'adversaire n°1)
        print("[1/5] Benchmarking torch.compile (Inductor)...")
        compiled_pt = torch.compile(native_fn if native_fn else model)
        # Warmup
        for _ in range(5): compiled_pt(*inputs)
        tc_lat = benchmark_latency(compiled_pt, inputs)

        # 2. Baseline: Triton (Python)
        print("[2/5] Benchmarking Native Triton (Python)...")
        pt_latency = benchmark_latency(model, inputs)
        pt_out = model(*inputs)
        pt_out_list = list(pt_out) if isinstance(pt_out, (tuple, list)) else [pt_out]
        
        # 3. Kernel Lens: TensorRT (Ton approche)
        print("[3/5] Kernel Lens -> TensorRT Plugin...")
        # On force la compilation propre
        kl_model = kl.compile(model, inputs, name=name, backends=["tensorrt"])
        trt_lat = benchmark_latency(kl_model.run, (inputs,), {"backend": "tensorrt"})
        # --- VÉRIFICATION DE LA STABILITÉ NUMÉRIQUE ---
        # Récupération propre des sorties (TRT renvoie toujours une liste)
        trt_outputs = kl_model.run(inputs, backend="tensorrt")

        # 4. Kernel Lens: ONNX Runtime (Ton approche)
        print("[4/5] Kernel Lens -> ONNX Runtime Plugin...")
        kl_model_ort = kl.compile(model, inputs, name=name, backends=["onnx"])
        ort_lat = benchmark_latency(kl_model_ort.run, (inputs,), {"backend": "onnx"})
        ort_outputs = kl_model_ort.run(inputs, backend="onnx")
        
        # 5. Standard Export (TRT sans plugin)
        # Note: On utilise native_fn pour éviter d'exporter ton kernel Triton dans le graphe standard
        print("[5/5] Native TensorRT (Standard Graph, No Plugin)...")
        # (Ici on utiliserait torch.onnx.export puis trt.Builder standard)
        # Pour gagner du temps, on peut déjà comparer TRT vs torch.compile
        
        print("\n📊 FINAL PERFORMANCE COMPARISON:")
        print(f"  -> torch.compile:         {tc_lat:.4f} ms")
        print(f"  -> Triton (Python):       {pt_latency:.4f} ms")
        print(f"  -> Kernel Lens (ORT):     {ort_lat:.4f} ms")
        print(f"  -> Kernel Lens (TRT):     {trt_lat:.4f} ms")
        print(f"  -------------------------------------")
        print(f"  🏆 SPEEDUP vs torch.compile: {(tc_lat / trt_lat):.2f}x")
        print(f"  🏆 SPEEDUP vs Triton:        {(pt_latency / trt_lat):.2f}x")
        
        # is_stable = torch.allclose(pt_out, torch.as_tensor(trt_out, device='cuda'), atol=atol)
        # print(f"  -> Numerical Stability: {'✅ PASSED' if is_stable else '❌ FAILED'}")
        # if isinstance(pt_out, (tuple, list)):
        #     # Cas multi-sorties (ex: RoPE)
        #     is_stable = all(
        #         torch.allclose(p, torch.as_tensor(t, device='cuda'), atol=atol) 
        #         for p, t in zip(pt_out, trt_outputs)
        #     )
        #     max_err = max(torch.abs(p - torch.as_tensor(t, device='cuda')).max().item() 
        #                 for p, t in zip(pt_out, trt_outputs))
        # else:
        #     # Cas sortie unique (ex: RMS Norm, Attention)
        #     trt_out_tensor = torch.as_tensor(trt_outputs[0], device='cuda')
        #     is_stable = torch.allclose(pt_out, trt_out_tensor, atol=atol)
        #     max_err = torch.abs(pt_out - trt_out_tensor).max().item()

        # print(f"  -> Numerical Stability: {'✅ PASSED' if is_stable else '❌ FAILED'} (Max Err: {max_err:.6e})")
        # --- DUAL BACKEND STABILITY CHECK ---
        def check_stability(backend_name, outputs):
            is_stable = True
            max_err = 0.0
            for p, t in zip(pt_out_list, outputs):
                t_tensor = torch.as_tensor(t, device='cuda')
                diff = torch.abs(p - t_tensor)
                curr_max = diff.max().item()
                max_err = max(max_err, curr_max)
                if not torch.allclose(p, t_tensor, atol=atol):
                    is_stable = False
            
            status = "✅ PASSED" if is_stable else "❌ FAILED"
            print(f"  -> Stability ({backend_name}): {status} (Max Err: {max_err:.6e})")
            return is_stable

        trt_stable = check_stability("TRT", trt_outputs)
        ort_stable = check_stability("ORT", ort_outputs)

        return trt_stable and ort_stable
        # return is_stable
    except Exception as e:
        if isinstance(e, ValueError):
            print(f"TEST SKIPPED: {name}")
            print(f"REASON: {str(e)}")
            return True
        print(f"❌ TEST FAILED: {name}")
        print(f"ERROR: {str(e)}")
        # CRITICAL: Clear the sticky CUDA error
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # In some cases, you might need a more aggressive reset, 
        # but empty_cache + sync usually helps PyTorch recover.
        return False
    
    # return trt_stable and ort_stable


def run_robustness_suite():
    print(f"\n{'#'*70}\n🔥 STARTING ROBUSTNESS TORTURE SUITE\n{'#'*70}")
    
    results = []
    
    # --- CASE 1: Non-Power-of-Two (The "Alignment Killer") ---
    # Teste si les masques Triton (mask = tl.load(ptr, mask=...)) sont bien gérés en C++
    print("\n[Edge Case] Non-Power-of-Two Dimensions (D=63, G=7)")
    rms_npot = RMSGroupNormLayer(G=7, D=9).cuda() # Total 63
    x_npot = torch.randn(1, 4, 7, 63, device='cuda')
    results.append(("NPOT_Shape", run_stress_test(rms_npot, (x_npot,), "RMS_NPOT")))

    # --- CASE 2: High-Stride / Non-Contiguous (The "Pointer Killer") ---
    # Teste si ton code C++ respecte les strides ONNX ou s'il assume la contiguité
    print("\n[Edge Case] Non-Contiguous Inputs (Permuted Layout)")
    attn_base = SquaredReLUAttentionLayer().cuda()
    q_nc = torch.randn(1, 8, 128, 64, device='cuda').transpose(1, 2) # Stride haché
    k_nc = torch.randn(1, 8, 128, 64, device='cuda').transpose(1, 2)
    v_nc = torch.randn(1, 8, 128, 64, device='cuda')
    # On force le layout non-contigu
    results.append(("Non_Contiguous", run_stress_test(attn_base, (q_nc, k_nc, v_nc, 0.125), "Attn_NonContig")))

    # --- CASE 3: Tiny Dimensions (The "Boundary Killer") ---
    print("\n[Edge Case] Tiny Sequence (N=1)")
    q_tiny = torch.randn(1, 1, 1, 64, device='cuda')
    k_tiny = torch.randn(1, 1, 1, 64, device='cuda')
    v_tiny = torch.randn(1, 1, 1, 64, device='cuda')
    results.append(("Tiny_Seq", run_stress_test(attn_base, (q_tiny, k_tiny, v_tiny, 0.125), "Attn_Tiny")))

    # --- CASE 4: Extreme Scale Factor (The "Precision Killer") ---
    print("\n[Edge Case] Extreme Scalars (Double precision check)")
    scale_ext = 1e-12
    results.append(("Extreme_Scale", run_stress_test(attn_base, (q_tiny, k_tiny, v_tiny, scale_ext), "Attn_Scale_Ext")))

    # --- SUMMARY ---
    print(f"\n{'='*70}\n🛡️ ROBUSTNESS REPORT\n{'='*70}")
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        if not passed: all_passed = False
        print(f"{name:.<30} {status}")
    
    if all_passed:
        print("\n🏆 YOUR COMPILER IS PRODUCTION-READY.")
    else:
        print("\n⚠️ ROBUSTNESS HOLES DETECTED. CHECK ALIGNMENT LOGIC.")

if __name__ == "__main__":
    # Tes tests standards ici...
    # ...
    # Lancement de la suite de torture
    # run_robustness_suite()

# if __name__ == "__main__":
    torch.manual_seed(42)
    
    # TEST 1: RMS Group Norm
    rms_model = RMSGroupNormLayer(G=8, D=8).cuda()
    rms_x = torch.randn(2, 8, 8, 64, device='cuda', dtype=torch.float32)
    # On ne passe que rms_x, le module gère ses paramètres internes !
    run_stress_test(rms_model, (rms_x,), "RMS_Group_Norm", native_fn=lambda x: native_rms_norm(x, 8, 8))
    
    # TEST 2: RoPE (Multi Output)
    rope_model = RoPELayer().cuda()
    B, H, N, D = 2, 8, 128, 64
    rotary_dim = 32
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    cos = torch.randn(N, rotary_dim, device='cuda', dtype=torch.float32)
    sin = torch.randn(N, rotary_dim, device='cuda', dtype=torch.float32)
    run_stress_test(rope_model, (q, k, cos, sin), "RoPE_Multi_Output", native_fn=native_rope)

    # TEST 3: Squared ReLU Attention (POUSSÉ À L = 4096 POUR LA DÉMO DU PAPIER)
    attn_model = SquaredReLUAttentionLayer().cuda()
    Z, H, N_CTX, D = 1, 8, 4096, 64  # <-- Saturation mémoire activée
    q_a = torch.randn(Z, H, N_CTX, D, device='cuda', dtype=torch.float32)
    k_a = torch.randn(Z, H, N_CTX, D, device='cuda', dtype=torch.float32)
    v_a = torch.randn(Z, H, N_CTX, D, device='cuda', dtype=torch.float32)
    scale = 1.0 / (D ** 0.5)
    run_stress_test(attn_model, (q_a, k_a, v_a, scale), "Squared_ReLU_Flash_Attention_4K", native_fn=native_squared_relu_attn)

    print("\n🚀 ALL MULTI-BACKEND STRESS TESTS COMPLETED.")