"""
Usefulness benchmark: folding alone vs kernels alone vs both.

Three ways to run the same block:

  A. FOLDED ONLY   -- express everything with standard operators and let the
                      graph compiler fold it. This is what you get if you cannot
                      (or do not) ship a custom kernel.
  B. KERNELS ONLY  -- run the custom Triton kernels, but execute the standard
                      layers unfused (eager), or fused by torch.compile. Note
                      that torch.compile does not reimplement the kernel: since
                      PyTorch 2.1 TorchDynamo captures a @triton.jit launch as a
                      triton_kernel_wrapper_mutation node (no graph break) and
                      Inductor emits a call to the user's kernel; it only fuses
                      the surrounding standard operators.
  C. BOTH          -- KernelLens: the engine folds the standard layers and runs
                      the Triton kernels as plugins, in one C++ graph.

The kernel wins where folding has to materialise (squared-ReLU attention is
O(N^2) if expressed with standard operators). Folding wins on the element-wise
chain around the projections. KernelLens is the only path that gets both.

Run:  python3 tests/benchmark_folding_vs_triton.py
"""
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import kernel_lens as kl


# --------------------------------------------------------------------------
# Custom Triton kernels
# --------------------------------------------------------------------------
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, o_ptr, sb, sm, N: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pb, pm = tl.program_id(0), tl.program_id(1)
    rx, ro = x_ptr + pb * sb + pm * sm, o_ptr + pb * sb + pm * sm
    c = tl.arange(0, BLOCK)
    m = c < N
    x = tl.load(rx + c, mask=m, other=0.0)
    w = tl.load(w_ptr + c, mask=m, other=1.0)
    v = tl.sum(x * x, axis=0) / N
    tl.store(ro + c, x * tl.rsqrt(v + eps) * w, mask=m)


@triton.jit
def swiglu_kernel(g_ptr, u_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    g = tl.load(g_ptr + off, mask=m, other=0.0)
    u = tl.load(u_ptr + off, mask=m, other=0.0)
    tl.store(o_ptr + off, (g * tl.sigmoid(g)) * u, mask=m)


@triton.jit
def sqrelu_attn_kernel(Q, K, V, O, scale: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr):
    pm, pz = tl.program_id(0), tl.program_id(1)
    base = pz * N * D
    off_m = pm * BM + tl.arange(0, BM)
    off_d = tl.arange(0, D)
    q = tl.load(Q + base + off_m[:, None] * D + off_d[None, :], mask=off_m[:, None] < N, other=0.0)
    acc = tl.zeros([BM, D], dtype=tl.float32)
    for n0 in range(0, N, BN):
        off_n = n0 + tl.arange(0, BN)
        k = tl.load(K + base + off_n[None, :] * D + off_d[:, None], mask=off_n[None, :] < N, other=0.0)
        v = tl.load(V + base + off_n[:, None] * D + off_d[None, :], mask=off_n[:, None] < N, other=0.0)
        s = tl.dot(q, k) * scale
        s = tl.maximum(s, 0.0)
        acc += tl.dot((s * s).to(tl.float32), v)
    tl.store(O + base + off_m[:, None] * D + off_d[None, :], acc, mask=off_m[:, None] < N)


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        B, M, N = x.shape
        o = torch.empty_like(x)
        rmsnorm_kernel[(B, M)](x, self.w, o, x.stride(0), x.stride(1), N=N, eps=1e-6,
                               BLOCK=triton.next_power_of_2(N))
        return o


class SwiGLU(nn.Module):
    def forward(self, g, u):
        o = torch.empty_like(g)
        n = g.numel()
        swiglu_kernel[(triton.cdiv(n, 1024),)](g, u, o, n, BLOCK=1024)
        return o


class Attn(nn.Module):
    def __init__(self, heads, dh, scale=0.125, custom=True):
        super().__init__()
        self.h, self.dh, self.scale, self.custom = heads, dh, scale, custom

    def forward(self, q, k, v):
        if self.custom:
            B, H, N, D = q.shape
            o = torch.empty_like(q)
            sqrelu_attn_kernel[(triton.cdiv(N, 64), B * H)](q, k, v, o, scale=self.scale,
                                                            N=N, D=D, BM=64, BN=64)
            return o
        s = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        s = torch.relu(s).pow(2)
        return torch.matmul(s, v)


# --------------------------------------------------------------------------
# Block, in a "custom" and a "standard" flavour
# --------------------------------------------------------------------------
def std_rmsnorm(x, eps=1e-6):
    v = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(v + eps)


class Block(nn.Module):
    def __init__(self, d, heads, ffn, custom=True):
        super().__init__()
        self.h, self.custom = heads, custom
        self.qkv = nn.Linear(d, 3 * d, bias=True)
        self.o = nn.Linear(d, d, bias=True)
        self.gate = nn.Linear(d, ffn, bias=True)
        self.up = nn.Linear(d, ffn, bias=True)
        self.down = nn.Linear(ffn, d, bias=True)
        self.attn = Attn(heads, d // heads, custom=custom)
        if custom:
            self.n1, self.n2, self.swiglu = RMSNorm(d), RMSNorm(d), SwiGLU()

    def forward(self, x):
        B, S, D = x.shape
        h = self.n1(x) if self.custom else std_rmsnorm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.h, D // self.h).transpose(1, 2).contiguous()
        k = k.view(B, S, self.h, D // self.h).transpose(1, 2).contiguous()
        v = v.view(B, S, self.h, D // self.h).transpose(1, 2).contiguous()
        a = self.attn(q, k, v).transpose(1, 2).reshape(B, S, D)
        # A standard, foldable epilogue chain. Folded engines collapse it into
        # the projection epilogue; an unfused path launches one kernel per op.
        for _ in range(8):
            a = a * 0.99 + 0.01
        x = x + self.o(a) * 1.1
        h2 = self.n2(x) if self.custom else std_rmsnorm(x)
        g, u = self.gate(h2), self.up(h2)
        act = (self.swiglu(g, u) if self.custom else F.silu(g) * u) * 0.9 + 0.05
        x = x + self.down(act) * 1.1
        return x


class Net(nn.Module):
    def __init__(self, layers, d, heads, ffn, custom=True):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d, heads, ffn, custom) for _ in range(layers)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def bench(fn, args=(), warmup=10, iters=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    L = int(os.environ.get("KL_L", 2)); D = int(os.environ.get("KL_D", 512))
    H = int(os.environ.get("KL_H", 8)); FFN = int(os.environ.get("KL_FFN", 2048))
    B = int(os.environ.get("KL_B", 1)); S = int(os.environ.get("KL_S", 4096))
    DT = os.environ.get("KL_DTYPE", "fp32")
    tdt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[DT]
    print(f"config: L={L} D={D} H={H} FFN={FFN} B={B} S={S} dtype={DT}")

    x = torch.randn(B, S, D, device="cuda", dtype=tdt)
    std = Net(L, D, H, FFN, custom=False).cuda().eval().to(tdt)
    cus = Net(L, D, H, FFN, custom=True).cuda().eval().to(tdt)
    # keep the two nets numerically aligned where it is easy (not required for timing)
    res = {}
    with torch.no_grad():
        res["A. Folded only (eager, standard ops)"] = bench(lambda: std(x))
        try:
            sc = torch.compile(std)
            for _ in range(3):
                sc(x)
            res["A. Folded only (torch.compile)"] = bench(lambda: sc(x))
        except Exception as ex:
            print("compile(std) failed:", ex)
        res["B. Kernels only (eager Triton)"] = bench(lambda: cus(x))
        try:
            cc = torch.compile(cus)
            for _ in range(3):
                cc(x)
            res["B. Kernels only (torch.compile)"] = bench(lambda: cc(x))
        except Exception as ex:
            print("compile(cus) failed:", ex)

        cm = kl.compile(cus, (x,), name="FoldVsTriton", backends=["onnx", "tensorrt"])
        res["C. Both (KernelLens ORT)"] = bench(lambda: cm.run((x,), backend="onnx"))
        res["C. Both (KernelLens TensorRT)"] = bench(lambda: cm.run((x,), backend="tensorrt"))

    base = res["B. Kernels only (eager Triton)"]
    print("\n--- Latency (ms) ---")
    for k, v in res.items():
        print(f"  {k:38}: {v:8.3f} ms  ({base / v:5.2f}x vs kernels-only)")
    with open("measured_folding_vs_triton.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()