import os
import torch
import torch.nn as nn
import triton
import triton.language as tl
import kernel_lens as kl


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


if __name__ == "__main__":
    x = torch.randn(4, 512, 4096, device="cuda")
    kl.compile(RMSNorm(4096).cuda(), (x,), name="LOC_RMSNorm", backends=["onnx", "tensorrt"])
    d = os.path.join(os.environ["KERNEL_LENS_CACHE_DIR"], "LOC_RMSNorm")
    total = 0
    files = []
    for root, _, names in os.walk(d):
        for n in names:
            if n.endswith((".cu", ".h", ".cpp")):
                p = os.path.join(root, n)
                lines = sum(1 for _ in open(p, errors="ignore"))
                total += lines
                files.append((lines, os.path.relpath(p, d)))
    for lines, name in sorted(files):
        print(f"{lines:6}  {name}")
    print("TOTAL", total, "lines across", len(files), "files")
    print("PYTHON (user code):", sum(1 for _ in open(__file__)) - 30, "lines for the module")