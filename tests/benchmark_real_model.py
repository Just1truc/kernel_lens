"""
Real pretrained model served through KernelLens.

OPT is used because its architecture (learned positional embeddings, no rotary
cache) is ONNX-exportable through the TorchScript path that KernelLens uses;
the RoPE-based families are not on transformers >= 5. Every LayerNorm in the
decoder is replaced by the project's own Triton RMSNorm kernel, so the model
calls a real @triton.jit kernel exactly as a production model would.

    python3 tests/benchmark_real_model.py --model facebook/opt-125m --seq 512
"""
import argparse
import json
import time

import torch
import torch.nn as nn
import triton
import triton.language as tl


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
    def __init__(self, dim: int, eps: float = 1e-6, dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor):
        orig = x.shape
        if x.dim() != 3:
            x = x.reshape(1, -1, orig[-1])
        B, M, N = x.shape
        out = torch.empty_like(x)
        grid = (B, M)
        rmsnorm_kernel[grid](
            x, self.weight, out,
            x.stride(0), x.stride(1),
            out.stride(0), out.stride(1),
            N=N, eps=self.eps, BLOCK_SIZE=triton.next_power_of_2(N)
        )
        return out.reshape(orig)


def swap_layernorms(module, dtype):
    """Replace every LayerNorm with the Triton RMSNorm, carrying the weight."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            dim = child.normalized_shape[0]
            new = RMSNormModule(dim, eps=float(child.eps), dtype=dtype)
            with torch.no_grad():
                new.weight.copy_(child.weight.detach().to(dtype))
            setattr(module, name, new)
            n += 1
        else:
            n += swap_layernorms(child, dtype)
    return n


class LogitsOnly(torch.nn.Module):
    """Return only logits; transformers 5.x otherwise returns a DynamicCache."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        return self.m(input_ids, use_cache=False).logits


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--backends", default="onnx,tensorrt")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dt = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}[args.dtype]

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dt, attn_implementation="eager").cuda().eval()

    n_swap = swap_layernorms(model, dt)
    model = model.to(dt)
    wrapped = LogitsOnly(model).cuda().eval()

    n_param = sum(p.numel() for p in model.parameters())
    print(f"loaded {args.model}: {n_param/1e9:.3f}B params, "
          f"{getattr(model.config,'num_hidden_layers','?')} layers, "
          f"{n_swap} LayerNorm -> Triton RMSNorm, dtype={args.dtype}")

    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (1, args.seq), device="cuda")

    with torch.no_grad():
        ref = wrapped(ids)
    print(f"reference logits {tuple(ref.shape)}")

    results = {"model": args.model, "params_B": n_param / 1e9,
               "triton_modules": n_swap, "dtype": args.dtype, "seq": args.seq}

    with torch.no_grad():
        results["eager_ms"] = bench(lambda: wrapped(ids), args.iters)
    print(f"eager          : {results['eager_ms']:.2f} ms")

    try:
        import kernel_lens as kl
        name = args.name or args.model.split("/")[-1].replace(".", "_")
        cm = kl.compile(wrapped, (ids,), name=name,
                        backends=args.backends.split(","))
        for b in args.backends.split(","):
            with torch.no_grad():
                got = cm.run((ids,), backend=b)
            g = got[0] if isinstance(got, (tuple, list)) else got
            diff = float((g.float() - ref.float()).abs().max())
            with torch.no_grad():
                ms = bench(lambda b=b: cm.run((ids,), backend=b), args.iters)
            results[f"kernel_lens_{b}_ms"] = ms
            results[f"kernel_lens_{b}_max_abs_diff"] = diff
            print(f"KernelLens {b:<9}: {ms:.2f} ms   max|diff|={diff:.3e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        results["kernel_lens_error"] = f"{type(e).__name__}: {e}"

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
