"""
Conformance suite: PyTorch module patterns that call Triton kernels.

Each case is compiled with KernelLens (ONNX Runtime and TensorRT) and compared
against native eager execution of the same kernels. The point is breadth: two
kernels in one graph, the same kernel called twice, multiple outputs, dynamic
scalars, strided inputs, loops, conditionals, nested helpers, autotune, dtypes,
and dynamic shapes.

Run:  python3 tests/test_conformance.py
"""
import os
import traceback
import torch
import torch.nn as nn
import triton
import triton.language as tl
import kernel_lens as kl


def maxdiff(a, b):
    if isinstance(a, (tuple, list)):
        return max(maxdiff(x, y) for x, y in zip(a, b))
    return (a.float() - b.float()).abs().max().item()


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------
@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m), mask=m)


@triton.jit
def scale_kernel(x_ptr, o_ptr, n, alpha, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, tl.load(x_ptr + off, mask=m) * alpha, mask=m)


@triton.jit
def mul_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, tl.load(x_ptr + off, mask=m) * 3.0, mask=m)


@triton.jit
def _helper(v):
    return v * v + 1.0


@triton.jit
def nested_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, _helper(tl.load(x_ptr + off, mask=m)), mask=m)


@triton.jit
def inplace_kernel(x_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(x_ptr + off, tl.load(x_ptr + off, mask=m) + 5.0, mask=m)


@triton.jit
def two_out_kernel(x_ptr, o1_ptr, o2_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    v = tl.load(x_ptr + off, mask=m)
    tl.store(o1_ptr + off, v * 2.0, mask=m)
    tl.store(o2_ptr + off, v + 1.0, mask=m)


@triton.jit
def grid3d_kernel(x_ptr, o_ptr, M, N, K, BLOCK: tl.constexpr):
    i, j, k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    idx = (i * N + j) * K + k
    tl.store(o_ptr + idx, tl.load(x_ptr + idx) * 2.0)


@triton.jit
def autotuned_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, tl.load(x_ptr + off, mask=m) * 4.0, mask=m)


# --------------------------------------------------------------------------
# module patterns
# --------------------------------------------------------------------------
class TwoKernels(nn.Module):
    def forward(self, x, y):
        n = x.numel()
        a = torch.empty_like(x)
        add_kernel[(triton.cdiv(n, 256),)](x, y, a, n, BLOCK=256)
        b = torch.empty_like(a)
        mul_kernel[(triton.cdiv(n, 256),)](a, b, n, BLOCK=256)
        return b


class SameKernelTwice(nn.Module):
    def forward(self, x):
        n = x.numel()
        a = torch.empty_like(x)
        scale_kernel[(triton.cdiv(n, 128),)](x, a, n, 2.0, BLOCK=128)
        b = torch.empty_like(a)
        scale_kernel[(triton.cdiv(n, 128),)](a, b, n, 3.0, BLOCK=128)
        return b


class TwoOutputs(nn.Module):
    def forward(self, x):
        n = x.numel()
        o1 = torch.empty_like(x)
        o2 = torch.empty_like(x)
        two_out_kernel[(triton.cdiv(n, 256),)](x, o1, o2, n, BLOCK=256)
        return o1, o2


class DynamicScalar(nn.Module):
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        n = x.numel()
        o = torch.empty_like(x)
        scale_kernel[(triton.cdiv(n, 128),)](x, o, n, self.alpha, BLOCK=128)
        return o


class Strided(nn.Module):
    def forward(self, x):
        xc = x.contiguous()
        n = xc.numel()
        o = torch.empty_like(xc)
        scale_kernel[(triton.cdiv(n, 128),)](xc, o, n, 2.0, BLOCK=128)
        return o


class Loop(nn.Module):
    def forward(self, x):
        n = x.numel()
        cur = x
        for _ in range(3):
            o = torch.empty_like(cur)
            scale_kernel[(triton.cdiv(n, 128),)](cur, o, n, 1.5, BLOCK=128)
            cur = o
        return cur


class Conditional(nn.Module):
    def forward(self, x):
        n = x.numel()
        o = torch.empty_like(x)
        scale_kernel[(triton.cdiv(n, 128),)](x, o, n, 2.0, BLOCK=128)
        if x.shape[0] > 1:
            o2 = torch.empty_like(o)
            mul_kernel[(triton.cdiv(n, 256),)](o, o2, n, BLOCK=256)
            return o2
        return o


class NestedHelper(nn.Module):
    def forward(self, x):
        n = x.numel()
        o = torch.empty_like(x)
        nested_kernel[(triton.cdiv(n, 256),)](x, o, n, BLOCK=256)
        return o


class Inplace(nn.Module):
    def forward(self, x):
        n = x.numel()
        inplace_kernel[(triton.cdiv(n, 128),)](x, n, BLOCK=128)
        return x


class Grid3D(nn.Module):
    def forward(self, x):
        M, N, K = x.shape
        o = torch.empty_like(x)
        grid3d_kernel[(M, N, K)](x, o, M, N, K, BLOCK=1)
        return o


class Autotuned(nn.Module):
    def forward(self, x):
        n = x.numel()
        o = torch.empty_like(x)
        autotuned_kernel[(triton.cdiv(n, 256),)](x, o, n, BLOCK=256)
        return o


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
def _clone(inputs):
    return tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in inputs)


def run_case(name, model, make_inputs, backends=("onnx", "tensorrt")):
    try:
        with torch.no_grad():
            base = make_inputs()
            ref = model(*_clone(base))
            cm = kl.compile(model, _clone(base), name=f"conf_{name}", backends=list(backends))
            results = {}
            for be in backends:
                out = cm.run(_clone(base), backend=be)
                results[be] = maxdiff(ref, out)
        ok = all(v < 2e-2 for v in results.values())
        detail = " ".join(f"{k}={v:.2e}" for k, v in results.items())
        print(f"[{'PASS' if ok else 'FAIL'}] {name:22} {detail}")
        return ok
    except Exception as e:
        print(f"[ERROR] {name:22} {type(e).__name__}: {str(e)[:160]}")
        if os.environ.get("KL_TRACE"):
            traceback.print_exc()
        return False


def run_dynamic_sweep(name, model, make_inputs, compile_batch, batches, backends=("onnx", "tensorrt")):
    """Compile once, then serve every batch size through that single engine.

    A shape-specialised engine cannot do this, so a bit-exact result at a batch
    other than the compiled one is the evidence that the launch grid and the
    shape-derived scalars really are functions of the runtime input.
    ``compile_batch`` sits in the middle of ``batches`` so the sweep goes both
    below and above the traced shape.
    """
    try:
        with torch.no_grad():
            # Build each batch's inputs ONCE. Calling make_inputs twice would
            # compare the engine against a different random tensor.
            data = {b: make_inputs(b) for b in batches}
            refs = {b: model(*_clone(data[b])) for b in batches}
            cm = kl.compile(model, make_inputs(compile_batch), name=f"dyn_{name}",
                            backends=list(backends), dynamic_shapes=True)
            results = {}
            for be in backends:
                worst, bad = 0.0, None
                for b in batches:
                    out = cm.run(_clone(data[b]), backend=be)
                    got = tuple(out.shape) if isinstance(out, torch.Tensor) else tuple(out[0].shape)
                    want = tuple(refs[b].shape)
                    if os.environ.get("KL_SWEEP_DBG"):
                        print(f"      [sweep] {be} batch={b} got={got} want={want} "
                              f"maxdiff={maxdiff(refs[b], out):.3e}", flush=True)
                    if got != want:
                        bad = f"batch {b}: shape {got} != {want}"
                        break
                    worst = max(worst, maxdiff(refs[b], out))
                results[be] = (worst, bad)
        ok = all(bad is None and v < 2e-2 for v, bad in results.values())
        detail = " ".join(
            f"{k}={'SHAPE ' + str(bad) if bad else f'{v:.2e}'}" for k, (v, bad) in results.items()
        )
        print(f"[{'PASS' if ok else 'FAIL'}] {name:22} compiled@{compile_batch} served={list(batches)} {detail}")
        return ok
    except Exception as e:
        print(f"[ERROR] {name:22} {type(e).__name__}: {str(e)[:180]}")
        if os.environ.get("KL_TRACE"):
            traceback.print_exc()
        return False


def main():
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    f32 = lambda *s: torch.randn(*s, device="cuda", dtype=torch.float32)
    f16 = lambda *s: torch.randn(*s, device="cuda", dtype=torch.float16)
    bf16 = lambda *s: torch.randn(*s, device="cuda", dtype=torch.bfloat16)

    cases = [
        ("two_kernels", TwoKernels(), lambda: (f32(4096), f32(4096)), ("onnx", "tensorrt")),
        ("same_kernel_twice", SameKernelTwice(), lambda: (f32(4096),), ("onnx", "tensorrt")),
        ("two_outputs", TwoOutputs(), lambda: (f32(4096),), ("onnx", "tensorrt")),
        ("dynamic_scalar", DynamicScalar(2.5), lambda: (f32(4096),), ("onnx", "tensorrt")),
        ("strided", Strided(), lambda: (f32(64, 64),), ("onnx", "tensorrt")),
        ("loop", Loop(), lambda: (f32(4096),), ("onnx", "tensorrt")),
        ("conditional", Conditional(), lambda: (f32(4, 1024),), ("onnx", "tensorrt")),
        ("nested_helper", NestedHelper(), lambda: (f32(4096),), ("onnx", "tensorrt")),
        ("inplace", Inplace(), lambda: (f32(4096),), ("onnx", "tensorrt")),
        ("grid3d", Grid3D(), lambda: (f32(8, 8, 8),), ("onnx", "tensorrt")),
        ("autotuned", Autotuned(), lambda: (f32(4096),), ("onnx", "tensorrt")),
        ("fp16", SameKernelTwice(), lambda: (f16(4096),), ("onnx", "tensorrt")),
        # ONNX Runtime 1.30 cannot bind bfloat16 custom-op tensors; TensorRT can.
        ("bf16 (TRT only)", SameKernelTwice(), lambda: (bf16(4096),), ("tensorrt",)),
        ("dynamic_batch", TwoKernels(), lambda: (f32(8, 512), f32(8, 512)), ("onnx", "tensorrt")),
    ]

    # dynamic_shapes=True is accepted only when the kernel's launch geometry is
    # a function of the input dims. For these kernels it is evaluated while
    # tracing, so the compiler must refuse rather than serve a wrong batch.
    dynamic_cases = [
        ("dynamic_batch_sweep", TwoKernels(),
         lambda b: (f32(b, 512), f32(b, 512)), 8, (1, 2, 4, 8, 16), ("onnx", "tensorrt")),
        ("dynamic_grid_sweep", SameKernelTwice(),
         lambda b: (f32(b * 256),), 2048, (256, 512, 2048, 4096), ("onnx", "tensorrt")),
    ]

    results = []
    for name, model, mk, backs in cases:
        results.append((name, run_case(name, model.cuda().eval(), mk, backends=backs)))

    for name, model, mk, cb, bs, backs in dynamic_cases:
        results.append((name, run_dynamic_sweep(name, model.cuda().eval(), mk, cb, bs, backends=backs)))

    npass = sum(1 for _, ok in results if ok)
    print(f"\n=== {npass}/{len(results)} conformance cases passed ===")
    for name, ok in results:
        if not ok:
            print("  failed:", name)


if __name__ == "__main__":
    main()