import inspect
import re
import torch
import triton
import triton.language as tl
from unittest.mock import patch
from typing import Any, List, Tuple
from torch.fx.experimental.proxy_tensor import make_fx

# Import our robust manifest definitions
from .manifest import KernelManifest, ArgumentDef

# Global storage for the context manager
_CAPTURED_MANIFESTS: List[KernelManifest] = []

class TritonSymIntTracingContext:
    """
    Context manager that patches Triton JIT invocation mechanics
    to capture the launch grid, arguments, and the resulting PTX.
    """
    def __init__(self):
        self.patches = []

    def __enter__(self):
        _CAPTURED_MANIFESTS.clear()
        original_run = triton.JITFunction.run

        def patched_run(jit_self, *args, **kwargs):
            sig = inspect.signature(jit_self.fn)
            triton_args = {'grid', 'num_warps', 'num_stages', 'num_ctas', 'enable_warp_illusions', 'cluster_dims', 'stream', 'device', 'warmup'}
            clean_kwargs = {k: v for k, v in kwargs.items() if k not in triton_args}
            bound_args = sig.bind(*args, **clean_kwargs)
            bound_args.apply_defaults()
            
            meta_kwargs = {name: value for name, value in bound_args.arguments.items()}

            grid_arg = kwargs.get('grid', (1, 1, 1))
            evaluated_grid = grid_arg(meta_kwargs) if callable(grid_arg) else grid_arg

            # --- AGGRESSIVE UNWRAP FOR TRITON COMPILER ---
            def unwrap(val):
                if isinstance(val, torch.SymInt):
                    hint = getattr(val.node, 'hint', 1)
                    if hint is None and hasattr(val.node.shape_env, 'size_hint'):
                        hint = val.node.shape_env.size_hint(val.node.expr)
                    return hint or 1
                if isinstance(val, torch.Tensor) and val.dim() == 0:
                    return int(val.item()) if val.dtype in [torch.int32, torch.int64] else float(val.item())
                return val

            clean_args = [unwrap(a) for a in args]
            clean_kwargs = {k: unwrap(v) for k, v in kwargs.items()}

            result = None
            try:
                # Execute with the clean native types so Triton doesn't crash during symbolic tracing
                result = original_run(jit_self, *clean_args, **clean_kwargs)
            except Exception as e:
                # We expect symbolic tracing to sometimes fail real execution, we catch and bypass
                pass
                
            out_tensors = result if isinstance(result, (tuple, list)) else [result]
            
            ptx = ""
            shared_memory_bytes = 0
            num_warps = 0
            mangled_name = jit_self.fn.__name__
            
            # --- PTX EXTRACTION ---
            caches = getattr(jit_self, 'cache', getattr(jit_self, 'device_caches', {}))
            for key_or_dev, value in caches.items():
                candidates = []
                if isinstance(value, tuple) and len(value) > 0 and isinstance(value[0], dict):
                    candidates = value[0].values()
                elif isinstance(value, dict):
                    candidates = value.values()
                elif isinstance(value, list):
                    candidates = value
                else:
                    candidates = [value]
                    
                for compiled in candidates:
                    if hasattr(compiled, 'asm') and 'ptx' in compiled.asm:
                        ptx = compiled.asm['ptx']
                        
                        def get_meta(prop, default=0):
                            if hasattr(compiled, 'metadata') and hasattr(compiled.metadata, prop):
                                return getattr(compiled.metadata, prop)
                            if hasattr(compiled, prop):
                                return getattr(compiled, prop)
                            return default
                            
                        mangled_name = get_meta('name', mangled_name)
                        shared_memory_bytes = get_meta('shared', 0)
                        num_warps = get_meta('num_warps', 0)
                        
                        if not num_warps and ptx:
                            match = re.search(r'\.reqntid\s+(\d+)', ptx)
                            if match: num_warps = int(match.group(1)) // 32
                        break
                if ptx:
                    match = re.search(r'\.visible\s+\.entry\s+([a-zA-Z0-9_]+)\(', ptx)
                    if match: mangled_name = match.group(1)
                    break

            # --- ABI SIGNATURE GENERATION ---
            manifest_args = []
            for i, (name, value) in enumerate(bound_args.arguments.items()):
                # Skip tl.constexpr arguments as they are baked into the PTX
                if sig.parameters[name].annotation is getattr(tl, 'constexpr', None):
                    continue

                if isinstance(value, torch.Tensor):
                    manifest_args.append(ArgumentDef(name, "unknown", tuple(value.shape), str(value.dtype)))
                elif isinstance(value, (int, float, torch.SymInt, bool)):
                    concrete_val = value
                    if isinstance(value, torch.SymInt):
                        concrete_val = getattr(value.node, 'hint', 1)
                        if concrete_val is None and hasattr(value.node.shape_env, 'size_hint'):
                            concrete_val = value.node.shape_env.size_hint(value.node.expr)
                        concrete_val = concrete_val or 1
                    
                    manifest_args.append(ArgumentDef(name, "scalar", (), str(type(concrete_val).__name__), concrete_val, _sym_ast=value))
            
            manifest = KernelManifest(
                kernel_name=mangled_name,
                ptx=ptx,
                shared_memory_bytes=shared_memory_bytes,
                num_warps=num_warps,
                arguments=manifest_args,
                _sym_grid_asts=evaluated_grid,
                _sym_out_asts=out_tensors[0].shape if getattr(out_tensors[0], 'shape', None) else ()
            )
            _CAPTURED_MANIFESTS.append(manifest)
            
            return result

        p = patch('triton.JITFunction.run', new=patched_run)
        p.start()
        self.patches.append(p)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in self.patches:
            p.stop()


def extract_manifests(module: torch.nn.Module, dummy_inputs: Tuple[Any, ...]) -> List[KernelManifest]:
    """
    Runs a module forward pass to capture Triton kernels, extract their PTX, 
    and build a symbolic AST mapping of their inputs/outputs.
    """
    global _CAPTURED_MANIFESTS
    
    # PASS 1: Extract real native GPU payload (PTX byte arrays)
    with TritonSymIntTracingContext():
        module(*dummy_inputs)
        
    pass1_manifests = list(_CAPTURED_MANIFESTS)
    _CAPTURED_MANIFESTS.clear()
    
    # PASS 2: Extract algebraic shape mapping via symbolic Make FX tracing
    with TritonSymIntTracingContext():
        try:
            make_fx(module, tracing_mode="symbolic", _allow_non_fake_inputs=True)(*dummy_inputs)
        except Exception:
            # make_fx tracing frequently orphans nodes dynamically, we expect it to crash occasionally
            pass
            
    pass2_manifests = list(_CAPTURED_MANIFESTS)
    
    # MERGE: Combine the hard PTX from Pass 1 with the SymInt ASTs from Pass 2
    merged = []
    for m1, m2 in zip(pass1_manifests, pass2_manifests):
        m1._sym_grid_asts = m2._sym_grid_asts
        for a1, a2 in zip(m1.arguments, m2.arguments):
            a1.shape = a2.shape
            a1._sym_ast = a2._sym_ast
        merged.append(m1)
        
    _CAPTURED_MANIFESTS.clear()
    return merged