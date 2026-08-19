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

            # print(f"\n[TRACER DEBUG] Launching Kernel: {jit_self.fn.__name__}")
            # tensor_map = {}
            # for name, value in bound_args.arguments.items():
            #     if isinstance(value, torch.Tensor):
            #         ptr = value.data_ptr()
            #         print(f"  -> {name}: Ptr={ptr}, Shape={list(value.shape)}, Stride={list(value.stride())}")
                    
            #         # Check for Aliasing
            #         if ptr in tensor_map:
            #             print(f"  ⚠️ ALERT: '{name}' is ALIASING '{tensor_map[ptr]}' (Same DataPtr!)")
            #         tensor_map[ptr] = name
            
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
            constexpr_arg_indices = set()
            for key_or_dev, value in reversed(list(caches.items())):
                items_to_check = []
                if isinstance(value, tuple) and len(value) > 0 and isinstance(value[0], dict):
                    items_to_check = list(value[0].items())
                elif isinstance(value, dict):
                    items_to_check = list(value.items())
                    
                for key_tuple, compiled in reversed(items_to_check):
                    if hasattr(compiled, 'asm') and 'ptx' in compiled.asm:
                        ptx = compiled.asm['ptx']
                        
                        if isinstance(key_tuple, str) and key_tuple.startswith('['):
                            import ast
                            try:
                                clean_str = key_tuple[:key_tuple.rfind(']')+1]
                                parsed_key = ast.literal_eval(clean_str)
                                for idx, arg_spec in enumerate(parsed_key):
                                    if isinstance(arg_spec, tuple) and len(arg_spec) > 0 and arg_spec[0] == 'constexpr':
                                        constexpr_arg_indices.add(idx)
                                    elif isinstance(arg_spec, str) and arg_spec == 'constexpr':
                                        constexpr_arg_indices.add(idx)
                            except Exception as e:
                                print(f"[TRACER DEBUG ERROR] ast.literal_eval failed: {e}")
                        elif isinstance(key_tuple, (tuple, list)):
                            for idx, arg_spec in enumerate(key_tuple):
                                if isinstance(arg_spec, tuple) and len(arg_spec) > 0 and arg_spec[0] == 'constexpr':
                                    constexpr_arg_indices.add(idx)
                                elif isinstance(arg_spec, str) and arg_spec == 'constexpr':
                                    constexpr_arg_indices.add(idx)
                        
                        if hasattr(compiled, 'src'):
                            if hasattr(compiled.src, 'constants') and isinstance(compiled.src.constants, dict):
                                for c_key in compiled.src.constants.keys():
                                    if isinstance(c_key, (tuple, list)) and len(c_key) > 0:
                                        constexpr_arg_indices.add(c_key[0])
                                    elif isinstance(c_key, int):
                                        constexpr_arg_indices.add(c_key)
                                    elif isinstance(c_key, str):
                                        for p_idx, p_name in enumerate(sig.parameters.keys()):
                                            if p_name == c_key:
                                                constexpr_arg_indices.add(p_idx)
                            if hasattr(compiled.src, 'signature') and isinstance(compiled.src.signature, dict):
                                for c_idx, (p_name, p_type) in enumerate(compiled.src.signature.items()):
                                    if p_type == 'constexpr':
                                        constexpr_arg_indices.add(c_idx)

                        def get_meta(prop, default=0):
                            if hasattr(compiled, 'metadata') and hasattr(compiled.metadata, prop):
                                return getattr(compiled.metadata, prop)
                            if hasattr(compiled, prop):
                                return getattr(compiled, prop)
                            return default
                            
                        mangled_name = get_meta('name', mangled_name)
                        num_warps = get_meta('num_warps', kwargs.get('num_warps', 4)) or 4
                        shared_memory_bytes = get_meta('shared', 0)
                        from ..config import is_verbose, debug_print
                        if is_verbose():
                            debug_print(f"[TRACER DEBUG] Captured shared_memory_bytes: {shared_memory_bytes}, num_warps: {num_warps}, constexpr_indices: {constexpr_arg_indices}")
                        
                        if ptx and num_warps == 4:
                            match = re.search(r'\.reqntid\s+(\d+)', ptx)
                            if match: num_warps = int(match.group(1)) // 32
                        if not num_warps:
                            num_warps = 4
                        break
                if ptx:
                    match = re.search(r'\.visible\s+\.entry\s+([a-zA-Z0-9_]+)\(', ptx)
                    if match: mangled_name = match.group(1)
                    break

            # --- ABI SIGNATURE GENERATION ---
            manifest_args = []
            for i, (name, value) in enumerate(bound_args.arguments.items()):
                # Skip tl.constexpr annotations
                if sig.parameters[name].annotation is getattr(tl, 'constexpr', None):
                    continue

                is_constexpr = (i in constexpr_arg_indices)
                if isinstance(value, torch.Tensor):
                    manifest_args.append(ArgumentDef(name, "unknown", tuple(value.shape), strides=tuple(value.stride()), dtype=str(value.dtype), is_constexpr=is_constexpr))
                elif isinstance(value, (int, float, torch.SymInt, bool)):
                    concrete_val = value
                    if isinstance(value, torch.SymInt):
                        concrete_val = getattr(value.node, 'hint', 1)
                        if concrete_val is None and hasattr(value.node.shape_env, 'size_hint'):
                            concrete_val = value.node.shape_env.size_hint(value.node.expr)
                        concrete_val = concrete_val or 1
                    
                    manifest_args.append(ArgumentDef(name, "scalar", (), (), str(type(concrete_val).__name__), concrete_val, _sym_ast=value, is_constexpr=is_constexpr))
            
            manifest = KernelManifest(
                kernel_name=mangled_name,
                ptx=ptx,
                shared_memory_bytes=shared_memory_bytes,
                num_warps=num_warps,
                arguments=manifest_args,
                _sym_grid_asts=evaluated_grid,
                _sym_out_asts=out_tensors[0].shape if getattr(out_tensors[0], 'shape', None) else (),
                fn=jit_self
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
    global _CAPTURED_MANIFESTS
    
    # PASS 1: Real Execution
    with TritonSymIntTracingContext():
        module(*dummy_inputs)
    pass1_manifests = list(_CAPTURED_MANIFESTS)
    _CAPTURED_MANIFESTS.clear()
    
    # PASS 2: Symbolic (Fake) Execution
    with TritonSymIntTracingContext():
        try:
            make_fx(module, tracing_mode="symbolic", _allow_non_fake_inputs=True)(*dummy_inputs)
        except Exception:
            pass
    pass2_manifests = list(_CAPTURED_MANIFESTS)
    
    from ..config import debug_print
    debug_print(f"\n[MANIFEST DEBUG] Pass 1 (Concrete Execution) Captured {len(pass1_manifests)} Manifest(s):")
    for i, m in enumerate(pass1_manifests):
        debug_print(f"  --- Pass 1 Manifest [{i}]: {m.kernel_name} ---")
        debug_print(f"      Shared Memory: {m.shared_memory_bytes} B | Warps: {m.num_warps} | Grid ASTs: {m._sym_grid_asts}")
        debug_print(f"      Arguments ({len(m.arguments)}):")
        for arg in m.arguments:
            debug_print(f"        • '{arg.name}': shape={arg.shape}, strides={arg.strides}, dtype={arg.dtype}, value={arg.value}, constexpr={arg.is_constexpr}, sym_ast={arg._sym_ast}")

    debug_print(f"\n[MANIFEST DEBUG] Pass 2 (Symbolic Fake Execution) Captured {len(pass2_manifests)} Manifest(s):")
    for i, m in enumerate(pass2_manifests):
        debug_print(f"  --- Pass 2 Manifest [{i}]: {m.kernel_name} ---")
        debug_print(f"      Shared Memory: {m.shared_memory_bytes} B | Warps: {m.num_warps} | Grid ASTs: {m._sym_grid_asts}")
        debug_print(f"      Arguments ({len(m.arguments)}):")
        for arg in m.arguments:
            debug_print(f"        • '{arg.name}': shape={arg.shape}, strides={arg.strides}, dtype={arg.dtype}, value={arg.value}, constexpr={arg.is_constexpr}, sym_ast={arg._sym_ast}")

    debug_print(f"\n[MERGE DEBUG] Merging Pass 1 (Concrete Shapes/PTX) + Pass 2 (Symbolic Grid ASTs)...")
    
    merged = []
    for i, (m1, m2) in enumerate(zip(pass1_manifests, pass2_manifests)):
        debug_print(f"  -> Merging Kernel {i} [{m1.kernel_name}]:")
        m1._sym_grid_asts = m2._sym_grid_asts
        
        for a1, a2 in zip(m1.arguments, m2.arguments):
            # LOG THE CONFLICT
            if a1.shape != a2.shape:
                debug_print(f"     ⚠️ SHAPE MISMATCH for '{a1.name}':")
                debug_print(f"        Pass 1 (Real): {a1.shape}")
                debug_print(f"        Pass 2 (Fake): {a2.shape} <--- THIS IS THE CULPRIT")
            
            # THE FIX: We keep Pass 1's shape but take Pass 2's AST logic
            a1._sym_ast = a2._sym_ast
            # a1.shape remains what it was in Pass 1
            
        merged.append(m1)
        
    _CAPTURED_MANIFESTS.clear()
    return merged

# def extract_manifests(module: torch.nn.Module, dummy_inputs: Tuple[Any, ...]) -> List[KernelManifest]:
#     """
#     Runs a module forward pass to capture Triton kernels, extract their PTX, 
#     and build a symbolic AST mapping of their inputs/outputs.
#     """
#     global _CAPTURED_MANIFESTS
    
#     # PASS 1: Extract real native GPU payload (PTX byte arrays)
#     with TritonSymIntTracingContext():
#         module(*dummy_inputs)
        
#     pass1_manifests = list(_CAPTURED_MANIFESTS)
#     _CAPTURED_MANIFESTS.clear()
    
#     # PASS 2: Extract algebraic shape mapping via symbolic Make FX tracing
#     with TritonSymIntTracingContext():
#         try:
#             make_fx(module, tracing_mode="symbolic", _allow_non_fake_inputs=True)(*dummy_inputs)
#         except Exception:
#             # make_fx tracing frequently orphans nodes dynamically, we expect it to crash occasionally
#             pass
            
#     pass2_manifests = list(_CAPTURED_MANIFESTS)
    
#     # MERGE: Combine the hard PTX from Pass 1 with the SymInt ASTs from Pass 2
#     # merged = []
#     # for m1, m2 in zip(pass1_manifests, pass2_manifests):
#     #     m1._sym_grid_asts = m2._sym_grid_asts
#     #     for a1, a2 in zip(m1.arguments, m2.arguments):
#     #         a1.shape = a2.shape
#     #         a1._sym_ast = a2._sym_ast
#     #     merged.append(m1)
#     # MERGE: Combine the hard PTX from Pass 1 with the SymInt ASTs from Pass 2
#     merged = []
#     for m1, m2 in zip(pass1_manifests, pass2_manifests):
#         m1._sym_grid_asts = m2._sym_grid_asts
        
#         # We MUST prioritize the shapes from Pass 1 (The REAL run)
#         # Pass 2 (make_fx) often guesses wrong shapes for custom ops
#         for a1, a2 in zip(m1.arguments, m2.arguments):
#             # a1 has the shape from the REAL GPU tensors [1, 512, 64, 64]
#             # a2 has the shape from the Fake FX tensors [1, 128, 64, 64]
            
#             # FORCE the real shape into the manifest
#             a1._sym_ast = a2._sym_ast 
#             # We keep a1.shape as it was captured in Pass 1!
            
#         merged.append(m1)
        
#     _CAPTURED_MANIFESTS.clear()
#     return merged