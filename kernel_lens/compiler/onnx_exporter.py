import triton
import inspect
import torch
from unittest.mock import patch

class TritonGlobalONNXExporter:
    def __init__(self, manifests):
        self.manifest_map = {m.kernel_name: m for m in manifests}
        self.patches = []

    def __enter__(self):
        orig_getitem = triton.JITFunction.__getitem__

        def patched_getitem(jit_self, grid):
            kernel_name = jit_self.fn.__name__ 
            
            def wrapper(*args, **kwargs):
                sig = inspect.signature(jit_self.fn)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                
                meta_kwargs = {k: v for k, v in bound.arguments.items()}
                evaluated_grid = grid(meta_kwargs) if callable(grid) else grid
                
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
                
                if kernel_name in self.manifest_map:
                    manifest = self.manifest_map[kernel_name]
                    
                    # 1. RUN THE REAL KERNEL FIRST
                    # This populates the output memory with real math so the graph has valid side-effects
                    orig_getitem(jit_self, evaluated_grid)(*clean_args, **clean_kwargs)
                    
                    # 2. PREPARE ARGS FOR ONNX TRACING
                    final_args = []
                    for arg_def in manifest.arguments:
                        val = bound.arguments[arg_def.name]
                        if isinstance(val, torch.SymInt):
                            val = unwrap(val)
                        
                        # CRITICAL FIX: Convert all scalars to Tensors!
                        # ONNX `g.op` ignores raw Python integers. They must be Tensors to enter the graph.
                        if isinstance(val, (int, float, bool)):
                            dtype = torch.float32 if isinstance(val, float) else torch.int64
                            val = torch.tensor([val], dtype=dtype, device='cuda' if torch.cuda.is_available() else 'cpu')
                            
                        final_args.append(val)
                        
                    # 3. INJECT CUSTOM ONNX NODE
                    class TritonONNXNode(torch.autograd.Function):
                        @staticmethod
                        def forward(ctx, *inputs):
                            # Return clones to establish a new tracked node boundary in the graph
                            outs = [inputs[i].clone() for i, a in enumerate(manifest.arguments) if a.kind == 'output']
                            if not outs: return inputs[0].clone()
                            return outs[0] if len(outs) == 1 else tuple(outs)

                        @staticmethod
                        def symbolic(g, *inputs):
                            out_count = max(1, len([a for a in manifest.arguments if a.kind == 'output']))
                            return g.op(f"triton_custom::{kernel_name}", *inputs, outputs=out_count)

                    dynamic_anchor = None
                    for a in final_args:
                        if isinstance(a, torch.Tensor) and not isinstance(a, torch.nn.Parameter):
                            dynamic_anchor = a
                            break
                            
                    # 2. Defeat TensorRT's static weight folding
                    if dynamic_anchor is not None:
                        # Create a computationally free scalar 0.0 with a rigid DAG dependency
                        zero_scalar = dynamic_anchor.reshape(-1)[0] * 0.0
                        secured_args = []
                        for a in final_args:
                            if isinstance(a, torch.nn.Parameter):
                                # Force TRT to treat the parameter as a dynamic execution buffer
                                secured_args.append(a + zero_scalar.to(a.dtype))
                            else:
                                secured_args.append(a)
                        final_args = secured_args

                    res = TritonONNXNode.apply(*final_args)
                    
                    # 4. WIRE THE GRAPH TOGETHER
                    out_idx = [i for i, a in enumerate(manifest.arguments) if a.kind == 'output']
                    if out_idx:
                        if len(out_idx) == 1:
                            bound.arguments[manifest.arguments[out_idx[0]].name].copy_(res)
                        else:
                            for i, idx in enumerate(out_idx):
                                bound.arguments[manifest.arguments[idx].name].copy_(res[i])
                                
                    # CRITICAL FIX: Return the tracked tensor so PyTorch connects the graph!
                    return res
                
                # Untraced Fallback (Normal Python execution)
                orig_getitem(jit_self, evaluated_grid)(*clean_args, **clean_kwargs)
                return None
                
            return wrapper

        p = patch('triton.JITFunction.__getitem__', new=patched_getitem)
        p.start()
        self.patches.append(p)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in self.patches:
            p.stop()