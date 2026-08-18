import triton
import inspect
import torch
from unittest.mock import patch
_ONNX_NODE_CACHE = {}

def get_onnx_node_class(kernel_name, manifest):
    if kernel_name in _ONNX_NODE_CACHE:
        return _ONNX_NODE_CACHE[kernel_name]
        
    out_args = [a for a in manifest.arguments if a.kind == 'output']
    out_count = max(1, len(out_args))
    
    class TritonONNXNode(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            # args contain out_count target output tensors, followed by node_inputs
            target_outs = args[:out_count]
            if out_count > 1:
                return tuple(t.new_zeros(t.shape) for t in target_outs)
            else:
                return target_outs[0].new_zeros(target_outs[0].shape)

        @staticmethod
        def symbolic(g, *args):
            target_outs = args[:out_count]
            node_inputs = args[out_count:]
            res = g.op(f"triton_custom::{kernel_name}", *node_inputs, outputs=out_count)
            if out_count > 1:
                for i, r in enumerate(res):
                    r.setType(target_outs[i].type())
                return res
            else:
                res.setType(target_outs[0].type())
                return res

    _ONNX_NODE_CACHE[kernel_name] = TritonONNXNode
    return TritonONNXNode

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
                    orig_getitem(jit_self, evaluated_grid)(*clean_args, **clean_kwargs)
                    
                    # 2. PREPARE ARGS FOR ONNX TRACING
                    from ..config import is_verbose
                    if is_verbose():
                        print(f"[ONNX EXPORT DEBUG] manifest.arguments:")
                        for idx, a in enumerate(manifest.arguments):
                            print(f"  arg {idx}: name='{a.name}', kind='{a.kind}', shape={a.shape}, dtype={a.dtype}")
                    
                    node_inputs = []
                    for arg_def in manifest.arguments:
                        if arg_def.kind == 'output':
                            continue
                        val = bound.arguments[arg_def.name]
                        if isinstance(val, torch.SymInt):
                            val = unwrap(val)
                        
                        target_device = args[0].device if (args and isinstance(args[0], torch.Tensor)) else 'cuda'
                        if isinstance(val, torch.Tensor) and val.dim() == 0:
                            dtype = torch.float32 if val.dtype in [torch.float32, torch.float64] else torch.int64
                            val = val.to(device=target_device, dtype=dtype).unsqueeze(0)
                        elif isinstance(val, (int, float, bool)):
                            dtype = torch.float32 if isinstance(val, float) else torch.int64
                            val = torch.tensor([val], dtype=dtype, device=target_device)
                            
                        node_inputs.append(val)
                    
                    self.saved_args = node_inputs
                        
                    target_outs = [bound.arguments[a.name] for a in manifest.arguments if a.kind == 'output']
                    if not target_outs:
                        target_outs = [args[0]]
                    ONNXNode = get_onnx_node_class(kernel_name, manifest)
                    res = ONNXNode.apply(*target_outs, *node_inputs)
                    if is_verbose():
                        print(f"[ONNX EXPORT DEBUG] res shape: {res.shape if isinstance(res, torch.Tensor) else [r.shape for r in res]}")
                    
                    # 4. WIRE THE GRAPH TOGETHER
                    out_idx = [i for i, a in enumerate(manifest.arguments) if a.kind == 'output']
                    if out_idx:
                        if len(out_idx) == 1:
                            target_out = bound.arguments[manifest.arguments[out_idx[0]].name]
                            print(f"[ONNX EXPORT DEBUG] copying res (shape {res.shape}) to target_out (shape {target_out.shape})")
                            target_out.copy_(res)
                        else:
                            for i, idx in enumerate(out_idx):
                                target_out = bound.arguments[manifest.arguments[idx].name]
                                target_out.copy_(res[i])
                                
                    # CRITICAL FIX: Return the tracked tensor(s) so PyTorch connects the graph!
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